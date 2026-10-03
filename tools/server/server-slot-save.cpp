#include "server-slot-save.h"

extern "C" {
#include "sha256.h"
}

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <sddl.h>
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;
constexpr size_t header_size = 40;
constexpr size_t buffer_size = 64 * 1024;
constexpr unsigned char magic[8] = { 'L', 'L', 'S', 'L', 'O', 'T', 0, 0 };
struct file_closer {
    void operator()(FILE * file) const { std::fclose(file); }
};
using file_ptr = std::unique_ptr<FILE, file_closer>;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("slot save: " + message);
}

#ifdef _WIN32
struct handle_closer {
    void operator()(HANDLE handle) const { CloseHandle(handle); }
};

struct local_free {
    void operator()(void * pointer) const { LocalFree(pointer); }
};

std::unique_ptr<void, local_free> directory_security() {
    HANDLE raw_token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw_token)) {
        fail("cannot open process token for private staging");
    }
    std::unique_ptr<void, handle_closer> token(raw_token);
    DWORD size = 0;
    if (GetTokenInformation(token.get(), TokenUser, nullptr, 0, &size) || GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        fail("cannot measure process token");
    }
    std::vector<unsigned char> information(size);
    if (!GetTokenInformation(token.get(), TokenUser, information.data(), size, &size)) {
        fail("cannot read process token");
    }
    LPWSTR raw_sid = nullptr;
    if (!ConvertSidToStringSidW(reinterpret_cast<TOKEN_USER *>(information.data())->User.Sid, &raw_sid)) {
        fail("cannot read process user");
    }
    std::unique_ptr<wchar_t, local_free> sid(raw_sid);
    const std::wstring acl = L"D:P(A;OICI;FA;;;" + std::wstring(sid.get()) + L")";
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(acl.c_str(), SDDL_REVISION_1, &descriptor, nullptr)) {
        fail("cannot create private staging permissions");
    }
    return std::unique_ptr<void, local_free>(descriptor);
}
#endif

void check_hook(const server_slot_save::failure_hook & hook, const char * operation) {
    if (hook && hook(operation)) {
        fail(std::string("injected ") + operation + " failure");
    }
}

file_ptr open_file(const fs::path & path, bool write) {
#ifdef _WIN32
    // Readers must permit replacement of the committed path while they hold the old generation.
    HANDLE handle = CreateFileW(path.c_str(), write ? GENERIC_WRITE : GENERIC_READ,
                               FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                               write ? CREATE_NEW : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        const auto error = GetLastError();
        fail("cannot open " + path.u8string() + " (Windows error " + std::to_string(error) + ")");
    }
    int fd = _open_osfhandle(reinterpret_cast<intptr_t>(handle), _O_BINARY | (write ? _O_WRONLY : _O_RDONLY));
    if (fd == -1) {
        CloseHandle(handle);
        fail("cannot open stream");
    }
    FILE * raw = _fdopen(fd, write ? "wb" : "rb");
    if (!raw) {
        _close(fd);
    }
#else
    FILE * raw;
    if (write) {
        const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (fd == -1) {
            fail("cannot create " + path.u8string());
        }
        raw = fdopen(fd, "wb");
        if (!raw) {
            close(fd);
        }
    } else {
        raw = std::fopen(path.c_str(), "rb");
    }
#endif
    if (!raw) {
        fail("cannot open " + path.u8string());
    }
    return file_ptr(raw);
}

uint64_t file_length(FILE * file) {
#ifdef _WIN32
    if (_fseeki64(file, 0, SEEK_END) != 0) {
        fail("cannot seek");
    }
    const auto length = _ftelli64(file);
    if (length < 0 || _fseeki64(file, 0, SEEK_SET) != 0) {
#else
    if (fseeko(file, 0, SEEK_END) != 0) {
        fail("cannot seek");
    }
    const auto length = ftello(file);
    if (length < 0 || fseeko(file, 0, SEEK_SET) != 0) {
#endif
        fail("cannot measure file");
    }
    return static_cast<uint64_t>(length);
}

void read_exact(FILE * file, unsigned char * data, size_t size) {
    if (size && std::fread(data, 1, size, file) != size) {
        fail("truncated file or read error");
    }
}

void check_end(FILE * file) {
    if (std::fgetc(file) != EOF || std::ferror(file)) {
        fail("unexpected trailing data or read error");
    }
}

void write_exact(FILE * file, const unsigned char * data, size_t size, const server_slot_save::failure_hook & hook) {
    check_hook(hook, "write");
    if (std::fwrite(data, 1, size, file) != size) {
        fail("write error");
    }
}

void close_file(file_ptr & file, const server_slot_save::failure_hook & hook, bool write) {
    if (write) {
        check_hook(hook, "flush");
        if (std::fflush(file.get()) != 0) {
            fail("flush error");
        }
    }
    // Close even when the test hook reports an error.
    const bool injected = hook && hook("close");
    const int result = std::fclose(file.release());
    if (injected || result != 0) {
        fail("close error");
    }
}

void put_u64(unsigned char * data, uint64_t value, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i) {
        data[i] = static_cast<unsigned char>(value >> (8 * i));
    }
}

uint64_t get_u64(const unsigned char * data, size_t bytes) {
    uint64_t value = 0;
    for (size_t i = 0; i < bytes; ++i) {
        value |= uint64_t(data[i]) << (8 * i);
    }
    return value;
}

void copy_component(FILE * input, FILE * output, uint64_t length, sha256_t & hash, const server_slot_save::failure_hook & hook) {
    std::array<unsigned char, buffer_size> buffer;
    while (length) {
        const size_t count = length < buffer.size() ? static_cast<size_t>(length) : buffer.size();
        read_exact(input, buffer.data(), count);
        sha256_update(&hash, buffer.data(), count);
        write_exact(output, buffer.data(), count, hook);
        length -= count;
    }
}

} // namespace

struct server_slot_save::impl {
    fs::path filename;
    fs::path directory;
    fs::path container;
    std::array<std::string, 3> paths;
    std::array<bool, 3> present = {};
    std::array<uint64_t, 3> lengths = {};
    failure_hook hook;
    bool owns_directory = false;
    bool started = false;

    ~impl() {
        if (!owns_directory) {
            return;
        }
        try {
            std::error_code ec;
            for (const auto & path : paths) {
                fs::remove(fs::u8path(path), ec);
            }
            fs::remove(container, ec);
            // Never recursively delete a directory or remove a name we did not create.
            fs::remove(directory, ec);
        } catch (...) {
        }
    }

    void begin() {
        if (started) {
            fail("save object already used");
        }
        started = true;
    }
};

server_slot_save::server_slot_save(const std::string & filepath, failure_hook hook) : p(new impl) {
    p->filename = fs::u8path(filepath);
    p->hook = std::move(hook);
    const fs::path parent = p->filename.has_parent_path() ? p->filename.parent_path() : fs::path(".");
#ifdef _WIN32
    // Staged files inherit an owner-only ACL, matching the private POSIX directory and files.
    auto descriptor = directory_security();
    SECURITY_ATTRIBUTES security = { sizeof(SECURITY_ATTRIBUTES), descriptor.get(), FALSE };
#endif
    std::random_device random;
    for (size_t attempt = 0; attempt < 64; ++attempt) {
        std::ostringstream suffix;
        suffix << ".llama-slot-" << std::hex << std::setfill('0');
        for (size_t i = 0; i < 4; ++i) {
            suffix << std::setw(8) << static_cast<uint32_t>(random());
        }
        p->directory = parent / suffix.str();
        p->container = p->directory / "container";
        for (size_t i = 0; i < p->paths.size(); ++i) {
            p->paths[i] = (p->directory / std::to_string(i)).u8string();
        }
#ifdef _WIN32
        if (CreateDirectoryW(p->directory.c_str(), &security)) {
            p->owns_directory = true;
            return;
        }
        if (GetLastError() != ERROR_ALREADY_EXISTS) {
#else
        if (mkdir(p->directory.c_str(), 0700) == 0) {
            p->owns_directory = true;
            return;
        }
        if (errno != EEXIST) {
#endif
            fail("cannot create staging directory");
        }
    }
    fail("cannot create unique staging directory");
}

server_slot_save::~server_slot_save() = default;

const std::string & server_slot_save::path(size_t index) const {
    return p->paths.at(index);
}

bool server_slot_save::has(size_t index) const {
    return p->present.at(index);
}

uint64_t server_slot_save::size(size_t index) const {
    return p->lengths.at(index);
}

uint64_t server_slot_save::total_size() const {
    return p->present[KV] ? header_size + SHA256_DIGEST_SIZE + p->lengths[KV] + p->lengths[MTMD] + p->lengths[CKPT] : 0;
}

void server_slot_save::commit(bool mtmd_present, bool ckpt_present) {
    p->begin();
    const std::array<bool, 3> present = { true, mtmd_present, ckpt_present };
    std::array<uint64_t, 3> lengths = {};
    std::array<file_ptr, 3> inputs;
    for (size_t i = 0; i < present.size(); ++i) {
        if (present[i]) {
            inputs[i] = open_file(fs::u8path(p->paths[i]), false);
            lengths[i] = file_length(inputs[i].get());
        }
    }
    if (!lengths[KV]) {
        fail("missing KV state");
    }
    uint64_t total_size = header_size + SHA256_DIGEST_SIZE;
    for (const uint64_t length : lengths) {
        if (length > std::numeric_limits<uint64_t>::max() - total_size) {
            fail("invalid component length");
        }
        total_size += length;
    }

    std::array<unsigned char, header_size> header = {};
    std::memcpy(header.data(), magic, sizeof(magic));
    put_u64(header.data() + 8, 1, 4);
    put_u64(header.data() + 12, 1 | (mtmd_present ? 2 : 0) | (ckpt_present ? 4 : 0), 4);
    for (size_t i = 0; i < lengths.size(); ++i) {
        put_u64(header.data() + 16 + 8 * i, lengths[i], 8);
    }
    sha256_t hash;
    sha256_init(&hash);
    sha256_update(&hash, header.data(), header.size());
    auto output = open_file(p->container, true);
    write_exact(output.get(), header.data(), header.size(), p->hook);
    for (size_t i = 0; i < present.size(); ++i) {
        if (present[i]) {
            copy_component(inputs[i].get(), output.get(), lengths[i], hash, p->hook);
            check_end(inputs[i].get());
            close_file(inputs[i], p->hook, false);
        }
    }
    std::array<unsigned char, SHA256_DIGEST_SIZE> digest;
    sha256_final(&hash, digest.data());
    write_exact(output.get(), digest.data(), digest.size(), p->hook);
    close_file(output, p->hook, true);
    check_hook(p->hook, "rename");
#ifdef _WIN32
    if (!MoveFileExW(p->container.c_str(), p->filename.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        fail("cannot replace committed save (Windows error " + std::to_string(GetLastError()) + ")");
    }
#else
    if (std::rename(p->container.c_str(), p->filename.c_str()) != 0) {
        fail("cannot replace committed save");
    }
#endif
    p->present = present;
    p->lengths = lengths;
}

void server_slot_save::unpack() {
    p->begin();
    auto input = open_file(p->filename, false);
    const uint64_t file_size = file_length(input.get());
    std::array<unsigned char, header_size> header;
    read_exact(input.get(), header.data(), header.size());
    if (std::memcmp(header.data(), magic, sizeof(magic)) != 0 || get_u64(header.data() + 8, 4) != 1) {
        fail("unsupported save format; legacy sidecar saves must be saved again");
    }
    const uint64_t flags = get_u64(header.data() + 12, 4);
    if (!(flags & 1) || (flags & ~uint64_t(7))) {
        fail("invalid component flags");
    }
    std::array<bool, 3> present;
    std::array<uint64_t, 3> lengths;
    uint64_t expected_size = header_size + SHA256_DIGEST_SIZE;
    for (size_t i = 0; i < lengths.size(); ++i) {
        present[i] = (flags & (1 << i)) != 0;
        lengths[i] = get_u64(header.data() + 16 + 8 * i, 8);
        if ((!present[i] && lengths[i]) || lengths[i] > std::numeric_limits<uint64_t>::max() - expected_size) {
            fail("invalid component length");
        }
        expected_size += lengths[i];
    }
    if (!lengths[KV] || expected_size != file_size) {
        fail("invalid save length");
    }
    sha256_t hash;
    sha256_init(&hash);
    sha256_update(&hash, header.data(), header.size());
    for (size_t i = 0; i < present.size(); ++i) {
        if (present[i]) {
            auto output = open_file(fs::u8path(p->paths[i]), true);
            copy_component(input.get(), output.get(), lengths[i], hash, p->hook);
            close_file(output, p->hook, true);
        }
    }
    std::array<unsigned char, SHA256_DIGEST_SIZE> expected_digest;
    std::array<unsigned char, SHA256_DIGEST_SIZE> actual_digest;
    read_exact(input.get(), expected_digest.data(), expected_digest.size());
    check_end(input.get());
    sha256_final(&hash, actual_digest.data());
    if (actual_digest != expected_digest) {
        fail("content digest mismatch");
    }
    close_file(input, p->hook, false);
    p->present = present;
    p->lengths = lengths;
}
