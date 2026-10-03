#include "server-slot-save.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        std::abort(); \
    } \
} while (0)

namespace fs = std::filesystem;

namespace {

struct test_directory {
    fs::path path;

    test_directory() {
        std::random_device random;
        for (size_t i = 0; i < 64; ++i) {
            path = fs::temp_directory_path() / ("test-slot-save-" + std::to_string(random()) + "-" + std::to_string(random()));
            if (fs::create_directory(path)) {
                return;
            }
        }
        throw std::runtime_error("cannot create test directory");
    }

    ~test_directory() {
        std::error_code ec;
        for (const auto & entry : fs::directory_iterator(path)) {
            fs::remove(entry.path(), ec);
        }
        fs::remove(path, ec);
    }
};

void write_file(const fs::path & path, const std::string & bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), bytes.size());
    file.close();
    CHECK(file.good());
}

std::string read_file(const fs::path & path) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.good());
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void stage(server_slot_save & save, const std::array<std::string, 3> & parts, bool mtmd = true, bool ckpt = true) {
    write_file(fs::u8path(save.path(server_slot_save::KV)), parts[0]);
    if (mtmd) {
        write_file(fs::u8path(save.path(server_slot_save::MTMD)), parts[1]);
    }
    if (ckpt) {
        write_file(fs::u8path(save.path(server_slot_save::CKPT)), parts[2]);
    }
}

void commit(const fs::path & path, const std::array<std::string, 3> & parts, bool mtmd = true, bool ckpt = true) {
    server_slot_save save(path.u8string());
    stage(save, parts, mtmd, ckpt);
    save.commit(mtmd, ckpt);
    CHECK(save.total_size() == fs::file_size(path));
}

void check_parts(const fs::path & path, const std::array<std::string, 3> & parts, bool mtmd = true, bool ckpt = true) {
    server_slot_save save(path.u8string());
    save.unpack();
    const std::array<bool, 3> present = { true, mtmd, ckpt };
    for (size_t i = 0; i < present.size(); ++i) {
        CHECK(save.has(i) == present[i]);
        CHECK(save.size(i) == (present[i] ? parts[i].size() : 0));
        if (present[i]) {
            CHECK(read_file(fs::u8path(save.path(i))) == parts[i]);
        } else {
            CHECK(!fs::exists(fs::u8path(save.path(i))));
        }
    }
    CHECK(save.total_size() == fs::file_size(path));
}

template <typename Callable>
void rejected(Callable run) {
    bool failed = false;
    try {
        run();
    } catch (const std::runtime_error &) {
        failed = true;
    }
    CHECK(failed);
}

void reject_file(const fs::path & path, const std::string & bytes) {
    write_file(path, bytes);
    server_slot_save save(path.u8string());
    rejected([&] { save.unpack(); });
    CHECK(!save.has(0));
    CHECK(save.total_size() == 0);
}

void check_no_staging(const fs::path & directory) {
    for (const auto & entry : fs::directory_iterator(directory)) {
        CHECK(!entry.is_directory());
    }
}

void test_roundtrip_and_overwrite(const fs::path & directory) {
    const auto path = directory / "slot.bin";
    const std::array<std::string, 3> first = { "tokens=17,42; KV generation one", "image one", "checkpoint one" };
    const std::array<std::string, 3> second = { std::string(262177, 'k'), std::string(131079, 'm'), std::string(65541, 'c') };
    commit(path, first);
    check_parts(path, first);
#ifndef _WIN32
    struct stat status;
    CHECK(stat(path.c_str(), &status) == 0);
    CHECK((status.st_mode & 0777) == 0600);
#endif
    for (size_t i = 0; i < 8; ++i) {
        commit(path, second);
        check_parts(path, second);
#ifndef _WIN32
        CHECK(stat(path.c_str(), &status) == 0);
        CHECK((status.st_mode & 0777) == 0600);
#endif
        commit(path, first);
        check_parts(path, first);
    }
    for (bool mtmd : { false, true }) {
        for (bool ckpt : { false, true }) {
            const std::array<std::string, 3> empty = { "KV", "", "" };
            commit(path, empty, mtmd, ckpt);
            check_parts(path, empty, mtmd, ckpt);
        }
    }
    // Absent optional components are explicit, even if a writer left a staged file.
    {
        server_slot_save save(path.u8string());
        stage(save, first);
        save.commit(false, false);
    }
    check_parts(path, first, false, false);
    check_no_staging(directory);
}

void test_invalid_containers(const fs::path & directory) {
    const auto path = directory / "invalid.bin";
    const std::array<std::string, 3> first = { "tokens=17,42; KV generation one", "image one", "checkpoint one" };
    const std::array<std::string, 3> second = { "tokens=17,42; KV generation two", "image two", "checkpoint two" };
    commit(path, first);
    const std::string original = read_file(path);
    commit(path, second);
    const std::string other = read_file(path);
    CHECK(original.size() == other.size());

    // Every byte is bound, including the magic, version, flags, all lengths and the digest itself.
    for (size_t i = 0; i < original.size(); ++i) {
        auto corrupted = original;
        corrupted[i] ^= 1;
        reject_file(path, corrupted);
    }
    for (size_t length = 0; length < original.size(); ++length) {
        reject_file(path, original.substr(0, length));
    }
    reject_file(path, original + "unexpected");
    reject_file(path, "legacy unbound KV with external sidecars");

    size_t offset = 40;
    for (size_t i = 0; i < first.size(); ++i) {
        auto mixed = original;
        mixed.replace(offset, first[i].size(), other, offset, first[i].size());
        reject_file(path, mixed);
        offset += first[i].size();
    }
    auto mixed = original;
    mixed.replace(40, first[0].size() + first[1].size(), other, 40, first[0].size() + first[1].size());
    reject_file(path, mixed);

    auto overflow = original;
    std::fill(overflow.begin() + 16, overflow.begin() + 24, '\xff');
    reject_file(path, overflow);

    // A present-empty component is different from an absent component in the digest.
    commit(path, { "KV", "", "" }, true, false);
    auto empty_present = read_file(path);
    empty_present[12] ^= 2;
    reject_file(path, empty_present);
    check_no_staging(directory);
}

void test_missing_components_and_rename(const fs::path & directory) {
    const auto path = directory / "missing.bin";
    rejected([&] { server_slot_save save(path.u8string()); save.commit(false, false); });
    CHECK(!fs::exists(path));
    rejected([&] {
        server_slot_save save(path.u8string());
        stage(save, { "", "", "" }, false, false);
        save.commit(false, false);
    });
    CHECK(!fs::exists(path));
    rejected([&] {
        server_slot_save save(path.u8string());
        stage(save, { "KV", "", "" }, false, false);
        save.commit(true, false);
    });
    rejected([&] {
        server_slot_save save(path.u8string());
        stage(save, { "KV", "", "" }, false, false);
        save.commit(false, true);
    });
    CHECK(!fs::exists(path));

    // A real rename failure must not be reported as a successful save.
    const auto blocked = directory / "occupied";
    CHECK(fs::create_directory(blocked));
    write_file(blocked / "keep", "untouched");
    rejected([&] { commit(blocked, { "KV", "media", "checkpoint" }); });
    CHECK(read_file(blocked / "keep") == "untouched");
    CHECK(fs::remove(blocked / "keep"));
    CHECK(fs::remove(blocked));
    check_no_staging(directory);
}

void test_failure_injection(const fs::path & directory) {
    const auto path = directory / "faults.bin";
    const std::array<std::string, 3> old_parts = { "old KV", "old media", "old checkpoint" };
    const std::array<std::string, 3> new_parts = { std::string(131073, 'k'), std::string(65537, 'm'), "new checkpoint" };
    std::vector<std::string> operations;
    {
        server_slot_save save(path.u8string(), [&](const char * operation) {
            operations.emplace_back(operation);
            return false;
        });
        stage(save, new_parts);
        save.commit(true, true);
    }
    CHECK(std::count(operations.begin(), operations.end(), "write") == 8);
    CHECK(std::count(operations.begin(), operations.end(), "flush") == 1);
    CHECK(std::count(operations.begin(), operations.end(), "close") == 4);
    CHECK(std::count(operations.begin(), operations.end(), "rename") == 1);
    for (size_t failure = 0; failure < operations.size(); ++failure) {
        commit(path, old_parts);
        const auto committed = read_file(path);
        size_t operation = 0;
        rejected([&] {
            server_slot_save save(path.u8string(), [&](const char *) { return operation++ == failure; });
            stage(save, new_parts);
            save.commit(true, true);
        });
        CHECK(read_file(path) == committed);
        check_parts(path, old_parts);
        check_no_staging(directory);
    }
    commit(path, new_parts);
    const auto committed = read_file(path);
    operations.clear();
    {
        server_slot_save save(path.u8string(), [&](const char * operation) {
            operations.emplace_back(operation);
            return false;
        });
        save.unpack();
    }
    CHECK(std::count(operations.begin(), operations.end(), "write") == 6);
    CHECK(std::count(operations.begin(), operations.end(), "flush") == 3);
    CHECK(std::count(operations.begin(), operations.end(), "close") == 4);
    for (size_t failure = 0; failure < operations.size(); ++failure) {
        size_t operation = 0;
        {
            server_slot_save save(path.u8string(), [&](const char *) { return operation++ == failure; });
            rejected([&] { save.unpack(); });
            CHECK(!save.has(0));
        }
        CHECK(read_file(path) == committed);
        check_no_staging(directory);
    }
    check_parts(path, new_parts);
    std::printf("Checked %zu unpack fault points and 14 commit fault points\n", operations.size());
}

void test_cleanup_ownership(const fs::path & directory) {
    const auto path = directory / "cleanup.bin";
    fs::path stage_directory;
    {
        server_slot_save save(path.u8string());
        stage_directory = fs::u8path(save.path(0)).parent_path();
#ifndef _WIN32
        struct stat status;
        CHECK(stat(stage_directory.c_str(), &status) == 0);
        CHECK((status.st_mode & 0777) == 0700);
#endif
        stage(save, { "KV", "media", "checkpoint" });
        write_file(stage_directory / "unowned", "keep");
    }
    CHECK(!fs::exists(stage_directory / "0"));
    CHECK(read_file(stage_directory / "unowned") == "keep");
    CHECK(fs::remove(stage_directory / "unowned"));
    CHECK(fs::remove(stage_directory));
    check_no_staging(directory);
}

#ifndef _WIN32
void test_process_interruption(const fs::path & directory) {
    const auto path = directory / "interrupted.bin";
    const std::array<std::string, 3> old_parts = { "old KV", "old media", "old checkpoint" };
    const std::array<std::string, 3> new_parts = { "new KV", "new media", "new checkpoint" };
    // Exit without destructors immediately before publish, then immediately after publish.
    for (bool before_rename : { true, false }) {
        commit(path, old_parts);
        {
            server_slot_save save(path.u8string(), [&](const char * operation) {
                if (before_rename && std::string(operation) == "rename") {
                    _exit(17);
                }
                return false;
            });
            stage(save, new_parts);
            const pid_t child = fork();
            CHECK(child >= 0);
            if (child == 0) {
                save.commit(true, true);
                _exit(17);
            }
            int status = 0;
            CHECK(waitpid(child, &status, 0) == child);
            CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 17);
            check_parts(path, before_rename ? old_parts : new_parts);
        }
        check_no_staging(directory);
    }
}
#endif

void test_concurrent_readers(const fs::path & directory) {
    const auto path = directory / "concurrent.bin";
    const std::array<std::string, 3> first = { std::string(131073, 'a'), std::string(65537, 'b'), "checkpoint one" };
    const std::array<std::string, 3> second = { std::string(131073, 'x'), std::string(65537, 'y'), "checkpoint two" };
    commit(path, first);
    // Pin an opened generation across replacement, including on Windows with delete sharing.
    std::atomic<bool> opened(false);
    std::atomic<bool> replaced(false);
    std::thread pinned_reader([&] {
        server_slot_save save(path.u8string(), [&](const char * operation) {
            if (std::string(operation) == "write" && !opened.exchange(true)) {
                while (!replaced.load()) {
                    std::this_thread::yield();
                }
            }
            return false;
        });
        save.unpack();
        for (size_t i = 0; i < first.size(); ++i) {
            CHECK(read_file(fs::u8path(save.path(i))) == first[i]);
        }
    });
    while (!opened.load()) {
        std::this_thread::yield();
    }
    commit(path, second);
    replaced = true;
    pinned_reader.join();

    std::atomic<bool> start(false);
    std::atomic<size_t> ready(0);
    std::atomic<size_t> reads(0);
    std::vector<std::thread> readers;
    for (size_t i = 0; i < 3; ++i) {
        readers.emplace_back([&] {
            ++ready;
            while (!start.load()) {
                std::this_thread::yield();
            }
            for (size_t n = 0; n < 24; ++n) {
                server_slot_save save(path.u8string());
                save.unpack();
                const auto kv = read_file(fs::u8path(save.path(0)));
                const auto & expected = kv == first[0] ? first : second;
                CHECK(kv == expected[0]);
                CHECK(read_file(fs::u8path(save.path(1))) == expected[1]);
                CHECK(read_file(fs::u8path(save.path(2))) == expected[2]);
                ++reads;
            }
        });
    }
    while (ready.load() != readers.size()) {
        std::this_thread::yield();
    }
    start = true;
    for (size_t i = 0; i < 24; ++i) {
        commit(path, i % 2 ? first : second);
    }
    for (auto & reader : readers) {
        reader.join();
    }
    CHECK(reads == 72);
    check_no_staging(directory);
}

} // namespace

int main() {
    test_directory directory;
    test_roundtrip_and_overwrite(directory.path);
    test_invalid_containers(directory.path);
    test_missing_components_and_rename(directory.path);
    test_failure_injection(directory.path);
    test_cleanup_ownership(directory.path);
#ifndef _WIN32
    test_process_interruption(directory.path);
#endif
    test_concurrent_readers(directory.path);
    std::puts("test-slot-save: OK");
    return 0;
}
