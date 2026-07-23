#include "server-common.h"

extern "C" {
#include "sha256.h"
}

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        std::abort(); \
    } \
} while (0)

namespace {

struct temp_file {
    std::filesystem::path path;

    ~temp_file() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

void check_media_equal(const server_tokens & expected, const server_tokens & actual) {
    size_t expected_idx = 0;
    size_t actual_idx = 0;

    while (true) {
        const auto expected_media = expected.find_next_media_chunk(expected_idx);
        const auto actual_media = actual.find_next_media_chunk(actual_idx);

        CHECK((expected_media.first == nullptr) == (actual_media.first == nullptr));
        if (expected_media.first == nullptr) {
            break;
        }

        CHECK(expected_media.second == actual_media.second);
        const mtmd_input_chunk * expected_chunk = expected_media.first->get();
        const mtmd_input_chunk * actual_chunk = actual_media.first->get();
        CHECK(expected_chunk != nullptr);
        CHECK(actual_chunk != nullptr);
        CHECK(mtmd_input_chunk_get_type(expected_chunk) == mtmd_input_chunk_get_type(actual_chunk));
        CHECK(mtmd_input_chunk_get_n_tokens(expected_chunk) == mtmd_input_chunk_get_n_tokens(actual_chunk));
        CHECK(mtmd_input_chunk_get_n_pos(expected_chunk) == mtmd_input_chunk_get_n_pos(actual_chunk));
        CHECK(std::string(mtmd_input_chunk_get_id(expected_chunk)) ==
              std::string(mtmd_input_chunk_get_id(actual_chunk)));

        expected_idx = expected_media.second;
        actual_idx = actual_media.second;
    }
}

template <typename T>
bool write_scalar(std::ofstream & out, const T & value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
    return out.good();
}

struct byte_writer {
    std::vector<uint8_t> buf;

    template <typename T>
    void put(T value) {
        const auto * p = reinterpret_cast<const uint8_t *>(&value);
        buf.insert(buf.end(), p, p + sizeof(T));
    }

    void put_bytes(const void * data, size_t n) {
        const auto * p = static_cast<const uint8_t *>(data);
        buf.insert(buf.end(), p, p + n);
    }
};

// A version-3 image chunk, byte for byte as earlier fork builds wrote it:
// clip_image_f32_batch still carried grid_x / grid_y then. Upstream has since
// removed those fields, so this build must consume and discard them — sidecar
// files saved before that catch-up have to stay loadable.
void check_v3_image_chunk_reads() {
    const std::string id = "image_1";

    byte_writer w;
    w.put<uint32_t>(MTMD_INPUT_CHUNK_TYPE_IMAGE);
    w.put<uint8_t>(1);            // has_image
    w.put<uint8_t>(0);            // has_audio
    w.put<uint64_t>(0);           // n_text
    // mtmd_image_tokens
    w.put<uint32_t>(4);           // nx
    w.put<uint32_t>(4);           // ny
    w.put<uint32_t>(0);           // pos = MTMD_POS_TYPE_NORMAL
    w.put<uint32_t>(0);           // image_idx
    w.put<uint32_t>(1);           // n_temporal_merge
    w.put<uint64_t>(id.size());
    w.put_bytes(id.data(), id.size());
    // clip_image_f32_batch
    w.put<uint8_t>(0);            // is_audio
    w.put<int32_t>(2);            // grid_x - version 3 only, dropped on read
    w.put<int32_t>(3);            // grid_y - version 3 only, dropped on read
    w.put<uint64_t>(1);           // n_entries
    w.put<int32_t>(4);            // entry nx
    w.put<int32_t>(4);            // entry ny
    w.put<uint8_t>(0);            // add_viewsep
    w.put<uint8_t>(0);            // add_newline
    w.put<uint64_t>(0);           // buf_count == 0 -> placeholder entry

    size_t bytes_read = 0;
    mtmd::input_chunk_ptr chunk(mtmd_input_chunk_deserialize_version(
        w.buf.data(), w.buf.size(), &bytes_read, 3));
    CHECK(chunk.get() != nullptr);
    CHECK(bytes_read == w.buf.size());
    CHECK(mtmd_input_chunk_get_type(chunk.get()) == MTMD_INPUT_CHUNK_TYPE_IMAGE);
    CHECK(std::string(mtmd_input_chunk_get_id(chunk.get())) == id);
    CHECK(mtmd_input_chunk_get_n_tokens(chunk.get()) == 16);

    // The same bytes must not parse as the current version: read without the
    // grid fields, the entry count is taken from grid_x/grid_y and is absurd.
    size_t v4_bytes_read = 0;
    CHECK(mtmd_input_chunk_deserialize_version(
        w.buf.data(), w.buf.size(), &v4_bytes_read,
        MTMD_INPUT_CHUNK_SERIALIZATION_VERSION) == nullptr);

    // Version 3 is read-only: this build has no truthful grid value to write.
    CHECK(mtmd_input_chunk_serialized_size_version(chunk.get(), 3) == 0);
    CHECK(mtmd_input_chunk_serialized_size_version(
        chunk.get(), MTMD_INPUT_CHUNK_SERIALIZATION_VERSION) > 0);
}

bool write_v2_sidecar(
        const std::filesystem::path & path,
        const server_tokens & tokens,
        const std::vector<uint8_t> & hash,
        size_t max_chunks = std::numeric_limits<size_t>::max()) {
    struct media_entry {
        uint64_t start;
        const mtmd_input_chunk * chunk;
    };

    std::vector<media_entry> media;
    size_t idx = 0;
    while (true) {
        const auto next = tokens.find_next_media_chunk(idx);
        if (next.first == nullptr) {
            break;
        }
        const mtmd_input_chunk * chunk = next.first->get();
        CHECK(chunk != nullptr);
        media.push_back({static_cast<uint64_t>(next.second), chunk});
        if (media.size() == max_chunks) {
            break;
        }
        idx = next.second;
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }

    constexpr uint32_t version = 2;
    const uint64_t total_tokens = tokens.size();
    const uint64_t n_chunks = media.size();
    out.write("MTMD", 4);
    if (!write_scalar(out, version)) return false;
    out.write(reinterpret_cast<const char *>(hash.data()), hash.size());
    if (!write_scalar(out, total_tokens)) return false;
    if (!write_scalar(out, n_chunks)) return false;

    for (const auto & entry : media) {
        const uint32_t type = static_cast<uint32_t>(mtmd_input_chunk_get_type(entry.chunk));
        const size_t size = mtmd_input_chunk_serialized_size_version(entry.chunk, version);
        CHECK(size > 0);
        const uint64_t size_u64 = size;
        std::vector<uint8_t> data(size);
        CHECK(mtmd_input_chunk_serialize_version(
            entry.chunk, data.data(), data.size(), version) == size);

        if (!write_scalar(out, entry.start)) return false;
        if (!write_scalar(out, type)) return false;
        if (!write_scalar(out, size_u64)) return false;
        out.write(reinterpret_cast<const char *>(data.data()), data.size());
        if (!out.good()) return false;
    }

    out.write("DTMD", 4);
    return out.good();
}

} // namespace

int main() {
    const std::array<uint8_t, SHA256_DIGEST_SIZE> expected_abc = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
    };
    std::array<uint8_t, SHA256_DIGEST_SIZE> actual_abc{};
    const unsigned char abc[] = {'a', 'b', 'c'};
    sha256_hash(actual_abc.data(), abc, sizeof(abc));
    CHECK(actual_abc == expected_abc);

    mtmd::input_chunks chunks(mtmd_test_create_input_chunks());
    CHECK(chunks.ptr != nullptr);

    server_tokens source(chunks, true);
    CHECK(source.has_media());

    std::vector<uint8_t> hash(32);
    for (size_t i = 0; i < hash.size(); ++i) {
        hash[i] = static_cast<uint8_t>(i);
    }

    const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    temp_file sidecar{
        std::filesystem::temp_directory_path() /
        ("llama-mtmd-sidecar-test-" + std::to_string(nonce) + ".mtmd")
    };

    const size_t n_written = source.save_mtmd_sidecar(sidecar.path.string(), hash);
    CHECK(n_written > 0);
    CHECK(std::filesystem::file_size(sidecar.path) == n_written);

    server_tokens restored(source.get_tokens(), true);
    CHECK(!restored.has_media());
    CHECK(restored.load_mtmd_sidecar(sidecar.path.string(), hash));
    CHECK(restored.has_media());
    CHECK(restored.get_tokens() == source.get_tokens());
    check_media_equal(source, restored);

    std::vector<uint8_t> wrong_hash = hash;
    wrong_hash[0] ^= 0xff;
    server_tokens hash_rejected = source.clone();
    CHECK(!hash_rejected.load_mtmd_sidecar(sidecar.path.string(), wrong_hash));
    CHECK(hash_rejected.has_media());
    check_media_equal(source, hash_rejected);

    {
        std::ofstream out(sidecar.path, std::ios::binary | std::ios::app);
        CHECK(out.is_open());
        out.put('\0');
        CHECK(out.good());
    }
    server_tokens corrupt_rejected = source.clone();
    CHECK(!corrupt_rejected.load_mtmd_sidecar(sidecar.path.string(), hash));
    CHECK(corrupt_rejected.has_media());
    check_media_equal(source, corrupt_rejected);

    temp_file v2_sidecar{
        std::filesystem::temp_directory_path() /
        ("llama-mtmd-sidecar-v2-test-" + std::to_string(nonce) + ".mtmd")
    };
    CHECK(write_v2_sidecar(v2_sidecar.path, source, hash));
    server_tokens restored_v2(source.get_tokens(), true);
    CHECK(restored_v2.load_mtmd_sidecar(v2_sidecar.path.string(), hash));
    CHECK(restored_v2.has_media());
    check_media_equal(source, restored_v2);

    temp_file incomplete_sidecar{
        std::filesystem::temp_directory_path() /
        ("llama-mtmd-sidecar-incomplete-test-" + std::to_string(nonce) + ".mtmd")
    };
    CHECK(write_v2_sidecar(incomplete_sidecar.path, source, hash, 1));
    server_tokens incomplete_rejected = source.clone();
    CHECK(!incomplete_rejected.load_mtmd_sidecar(incomplete_sidecar.path.string(), hash));
    CHECK(incomplete_rejected.has_media());
    check_media_equal(source, incomplete_rejected);

    check_v3_image_chunk_reads();

    std::printf("mtmd sidecar round-trip, legacy-version and rejection tests passed\n");
    return 0;
}
