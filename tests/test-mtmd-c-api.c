#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "mtmd.h"
#include "mtmd-helper.h"

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        abort(); \
    } \
} while (0)

static void assert_chunks_equal(
        const mtmd_input_chunk * expected,
        const mtmd_input_chunk * actual) {
    const enum mtmd_input_chunk_type type = mtmd_input_chunk_get_type(expected);
    CHECK(mtmd_input_chunk_get_type(actual) == type);
    CHECK(mtmd_input_chunk_get_n_tokens(actual) == mtmd_input_chunk_get_n_tokens(expected));
    CHECK(mtmd_input_chunk_get_n_pos(actual) == mtmd_input_chunk_get_n_pos(expected));

    if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t n_expected = 0;
        size_t n_actual = 0;
        const llama_token * expected_tokens = mtmd_input_chunk_get_tokens_text(expected, &n_expected);
        const llama_token * actual_tokens = mtmd_input_chunk_get_tokens_text(actual, &n_actual);
        CHECK(n_actual == n_expected);
        CHECK(memcmp(actual_tokens, expected_tokens, n_expected * sizeof(llama_token)) == 0);
    } else {
        CHECK(strcmp(mtmd_input_chunk_get_id(actual), mtmd_input_chunk_get_id(expected)) == 0);
    }
}

static void test_chunk_serialization(const mtmd_input_chunk * chunk, uint32_t version) {
    const size_t size = mtmd_input_chunk_serialized_size_version(chunk, version);
    CHECK(size > 0);
    fprintf(stderr, "serialization version %u, %zu bytes\n", version, size);

    uint8_t * data = (uint8_t *) malloc(size);
    CHECK(data != NULL);
    CHECK(mtmd_input_chunk_serialize_version(chunk, data, size - 1, version) == 0);
    CHECK(mtmd_input_chunk_serialize_version(chunk, data, size, version) == size);

    size_t bytes_read = 0;
    mtmd_input_chunk * restored = mtmd_input_chunk_deserialize_version(
        data, size, &bytes_read, version);
    CHECK(restored != NULL);
    CHECK(bytes_read == size);
    assert_chunks_equal(chunk, restored);
    mtmd_input_chunk_free(restored);

    if (version == MTMD_INPUT_CHUNK_SERIALIZATION_VERSION) {
        CHECK(mtmd_input_chunk_serialized_size(chunk) == size);
        CHECK(mtmd_input_chunk_serialize(chunk, data, size) == size);
        restored = mtmd_input_chunk_deserialize(data, size, &bytes_read);
        CHECK(restored != NULL);
        CHECK(bytes_read == size);
        assert_chunks_equal(chunk, restored);
        mtmd_input_chunk_free(restored);
    }

    bytes_read = 123;
    CHECK(mtmd_input_chunk_deserialize_version(data, size - 1, &bytes_read, version) == NULL);
    CHECK(bytes_read == 0);

    free(data);
}

int main(void) {
    printf("\n\nTesting libmtmd C API...\n");
    printf("--------\n\n");

    struct mtmd_context_params params = mtmd_context_params_default();
    printf("Default image marker: %s\n", params.image_marker);

    mtmd_input_chunks * chunks = mtmd_test_create_input_chunks();

    if (!chunks) {
        fprintf(stderr, "Failed to create input chunks\n");
        return 1;
    }

    // simple test for the helper
    size_t n_tokens_total = mtmd_helper_get_n_tokens(chunks);
    printf("Total tokens in chunks: %zu\n", n_tokens_total);
    assert(n_tokens_total > 0);

    size_t n_chunks = mtmd_input_chunks_size(chunks);
    printf("Number of chunks: %zu\n", n_chunks);
    assert(n_chunks > 0);

    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        assert(chunk != NULL);
        enum mtmd_input_chunk_type type = mtmd_input_chunk_get_type(chunk);
        printf("Chunk %zu type: %d\n", i, type);
        fprintf(stderr, "testing chunk %zu, type %d\n", i, type);
        CHECK(mtmd_input_chunk_serialized_size_version(chunk, 1) == 0);
        CHECK(mtmd_input_chunk_serialized_size_version(
            chunk, MTMD_INPUT_CHUNK_SERIALIZATION_VERSION + 1) == 0);
        test_chunk_serialization(chunk, 2);
        test_chunk_serialization(chunk, MTMD_INPUT_CHUNK_SERIALIZATION_VERSION);

        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            printf("    Text chunk with %zu tokens\n", n_tokens);
            assert(tokens != NULL);
            assert(n_tokens > 0);
            for (size_t j = 0; j < n_tokens; j++) {
                assert(tokens[j] >= 0);
                printf("    > Token %zu: %d\n", j, tokens[j]);
            }

        } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const mtmd_image_tokens * image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            size_t n_tokens = mtmd_image_tokens_get_n_tokens(image_tokens);
            // get position of the last token, which should be (nx - 1, ny - 1)
            struct mtmd_decoder_pos pos = mtmd_image_tokens_get_decoder_pos(image_tokens, 0, n_tokens - 1);
            size_t nx = pos.x + 1;
            size_t ny = pos.y + 1;
            const char * id = mtmd_image_tokens_get_id(image_tokens);
            assert(n_tokens > 0);
            assert(nx > 0);
            assert(ny > 0);
            assert(id != NULL);
            printf("    Image chunk with %zu tokens\n", n_tokens);
            printf("    Image size: %zu x %zu\n", nx, ny);
            printf("    Image ID: %s\n", id);
        }
    }

    // Free the chunks
    mtmd_input_chunks_free(chunks);

    printf("\n\nDONE: test libmtmd C API...\n");

    return 0;
}
