#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

// A single commit binds the KV state and its optional metadata. Old sidecar saves are not accepted.
class server_slot_save {
public:
    enum component : size_t { KV = 0, MTMD = 1, CKPT = 2 };
    using failure_hook = std::function<bool(const char * operation)>;

    explicit server_slot_save(const std::string & filepath, failure_hook fail = {});
    ~server_slot_save();
    server_slot_save(const server_slot_save &) = delete;
    server_slot_save & operator=(const server_slot_save &) = delete;

    const std::string & path(size_t index) const;
    bool has(size_t index) const;
    uint64_t size(size_t index) const;
    uint64_t total_size() const;

    // Writers must close the staged components successfully before commit. Errors throw std::runtime_error.
    // Replacement is atomic against process interruption, but does not promise power-loss durability. Saves are private to the process user.
    void commit(bool mtmd_present, bool ckpt_present);
    // Verify the entire save before the caller can use any extracted component.
    void unpack();

private:
    struct impl;
    std::unique_ptr<impl> p;
};
