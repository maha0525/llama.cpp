#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

// Linux-only test shim. The server has no fault injection settings.
static const char * component(const char * path) {
    const char * root = getenv("SLOT_SAVE_FAULT_ROOT");
    if (!root || strncmp(path, root, strlen(root)) != 0) {
        return NULL;
    }
    const char * stage = path + strlen(root);
    if (strncmp(stage, "/.llama-slot-", 13) != 0) {
        return NULL;
    }
    const char * name = strchr(stage + 1, '/');
    if (!name) {
        return NULL;
    }
    ++name;
    if (strcmp(name, "0") && strcmp(name, "1") && strcmp(name, "2") && strcmp(name, "container")) {
        return NULL;
    }
    return name;
}

static void append_trace(const char * op, const char * name, int failed) {
    const char * path = getenv("SLOT_SAVE_FAULT_TRACE");
    if (!path) {
        return;
    }
    int fd = syscall(SYS_openat, AT_FDCWD, path, O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (fd >= 0) {
        char line[128];
        int n = snprintf(line, sizeof(line), "%s %s %d\n", op, name, failed);
        syscall(SYS_write, fd, line, n);
        syscall(SYS_close, fd);
    }
}

static int fail_path(const char * path, const char * op) {
    const int saved_errno = errno;
    const char * name = component(path);
    if (!name) {
        errno = saved_errno;
        return 0;
    }

    int failed = 0;
    const char * marker = getenv("SLOT_SAVE_FAULT_MARKER");
    int fd = marker ? syscall(SYS_openat, AT_FDCWD, marker, O_RDONLY, 0) : -1;
    if (fd >= 0) {
        char buf[128] = {0};
        ssize_t n = syscall(SYS_read, fd, buf, sizeof(buf) - 1);
        syscall(SYS_close, fd);
        char wanted_op[32], wanted_name[32];
        unsigned generation, occurrence;
        static unsigned last_generation = 0, seen = 0;
        if (n > 0 && sscanf(buf, "%u %31s %31s %u", &generation, wanted_op, wanted_name, &occurrence) == 4) {
            if (generation != last_generation) {
                last_generation = generation;
                seen = 0;
            }
            if (!strcmp(wanted_op, op) && !strcmp(wanted_name, name)) {
                failed = ++seen == occurrence;
            }
        }
    }
    append_trace(op, name, failed);
    errno = saved_errno;
    return failed;
}

static int fail_fd(int fd, const char * op) {
    const int saved_errno = errno;
    int flags = fcntl(fd, F_GETFL);
    if (flags < 0 || (flags & O_ACCMODE) == O_RDONLY) {
        errno = saved_errno;
        return 0;
    }
    char link[64], path[PATH_MAX];
    snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
    ssize_t n = readlink(link, path, sizeof(path) - 1);
    errno = saved_errno;
    if (n < 0) {
        return 0;
    }
    path[n] = 0;
    return fail_path(path, op);
}

size_t fwrite(const void * ptr, size_t size, size_t nmemb, FILE * stream) {
    static size_t (*real_fwrite)(const void *, size_t, size_t, FILE *);
    if (!real_fwrite) {
        real_fwrite = dlsym(RTLD_NEXT, "fwrite");
    }
    if (size && nmemb && fail_fd(fileno(stream), "fwrite")) {
        errno = ENOSPC;
        return 0;
    }
    return real_fwrite(ptr, size, nmemb, stream);
}

int fflush(FILE * stream) {
    static int (*real_fflush)(FILE *);
    if (!real_fflush) {
        real_fflush = dlsym(RTLD_NEXT, "fflush");
    }
    if (stream && fail_fd(fileno(stream), "fflush")) {
        errno = ENOSPC;
        return EOF;
    }
    return real_fflush(stream);
}

int fclose(FILE * stream) {
    static int (*real_fclose)(FILE *);
    if (!real_fclose) {
        real_fclose = dlsym(RTLD_NEXT, "fclose");
    }
    int failed = fail_fd(fileno(stream), "fclose");
    int result = real_fclose(stream);
    if (failed) {
        errno = ENOSPC;
        return EOF;
    }
    return result;
}

ssize_t write(int fd, const void * ptr, size_t count) {
    static ssize_t (*real_write)(int, const void *, size_t);
    if (!real_write) {
        real_write = dlsym(RTLD_NEXT, "write");
    }
    if (count && fail_fd(fd, "write")) {
        errno = ENOSPC;
        return -1;
    }
    return real_write(fd, ptr, count);
}

ssize_t writev(int fd, const struct iovec * iov, int count) {
    static ssize_t (*real_writev)(int, const struct iovec *, int);
    if (!real_writev) {
        real_writev = dlsym(RTLD_NEXT, "writev");
    }
    if (count && fail_fd(fd, "writev")) {
        errno = ENOSPC;
        return -1;
    }
    return real_writev(fd, iov, count);
}

int close(int fd) {
    static int (*real_close)(int);
    if (!real_close) {
        real_close = dlsym(RTLD_NEXT, "close");
    }
    int failed = fail_fd(fd, "close");
    int result = real_close(fd);
    if (failed) {
        errno = ENOSPC;
        return -1;
    }
    return result;
}

int rename(const char * oldpath, const char * newpath) {
    static int (*real_rename)(const char *, const char *);
    if (!real_rename) {
        real_rename = dlsym(RTLD_NEXT, "rename");
    }
    if (fail_path(oldpath, "rename")) {
        errno = ENOSPC;
        return -1;
    }
    return real_rename(oldpath, newpath);
}
