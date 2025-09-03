module;

#include <cstdlib>
#include <stdlib.h>
#include <vector>

export module ndarray;

namespace needle {
namespace cpu {

export using scalar_t = float;
export constexpr size_t ELEM_SIZE = sizeof(scalar_t);

export constexpr size_t ALIGNMENT = 256;
export constexpr size_t TILE = 8;
export constexpr size_t TILE_SIZE = TILE * TILE;

/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT
 * boundaries in memory.  This alignment should be at least TILE * ELEM_SIZE
 */
export struct AlignedArray {
    AlignedArray(const size_t size) {
        int ret = posix_memalign((void **)&ptr, ALIGNMENT, size * ELEM_SIZE);
        if (ret != 0)
            throw std::bad_alloc();
        this->size = size;
        // when converting a numpy array, we don't own the memory
        this->numpy_handle = nullptr;
    }
    ~AlignedArray() {
        if (numpy_handle == nullptr) {
            free(ptr);
        }
    }
    size_t ptr_as_int() { return (size_t)ptr; }
    scalar_t *ptr;
    size_t size;
    void *numpy_handle;
};

} // namespace cpu
} // namespace needle
