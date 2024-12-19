module;

#include <cstdlib>
#include <omp.h>
#include <stdint.h>
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
    }
    ~AlignedArray() { free(ptr); }
    size_t ptr_as_int() { return (size_t)ptr; }
    scalar_t *ptr;
    size_t size;
};
export void Fill(AlignedArray *out, scalar_t val) {
    /**
     * Fill the values of an aligned array with val
     */
    std::fill(out->ptr, out->ptr + out->size, val);
}

export void Compact(const AlignedArray &a, AlignedArray *out,
                    const std::vector<uint32_t> &shape,
                    const std::vector<uint32_t> &strides, const size_t offset) {
    /**
     * Compact an array in memory
     *
     * Args:
     *   a: non-compact representation of the array, given as input
     *   out: compact version of the array to be written
     *   shape: shapes of each dimension for a and out
     *   strides: strides of the *a* array (not out, which has compact strides)
     *   offset: offset of the *a* array (not out, which has zero offset, being
     * compact)
     *
     * Returns:
     * void
     */
    size_t num_dims = shape.size();
    // precalculate strides for compact array
    std::vector<uint32_t> compact_strides(num_dims, 1);
    for (int i = num_dims - 2; i >= 0; --i) {
        compact_strides[i] = compact_strides[i + 1] * shape[i + 1];
    }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < out->size; i++) {
        uint32_t idx_offset = offset;

        uint32_t temp = i;
        for (size_t j = 0; j < num_dims; j++) {
            uint32_t idx = temp / compact_strides[j];
            idx_offset += idx * strides[j];
            temp %= compact_strides[j];
        }
        out->ptr[i] = a.ptr[idx_offset];
    };
}
} // namespace cpu
} // namespace needle
