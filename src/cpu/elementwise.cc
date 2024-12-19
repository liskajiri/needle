module;

#include <cmath>
#include <omp.h>
#include <stdint.h>
#include <vector>

import ndarray;

export module elementwise;

namespace needle {
namespace cpu {
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

export void EwiseSetitem(const AlignedArray &a, AlignedArray *out,
                         const std::vector<uint32_t> &shape,
                         const std::vector<uint32_t> &strides,
                         const size_t offset) {
    /**
     * Set items in a (non-compact) array
     *
     * Args:
     *   a: _compact_ array whose items will be written to out
     *   out: non-compact array whose items are to be written
     *   shape: shapes of each dimension for a and out
     *   strides: strides of the *out* array (not a, which has compact strides)
     *   offset: offset of the *out* array (not a, which has zero offset, being
     * compact)
     */
    size_t num_dims = shape.size();
    // precalculate strides for compact array
    std::vector<uint32_t> compact_strides(num_dims, 1);
    for (int i = num_dims - 2; i >= 0; --i) {
        compact_strides[i] = compact_strides[i + 1] * shape[i + 1];
    }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < a.size; i++) {
        uint32_t idx_offset = offset;

        uint32_t temp = i;
        for (size_t j = 0; j < num_dims; j++) {
            uint32_t idx = temp / compact_strides[j];
            idx_offset += idx * strides[j];
            temp %= compact_strides[j];
        }
        out->ptr[idx_offset] = a.ptr[i];
    };
}

template <typename Func>
void EwiseOp(const AlignedArray &a, const AlignedArray &b, AlignedArray *out,
             Func func) {
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < a.size; ++i) {
        out->ptr[i] = func(a.ptr[i], b.ptr[i]);
    }
}

export void EwiseAdd(const AlignedArray &a, const AlignedArray &b,
                     AlignedArray *out) {
    EwiseOp(a, b, out, std::plus<float>());
}

export void EwiseMul(const AlignedArray &a, const AlignedArray &b,
                     AlignedArray *out) {
    EwiseOp(a, b, out, std::multiplies<float>());
}

export void EwiseDiv(const AlignedArray &a, const AlignedArray &b,
                     AlignedArray *out) {
    EwiseOp(a, b, out, std::divides<float>());
}

export void EwiseMaximum(const AlignedArray &a, const AlignedArray &b,
                         AlignedArray *out) {
    EwiseOp(a, b, out, [](float x, float y) { return std::max(x, y); });
}

export void EwiseEq(const AlignedArray &a, const AlignedArray &b,
                    AlignedArray *out) {
    EwiseOp(a, b, out, std::equal_to<float>());
}
export void EwiseGe(const AlignedArray &a, const AlignedArray &b,
                    AlignedArray *out) {
    EwiseOp(a, b, out, std::greater_equal<float>());
}

export void EwiseLog(const AlignedArray &a, AlignedArray *out) {
    for (size_t i = 0; i < a.size; ++i) {
        out->ptr[i] = std::log(a.ptr[i]);
    }
}

export void EwiseExp(const AlignedArray &a, AlignedArray *out) {
    for (size_t i = 0; i < a.size; ++i) {
        out->ptr[i] = std::exp(a.ptr[i]);
    }
}

export void EwiseTanh(const AlignedArray &a, AlignedArray *out) {
    for (size_t i = 0; i < a.size; ++i) {
        out->ptr[i] = std::tanh(a.ptr[i]);
    }
}
} // namespace cpu
} // namespace needle
