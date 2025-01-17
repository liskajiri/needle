module;

#include <cmath>
#include <omp.h>
#include <vector>

#include <nanobind/stl/tuple.h>

import ndarray;
export module scalar_ops;

namespace needle {
namespace cpu {
export void Fill(AlignedArray *out, const scalar_t val) {
    /**
     * Fill the values of an aligned array with val
     */
    // Parallelization here is questionable
    std::fill(out->ptr, out->ptr + out->size, val);
}

export void ScalarSetitem(const size_t size, const scalar_t val,
                          AlignedArray *out, const nanobind::tuple &shape,
                          const nanobind::tuple &strides, const size_t offset) {
    /**
     * Set items in a (non-compact) array
     *
     * Args:
     *   size: number of elements to write in out array (note that this will not
     * be the same as out.size, because out is a non-compact subset array);  it
     * _will_ be the same as the product of items in shape, but convenient to
     * just pass it here. val: scalar value to write to out: non-compact array
     * whose items are to be written shape: shapes of each dimension of out
     * strides: strides of the out array offset: offset of the out array
     */
    // vector for indexes when iterating over the arrays
    size_t num_dims = shape.size();
    // precalculate strides for compact array
    std::vector<uint32_t> compact_strides(num_dims, 1);
    for (int i = num_dims - 2; i >= 0; --i) {
        compact_strides[i] =
            compact_strides[i + 1] * nanobind::cast<size_t>(shape[i + 1]);
    }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
        uint32_t idx_offset = offset;

        uint32_t temp = i;
        for (size_t j = 0; j < num_dims; j++) {
            uint32_t idx = temp / compact_strides[j];
            idx_offset += idx * nanobind::cast<size_t>(strides[j]);
            temp %= compact_strides[j];
        }
        out->ptr[idx_offset] = val;
    };
}

// scalar operations
template <typename Func>
void ScalarOp(const AlignedArray &a, const float scalar, AlignedArray *out,
              Func func) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < a.size; ++i) {
        out->ptr[i] = func(a.ptr[i], scalar);
    }
}

export void ScalarAdd(const AlignedArray &a, const float scalar,
                      AlignedArray *out) {
    ScalarOp(a, scalar, out, std::plus<float>());
}

export void ScalarMul(const AlignedArray &a, const float scalar,
                      AlignedArray *out) {
    ScalarOp(a, scalar, out, std::multiplies<float>());
}

export void ScalarDiv(const AlignedArray &a, const float scalar,
                      AlignedArray *out) {
    ScalarOp(a, scalar, out, std::divides<float>());
}

export void ScalarPower(const AlignedArray &a, const float scalar,
                        AlignedArray *out) {
    ScalarOp(a, scalar, out, [](float x, float y) { return std::pow(x, y); });
}

export void ScalarMaximum(const AlignedArray &a, const float scalar,
                          AlignedArray *out) {
    ScalarOp(a, scalar, out, [](float x, float y) { return std::max(x, y); });
}

export void ScalarEq(const AlignedArray &a, const float scalar,
                     AlignedArray *out) {
    ScalarOp(a, scalar, out, std::equal_to<float>());
}

export void ScalarGe(const AlignedArray &a, const float scalar,
                     AlignedArray *out) {
    ScalarOp(a, scalar, out, std::greater_equal<float>());
}

} // namespace cpu
} // namespace needle
