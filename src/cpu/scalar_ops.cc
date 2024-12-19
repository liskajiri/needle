module;

#include <cmath>
#include <omp.h>
#include <vector>
#include <x86intrin.h>

import ndarray;
export module scalar_ops;

namespace needle {
namespace cpu {
// TODO: Same signature as elementwise.cc
export void ScalarSetitem(const size_t size, scalar_t val, AlignedArray *out,
                          const std::vector<uint32_t> &shape,
                          const std::vector<uint32_t> &strides, size_t offset) {
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
    std::vector<uint32_t> indexes(shape.size(), 0);
    for (size_t i = 0; i < size; i++) {
        uint32_t idx_offset = offset;
        for (size_t j = 0; j < indexes.size(); j++) {
            idx_offset += indexes[j] * strides[j];
        }
        out->ptr[idx_offset] = val;

        // increment last axis
        indexes[indexes.size() - 1] += 1;
        for (int j = shape.size() - 1; j >= 0; j--) {
            // overflow addition
            if (indexes[j] == shape[j]) {
                indexes[j] = 0;
                if (j >= 1) {
                    indexes[j - 1] += 1;
                }
            }
        }
    };
}

// scalar operations
template <typename Func>
void ScalarOp(const AlignedArray &a, float scalar, AlignedArray *out,
              Func func) {
    // TODO: openmp
    for (size_t i = 0; i < a.size; ++i) {
        out->ptr[i] = func(a.ptr[i], scalar);
    }
}

export void ScalarAdd(const AlignedArray &a, float scalar, AlignedArray *out) {
    ScalarOp(a, scalar, out, std::plus<float>());
}

export void ScalarMul(const AlignedArray &a, float scalar, AlignedArray *out) {
    ScalarOp(a, scalar, out, std::multiplies<float>());
}

export void ScalarDiv(const AlignedArray &a, float scalar, AlignedArray *out) {
    ScalarOp(a, scalar, out, std::divides<float>());
}

export void ScalarPower(const AlignedArray &a, float scalar,
                        AlignedArray *out) {
    ScalarOp(a, scalar, out, [](float x, float y) { return std::pow(x, y); });
}

export void ScalarMaximum(const AlignedArray &a, float scalar,
                          AlignedArray *out) {
    ScalarOp(a, scalar, out, [](float x, float y) { return std::max(x, y); });
}

export void ScalarEq(const AlignedArray &a, float scalar, AlignedArray *out) {
    ScalarOp(a, scalar, out, std::equal_to<float>());
}

export void ScalarGe(const AlignedArray &a, float scalar, AlignedArray *out) {
    ScalarOp(a, scalar, out, std::greater_equal<float>());
}

} // namespace cpu
} // namespace needle
