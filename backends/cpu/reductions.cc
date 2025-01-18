module;

#include <cmath>
#include <omp.h>
import ndarray;

export module reductions;

namespace needle {
namespace cpu {
export void ReduceMax(const AlignedArray &a, AlignedArray *out,
                      size_t reduce_size) {
    /**
     * Reduce by taking maximum over `reduce_size` contiguous blocks.
     *
     * Args:
     *   a: compact array of size a.size = out.size * reduce_size to reduce over
     *   out: compact array to write into
     *   reduce_size: size of the dimension to reduce over
     */

#pragma omp parallel for
    for (size_t i = 0; i < out->size; i++) {
        // Initialize max value with the first element in the block
        float curr_max = a.ptr[i * reduce_size];

#pragma omp simd reduction(max : curr_max)
        for (size_t j = 0; j < reduce_size; j++) {
            curr_max = std::max(curr_max, a.ptr[i * reduce_size + j]);
        }

        out->ptr[i] = curr_max;
    }
}

export void ReduceSum(const AlignedArray &a, AlignedArray *out,
                      size_t reduce_size) {
    /**
     * Reduce by taking sum over `reduce_size` contiguous blocks.
     *
     * Args:
     *   a: compact array of size a.size = out.size * reduce_size to reduce over
     *   out: compact array to write into
     *   reduce_size: size of the dimension to reduce over
     */
#pragma omp parallel for
    for (size_t i = 0; i < out->size; i++) {
        double temp = 0.0;

#pragma omp simd reduction(+ : temp)
        for (size_t j = 0; j < reduce_size; j++) {
            temp += a.ptr[i * reduce_size + j];
        }

        out->ptr[i] = temp;
    }
}

} // namespace cpu
} // namespace needle
