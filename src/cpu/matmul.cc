module;

#include <cmath>
#include <omp.h>
#include <x86intrin.h>

import ndarray;
import scalar_ops;

export module matmul;

namespace needle {
namespace cpu {

export void Matmul(const AlignedArray &a, const AlignedArray &b,
                   AlignedArray *out, uint32_t m, uint32_t n, uint32_t p) {
    /**
     * Multiply two (compact) matrices into an output (also compact) matrix. For
     * this implementation you can use the "naive" three-loop algorithm.
     *
     * Args:
     *   a: compact 2D array of size m x n
     *   b: compact 2D array of size n x p
     *   out: compact 2D array of size m x p to write the output to
     *   m: rows of a / out
     *   n: columns of a / rows of b
     *   p: columns of b / out
     */
    Fill(out, 0.0);

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
#pragma omp simd
            for (size_t k = 0; k < n; k++) {
                out->ptr[i * p + j] += a.ptr[i * n + k] * b.ptr[k * p + j];
            }
        }
    }
}

inline float horizontal_sum(__m256 vec) {
    /*
    Given a 256-bit vector, sum the 8 floats in the vector and return the
    result.
    */
    // Sum the high and low halves of the vector
    __m128 low = _mm256_castps256_ps128(vec);    // Extract lower 128 bits
    __m128 high = _mm256_extractf128_ps(vec, 1); // Extract higher 128 bits
    __m128 sum128 = _mm_add_ps(low, high);       // Add the two halves

    // Perform the horizontal sum on the 128-bit vector
    __m128 shuf = _mm_movehdup_ps(sum128);  // Shuffle for cross sums
    __m128 sums = _mm_add_ps(sum128, shuf); // Add shuffled values
    shuf = _mm_movehl_ps(shuf, sums);       // Move upper to lower
    sums = _mm_add_ss(sums, shuf);          // Add the final two

    return _mm_cvtss_f32(sums); // Extract the result
}

inline void AlignedDot(const scalar_t *__restrict__ a,
                       const scalar_t *__restrict__ b,
                       scalar_t *__restrict__ out) {

    /**
     * Multiply together two TILE x TILE matrices, and _add _the result to out
     * (it is important to add the result to the existing out, which you should
     * not set to zero beforehand).  We are including the compiler flags here
     * that enable the compile to properly use vector operators to implement
     * this function.  Specifically, the __restrict__ keyword indicates to the
     * compile that a, b, and out don't have any overlapping memory.
     * Similarly the __builtin_assume_aligned keyword tells the compiler
     * that the input array will be aligned to the appropriate blocks in memory,
     * which also helps the compiler vectorize the code.
     *
     * Args:
     *   a: compact 2D array of size TILE x TILE
     *   b: compact 2D array of size TILE x TILE
     *   out: compact 2D array of size TILE x TILE to write to
     */

    a = (const scalar_t *)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
    b = (const scalar_t *)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
    out = (scalar_t *)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

#pragma omp simd
    for (size_t i = 0; i < TILE; i++) {
        for (size_t j = 0; j < TILE; j++) {
            scalar_t sum = 0.0;
            for (size_t k = 0; k < TILE; k++) {
                sum += a[i * TILE + k] * b[k * TILE + j];
                // transposed B
                // out[i * TILE + j] += a[i * TILE + k] * b[j * TILE + k];
            }
            out[i * TILE + j] += sum;
        }
    }

    // // row a
    // for (uint16_t i = 0; i < TILE; i++)
    // {
    //     // col b
    //     for (uint16_t j = 0; j < TILE; j++)
    //     {
    //         __m256 out_vec = _mm256_setzero_ps();
    //         // row b, col a
    //         for (uint16_t k = 0; k < TILE; k += 8)
    //         {
    //             // load 8 floats from a and b
    //             __m256 a_vec = _mm256_load_ps(&a[i * TILE + k]);
    //             // For the transposed version
    //             // __m256 b_vec = _mm256_load_ps(&b[j * TILE + k]);

    //             __m256 b_vec = _mm256_set_ps(
    //                 b[(k + 7) * TILE + j],
    //                 b[(k + 6) * TILE + j],
    //                 b[(k + 5) * TILE + j],
    //                 b[(k + 4) * TILE + j],
    //                 b[(k + 3) * TILE + j],
    //                 b[(k + 2) * TILE + j],
    //                 b[(k + 1) * TILE + j],
    //                 b[k * TILE + j]);

    //             out_vec = _mm256_fmadd_ps(a_vec, b_vec, out_vec);
    //         }
    //         // result of one element
    //         // sum the 8 floats in out_vec
    //         out[i * TILE + j] += horizontal_sum(out_vec);
    //     }
    // }
}

export void MatmulTiled(const AlignedArray &a, const AlignedArray &b,
                        AlignedArray *out, uint32_t m, uint32_t n, uint32_t p) {
    /**
     * Matrix multiplication on tiled representations of array.  In this
     * setting, a, b, and out are all *4D* compact arrays of the appropriate
     * size, e.g. a is an array of size a[m/TILE][n/TILE][TILE][TILE]
     *
     * This function will only be called when m, n, p are all
     * multiples of TILE.
     *
     * Args:
     *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
     *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
     *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
     *   m: rows of a / out
     *   n: columns of a / rows of b
     *   p: columns of b / out
     *
     */
    size_t a_rows = m / TILE;
    size_t a_cols = n / TILE;
    size_t b_cols = p / TILE;

    // first set out array to zero
    Fill(out, 0.0);

    // transpose B
    // AlignedArray b_transposed(b.size);
    // for (size_t i = 0; i < a_cols; i++)
    // {
    //     for (size_t j = 0; j < b_cols; j++)
    //     {
    //         for (size_t k = 0; k < TILE; k++)
    //         {
    //             for (size_t l = 0; l < TILE; l++)
    //             {
    //                 b_transposed.ptr[(j * a_cols + i) * TILE_SIZE + l * TILE
    //                 + k] =
    //                     b.ptr[(i * b_cols + j) * TILE_SIZE + k * TILE + l];
    //             }
    //         }
    //     }
    // }

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t row_a = 0; row_a < a_rows; row_a++) {
        for (size_t col_b = 0; col_b < b_cols; col_b++) {
            // block[i][j][:][:]
            scalar_t *out_block =
                (scalar_t *)(out->ptr + TILE_SIZE * (row_a * b_cols + col_b));
            for (size_t col_a = 0; col_a < a_cols; col_a++) {
                // tile multiplication
                scalar_t *tile_a =
                    (scalar_t *)(a.ptr + TILE_SIZE * (row_a * a_cols + col_a));
                scalar_t *tile_b =
                    (scalar_t *)(b.ptr + TILE_SIZE * (col_a * b_cols + col_b));
                // float *tile_b = (scalar_t *)(b_transposed.ptr + TILE_SIZE *
                // (j * a_cols + k));
                AlignedDot(tile_a, tile_b, out_block);
            }
        }
    }
}
} // namespace cpu
} // namespace needle
