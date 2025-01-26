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

#pragma omp parallel for if (m * n * p > 16 * 16 * 16) collapse(2)             \
    schedule(static)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
#pragma omp simd
            for (size_t k = 0; k < n; k++) {
                out->ptr[i * p + j] += a.ptr[i * n + k] * b.ptr[k * p + j];
            }
        }
    }
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
            }
            out[i * TILE + j] += sum;
        }
    }
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
                AlignedDot(tile_a, tile_b, out_block);
            }
        }
    }
}
} // namespace cpu
} // namespace needle
