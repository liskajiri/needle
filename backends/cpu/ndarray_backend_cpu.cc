module;

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <memory>

import ndarray;
import elementwise;
import scalar_ops;
import matmul;
import reductions;

export module ndarray_backend_cpu;

NB_MODULE(ndarray_backend_cpu, m) {
    namespace nb = nanobind;
    using namespace needle;
    using namespace cpu;

    m.attr("__device_name__") = "cpu";
    m.attr("__tile_size__") = TILE;

    nb::class_<AlignedArray>(m, "Array")
        .def(nb::init_implicit<size_t>(), nb::rv_policy::take_ownership)
        .def("ptr", &AlignedArray::ptr_as_int)
        // read only
        .def_ro("size", &AlignedArray::size);

    m.def(
        "to_numpy",
        [](const AlignedArray &a, const nb::tuple &shape_tuple,
           const nb::tuple &strides_tuple, const size_t offset) {
            const size_t ndim = shape_tuple.size();
            auto shape_arr = std::make_unique<size_t[]>(ndim);
            auto strides_arr = std::make_unique<int64_t[]>(ndim);

            for (size_t i = 0; i < ndim; i++) {
                shape_arr[i] = nb::cast<size_t>(shape_tuple[i]);
                strides_arr[i] =
                    static_cast<int64_t>(nb::cast<size_t>(strides_tuple[i]));
            }

            return nb::ndarray<nb::numpy, scalar_t>(
                a.ptr + offset, ndim, shape_arr.get(),
                nb::capsule(&a, [](void *) noexcept {}), strides_arr.get());
        }
        // Keep AlignedArray alive while numpy array exists
        ,
        nb::keep_alive<0, 1>());

    // convert from numpy (with copying)
    m.def(
        "from_numpy",
        [](nb::ndarray<scalar_t> &a, AlignedArray *out) {
            out->ptr = static_cast<scalar_t *>(a.data());
            out->size = a.size();
            out->numpy_handle = a.data();
        },
        nb::keep_alive<2, 1>());

    m.def("fill", Fill);
    m.def("compact", Compact);

    m.def("ewise_setitem", EwiseSetitem);
    m.def("ewise_add", EwiseAdd);
    m.def("ewise_mul", EwiseMul);
    m.def("ewise_div", EwiseDiv);
    m.def("ewise_maximum", EwiseMaximum);
    m.def("ewise_eq", EwiseEq);
    m.def("ewise_ge", EwiseGe);
    m.def("ewise_log", EwiseLog);
    m.def("ewise_exp", EwiseExp);
    m.def("ewise_tanh", EwiseTanh);

    m.def("scalar_setitem", ScalarSetitem);
    m.def("scalar_add", ScalarAdd);
    m.def("scalar_mul", ScalarMul);
    m.def("scalar_div", ScalarDiv);
    m.def("scalar_power", ScalarPower);
    m.def("scalar_maximum", ScalarMaximum);
    m.def("scalar_eq", ScalarEq);
    m.def("scalar_ge", ScalarGe);

    m.def("matmul", Matmul);
    m.def("matmul_tiled", MatmulTiled);

    m.def("reduce_max", ReduceMax);
    m.def("reduce_sum", ReduceSum);
}
