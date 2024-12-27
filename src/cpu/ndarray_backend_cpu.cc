module;

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <nanobind/nb_cast.h>

#include <algorithm>
#include <vector>

import ndarray;
import elementwise;
import scalar_ops;
import matmul;
import reductions;

export module ndarray_backend_cpu;

// TODO: nb::ndarray can alloc memory cheaply - connect to python?
NB_MODULE(ndarray_backend_cpu, m) {
    namespace nb = nanobind;
    using namespace needle;
    using namespace cpu;

    m.attr("__device_name__") = "cpu";
    m.attr("__tile_size__") = TILE;

    nb::class_<AlignedArray>(m, "Array")
        // TODO: return_value_policy is not present
        .def(nb::init_implicit<size_t>())
        .def("ptr", &AlignedArray::ptr_as_int)
        // read only
        .def_ro("size", &AlignedArray::size);

    // return numpy array - should not be copied
    m.def("to_numpy", [](const AlignedArray &a, const nb::tuple &shape_tuple,
                         const nb::tuple &strides_tuple, const size_t offset) {
        std::vector<size_t> shape;
        for (const auto &dim : shape_tuple) {
            shape.push_back(nb::cast<size_t>(dim));
        }
        // TODO: strides
        return nb::ndarray<nb::numpy, scalar_t>(
            a.ptr + offset, shape_tuple.size(), shape.data(), nullptr);
    });

    // convert from numpy (with copying)
    // TODO: nb::ndarray probably can do it without copy
    m.def("from_numpy", [](nb::ndarray<scalar_t> a, AlignedArray *out) {
        memcpy(out->ptr, a.data(), out->size * ELEM_SIZE);
    });

    // TODO: function Array constraints
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
