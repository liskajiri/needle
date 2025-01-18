# needle

## TODOs

- [ ] Refactor git history
- [ ] Decompose cpu and CUDA backend files

### Tests

- [x] mark slow tests
- [ ] improve memory efficiency test of optimizers
- [ ] Codecov
- [ ] automatic proptesting based on types [https://timothycrosley.github.io/hypothesis-auto/]
- [ ] Hypothesis testing / proptesting
  - are there any other cool testing strategies?
- [ ] Numpy_assert_equal: By default, these assertion functions only compare the numerical values in the arrays. Consider using the strict=True option to check the array dtype and shape, too.
- [ ] Use [https://github.com/tarpas/pytest-testmon]

### Benchmarks

- [ ] parca - eBPF (<https://github.com/parca-dev/parca>)
- [ ] Add codspeed with pytest
- [ ] profiling: Scalene / py-spy
- [ ] Try Airspeed Velocity
- [ ] the c++ parallel std

### Docs

- [ ] Docs - mkdocs
  - [] Combine Doxygen with mkdocs (<https://github.com/JakubAndrysek/mkdoxy>)
  - <https://github.com/mkdocs/catalog?tab=readme-ov-file#-api-documentation-building>
  - autoDocstring - Python Docstring Generator
- [ ] type hints - strict types VS code setting
- [ ] add types automatically - including numpy shapes
  - <https://github.com/RightTyper/RightTyper>
- [ ] set up MyPy / PyRight / ruff typing if available
- [x] Add more ruff rules
- [ ] Proper readme

### Implementation

- Implement buffer protocol / array_interface / dlpack for Numpy/Pytorch interop
  - [https://docs.python.org/3/c-api/buffer.html]
  - [https://numpy.org/devdocs/user/basics.interoperability.html]
  - <https://krokotsch.eu/posts/deep-learning-unit-tests/>

### Build

- [ ] Use pixi-build
- [ ] improve imports - better separation
  - <https://github.com/Erotemic/mkinit>
- [ ] tighten up dependencies
  - [ ] Get rid of numpy as an external dependency - only optional
- [ ] release packages
  - [ ] conda
  - [ ] pypi
- [ ] use hidden _ for utility functions
- examples: notebooks - fully static wasm notebook
  - [ ] runnable notebook - marimo or google colab

## Far TODOs

- [ ] Triton backend
- [ ] Mojo backend
- [ ] Try using PyPy / some other jit compiler like numba
- [ ] torch.compile() like interface
