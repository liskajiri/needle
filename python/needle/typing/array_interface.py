from typing import NotRequired, TypedDict


class ArrayInterface(TypedDict):
    """Array interface for NumPy interop.

    Details: https://numpy.org/doc/stable/reference/arrays.interface.html
    """

    version: int
    shape: tuple[int, ...]
    typestr: str
    descr: NotRequired[list[tuple[str, str] | tuple[str, str, str]]]
    data: NotRequired[tuple[int, bool]]
    strides: NotRequired[tuple[int, ...]]
    mask: NotRequired[int]
    offset: NotRequired[int]
