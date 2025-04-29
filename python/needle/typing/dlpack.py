from enum import IntEnum
from typing import Protocol, runtime_checkable


class DLPackDeviceType(IntEnum):
    """
    DLPack device types.
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack_device__.html#array_api.array.__dlpack_device__
    """

    CPU = 1
    CUDA = 2
    CPU_PINNED = 3
    OPENCL = 4
    VULKAN = 7
    METAL = 8
    VPI = 9
    ROCM = 10
    CUDA_MANAGED = 13
    ONE_API = 14


type DLPackDeviceId = int


@runtime_checkable
class SupportsDLPack(Protocol):
    """Protocol for objects that support DLPack."""

    def __dlpack__(
        self,
        *,
        max_version: tuple[int, int] = (2024, 12),
        stream: int | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool = False,
    ) -> object:
        """Export array as a DLPack capsule.

        Args:
            max_version (tuple[int, int], optional): Maximum DLPack version to use.
                Defaults to (2024, 12).
            stream (int | None, optional): Stream ID for the DLPack device.
                Defaults to None.
            dl_device (tuple[int, int] | None, optional):
                Device ID and type for the DLPack device.
                Defaults to None.
            copy (bool, optional): Whether to copy the data.
                Defaults to False.

        Returns:
            A DLPack capsule that can be consumed by other frameworks.
            The capsule owns a copy of the array data to ensure safety.
        """
        ...

    def __dlpack_device__(self) -> tuple[DLPackDeviceType, DLPackDeviceId]:
        """
        Returns a tuple of (device_type, device_id) representing the DLPack device.

        Device types follow DLPack:

        Returns:
            tuple: (device_type, device_id)
        """
        ...
