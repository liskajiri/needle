"""Custom exceptions for the Needle library."""


class NeedleError(Exception):
    """Base exception class for Needle."""

    pass


class BroadcastError(NeedleError):
    """Raised when arrays cannot be broadcast together."""

    def __init__(self, shapes):
        message = f"Incompatible shapes for broadcasting: {shapes}"
        super().__init__(message)
