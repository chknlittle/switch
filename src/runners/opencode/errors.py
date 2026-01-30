"""OpenCode-specific exceptions.

These exception types make it easier for higher-level error handling (e.g. the
dispatcher/session layer) to format failures consistently without scraping
strings.
"""

from __future__ import annotations


class OpenCodeError(RuntimeError):
    """Base class for OpenCode runner/client errors."""


class OpenCodeHTTPError(OpenCodeError):
    """HTTP error from the OpenCode server."""

    def __init__(
        self,
        status: int,
        *,
        method: str,
        url: str,
        detail: str | None = None,
    ):
        self.status = int(status)
        self.method = method
        self.url = url
        self.detail = detail
        super().__init__(self.__str__())

    def __str__(self) -> str:
        detail = (self.detail or "").strip()
        if detail:
            return f"OpenCode HTTP {self.status} {self.method} {self.url}: {detail}"
        return f"OpenCode HTTP {self.status} {self.method} {self.url}"


class OpenCodeProtocolError(OpenCodeError):
    """Malformed/invalid data from the OpenCode server."""

    def __init__(self, message: str, *, payload_preview: str | None = None):
        self.message = message
        self.payload_preview = payload_preview
        super().__init__(self.__str__())

    def __str__(self) -> str:
        if self.payload_preview:
            return f"OpenCode protocol error: {self.message} (payload={self.payload_preview!r})"
        return f"OpenCode protocol error: {self.message}"
