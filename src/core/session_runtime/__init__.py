"""Session runtime (core orchestration).

This package implements a single-session runtime with:
- serialized message processing (one-at-a-time)
- unified cancellation (drop queued + cancel in-flight)
- engine-agnostic runner orchestration (Claude/Pi/Debate)

Transport (XMPP) and persistence are injected via ports.
"""

from src.core.session_runtime.runtime import SessionRuntime

__all__ = ["SessionRuntime"]
