"""Ralph loop mixin for SessionBot."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.db import RalphLoopRepository
    from src.ralph import RalphLoop


class RalphMixin:
    """Mixin providing Ralph loop state and execution for SessionBot."""

    ralph_loops: "RalphLoopRepository"
    ralph_loop: "RalphLoop | None"

    def init_ralph(self, ralph_loops: "RalphLoopRepository") -> None:
        """Initialize Ralph-related state."""
        self.ralph_loops = ralph_loops
        self.ralph_loop = None

    async def run_ralph(self) -> None:
        """Run the Ralph loop, handling errors and cleanup."""
        if not self.ralph_loop:
            return
        try:
            await self.ralph_loop.run()
        except Exception as e:
            self.log.exception("Ralph loop error")  # type: ignore[attr-defined]
            self.send_reply(f"Ralph crashed: {e}")  # type: ignore[attr-defined]
        finally:
            self.ralph_loop = None
            self.processing = False  # type: ignore[attr-defined]
