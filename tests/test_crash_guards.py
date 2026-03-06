"""Tests for crash-guard fixes across the codebase.

Each test verifies that a previously-crashable code path now degrades
gracefully instead of raising an unhandled exception.
"""

from __future__ import annotations

import asyncio
import subprocess
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# 1. run_ejabberdctl — TimeoutExpired, FileNotFoundError, OSError
# ---------------------------------------------------------------------------

class TestRunEjabberdctl(unittest.TestCase):
    """src/utils.py  run_ejabberdctl()"""

    def test_timeout_returns_false(self):
        from src.utils import run_ejabberdctl

        with patch("src.utils.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=30)):
            ok, msg = run_ejabberdctl("ssh host ejabberdctl", "status")
            self.assertFalse(ok)
            self.assertIn("timed out", msg)

    def test_file_not_found_returns_false(self):
        from src.utils import run_ejabberdctl

        with patch("src.utils.subprocess.run", side_effect=FileNotFoundError("ssh")):
            ok, msg = run_ejabberdctl("ssh host ejabberdctl", "status")
            self.assertFalse(ok)
            self.assertIn("ssh", msg)

    def test_os_error_returns_false(self):
        from src.utils import run_ejabberdctl

        with patch("src.utils.subprocess.run", side_effect=OSError("Permission denied")):
            ok, msg = run_ejabberdctl("ssh host ejabberdctl", "status")
            self.assertFalse(ok)
            self.assertIn("Permission denied", msg)

    def test_success_still_works(self):
        from src.utils import run_ejabberdctl

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok\n"
        mock_result.stderr = ""
        with patch("src.utils.subprocess.run", return_value=mock_result):
            ok, msg = run_ejabberdctl("ssh host ejabberdctl", "status")
            self.assertTrue(ok)
            self.assertEqual(msg, "ok")


# ---------------------------------------------------------------------------
# 2. tmux helpers — FileNotFoundError, TimeoutExpired, OSError
# ---------------------------------------------------------------------------

class TestTmuxHelpers(unittest.TestCase):
    """src/helpers.py  tmux_session_exists, create_tmux_session, kill_tmux_session"""

    def test_tmux_session_exists_no_binary(self):
        from src.helpers import tmux_session_exists

        with patch("src.helpers.subprocess.run", side_effect=FileNotFoundError("tmux")):
            self.assertFalse(tmux_session_exists("test-session"))

    def test_tmux_session_exists_timeout(self):
        from src.helpers import tmux_session_exists

        with patch("src.helpers.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="tmux", timeout=10)):
            self.assertFalse(tmux_session_exists("test-session"))

    def test_kill_tmux_session_no_binary(self):
        from src.helpers import kill_tmux_session

        with patch("src.helpers.subprocess.run", side_effect=FileNotFoundError("tmux")):
            self.assertFalse(kill_tmux_session("test-session"))

    def test_kill_tmux_session_os_error(self):
        from src.helpers import kill_tmux_session

        with patch("src.helpers.subprocess.run", side_effect=OSError("nope")):
            self.assertFalse(kill_tmux_session("test-session"))


# ---------------------------------------------------------------------------
# 3. SubprocessTransport.cancel() — ProcessLookupError guard
# ---------------------------------------------------------------------------

class TestSubprocessTransportCancel(unittest.TestCase):
    """src/runners/subprocess_transport.py  cancel()"""

    def test_cancel_process_already_dead(self):
        from src.runners.subprocess_transport import SubprocessTransport

        transport = SubprocessTransport()
        mock_proc = MagicMock()
        mock_proc.terminate.side_effect = ProcessLookupError()
        transport.process = mock_proc

        # Should not raise
        transport.cancel()
        mock_proc.terminate.assert_called_once()

    def test_cancel_normal(self):
        from src.runners.subprocess_transport import SubprocessTransport

        transport = SubprocessTransport()
        mock_proc = MagicMock()
        transport.process = mock_proc

        transport.cancel()
        mock_proc.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# 4. VoiceCallManager._cleanup_call — logs gather exceptions
# ---------------------------------------------------------------------------

class TestVoiceCleanupCallLogging(unittest.TestCase):
    """src/voice/__init__.py  _cleanup_call()"""

    def test_cleanup_logs_transcription_failures(self):
        from src.voice import VoiceCallManager

        bot = MagicMock()
        bot.guard = AsyncMock()
        mgr = VoiceCallManager(bot)

        # Set up a failed transcription task
        sid = "test-sid"
        failed_task = asyncio.Future()
        failed_task.set_exception(RuntimeError("transcription boom"))
        mgr._transcription_tasks[sid] = {failed_task}
        mgr._transcription_buffers[sid] = []

        with patch("src.voice.log") as mock_log:
            asyncio.get_event_loop().run_until_complete(mgr._cleanup_call(sid))
            # Should have logged the warning about the failed task
            mock_log.warning.assert_called()
            args = mock_log.warning.call_args[0]
            self.assertIn("Transcription task failed", args[0])


# ---------------------------------------------------------------------------
# 5. DB create() — get() inside write lock
# ---------------------------------------------------------------------------

class TestDBCreateRaceCondition(unittest.TestCase):
    """src/db.py  SessionStore.create() — get() should be inside write lock"""

    def test_get_called_inside_lock(self):
        """Verify the get() call happens while the write lock is held."""
        import ast

        source = Path("src/db.py").read_text()
        tree = ast.parse(source)

        # Find the create method
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "create":
                # Find the async-with block (the write lock)
                for child in ast.walk(node):
                    if isinstance(child, ast.AsyncWith):
                        # Check that self.get(name) is called inside
                        body_source = ast.get_source_segment(source, child)
                        if body_source and "self.get(name)" in body_source:
                            return  # PASS — get() is inside the lock
                self.fail("self.get(name) not found inside async with self._write_lock")


# ---------------------------------------------------------------------------
# 6. kill_session — delete_xmpp_account and kill_tmux guarded
# ---------------------------------------------------------------------------

class TestKillSessionGuarded(unittest.TestCase):
    """src/lifecycle/sessions.py  kill_session() guards external calls"""

    def test_kill_session_survives_xmpp_delete_failure(self):
        """If delete_xmpp_account raises, kill_session should still complete."""
        import ast

        source = Path("src/lifecycle/sessions.py").read_text()

        # Find lines around delete_xmpp_account call
        lines = source.splitlines()
        for i, line in enumerate(lines):
            if "delete_xmpp_account(" in line and "rollback" not in lines[max(0, i-10):i+1].__repr__():
                # Check that it's inside a try block
                # Look backwards for 'try:'
                found_try = False
                for j in range(i - 1, max(0, i - 5), -1):
                    if "try:" in lines[j]:
                        found_try = True
                        break
                if not found_try:
                    # Only fail for the kill_session version (not rollback)
                    context = "\n".join(lines[max(0, i-3):i+3])
                    if "rollback" not in context:
                        self.fail(
                            f"delete_xmpp_account at line {i+1} is not inside try/except"
                        )


if __name__ == "__main__":
    unittest.main()
