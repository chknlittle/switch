"""
Base XMPP bot shared by the dispatcher, directory, and session bots.
"""

import asyncio
import logging

from slixmpp.clientxmpp import ClientXMPP

from src.utils import build_message_meta


class BaseXMPPBot(ClientXMPP):
    """
    Base class for all XMPP bots with common setup.

    Provides:
    - Standard plugin registration (xep_0199, xep_0085, xep_0280)
    - Unencrypted plain auth setup
    - Common connect method
    - send_reply and send_typing helpers
    """

    def __init__(self, jid: str, password: str, recipient: str | None = None):
        super().__init__(jid, password)
        self.recipient = recipient
        self._connected_event = asyncio.Event()
        self._last_connect_target: tuple[str, int] | None = None

        # Common plugins
        self.register_plugin("xep_0199")  # Ping
        self.register_plugin("xep_0085")  # Chat State Notifications
        self.register_plugin("xep_0280")  # Message Carbons
        self.register_plugin("xep_0030")  # Service Discovery
        self.register_plugin("xep_0115")  # Entity Capabilities (caps in presence)

    def _prepare_connection_settings(self) -> None:
        """Apply Switch's standard XMPP transport settings."""
        self["feature_mechanisms"].unencrypted_plain = True  # type: ignore[attr-defined]
        self.enable_starttls = False
        self.enable_direct_tls = False
        self.enable_plaintext = True

    def _consume_connect_result(
        self, result: object, *, server: str, port: int
    ) -> None:
        if not (asyncio.iscoroutine(result) or isinstance(result, asyncio.Future)):
            return

        task = asyncio.ensure_future(result)

        def _done(t: asyncio.Future) -> None:
            try:
                t.result()
            except asyncio.CancelledError:
                return
            except Exception:
                log = getattr(self, "log", logging.getLogger("xmpp"))
                log.warning(
                    "XMPP connect failed for %s:%s",
                    server,
                    port,
                    exc_info=True,
                )

        task.add_done_callback(_done)

    def connect_to_server(self, server: str, port: int = 5222):
        """Connect with standard settings (unencrypted, no TLS)."""
        self._last_connect_target = (server, port)
        self._prepare_connection_settings()
        result = self.connect(server, port)
        self._consume_connect_result(result, server=server, port=port)

    def reconnect_to_server(self) -> None:
        """Reconnect using the last known server/port and settings."""
        target = self._last_connect_target
        if not target:
            raise RuntimeError("No prior XMPP server configured for reconnect")
        server, port = target
        self._prepare_connection_settings()
        result = self.connect(server, port)
        self._consume_connect_result(result, server=server, port=port)

    def set_connected(self, connected: bool) -> None:
        if connected:
            self._connected_event.set()
        else:
            self._connected_event.clear()

    def is_connected(self) -> bool:
        return self._connected_event.is_set()

    async def wait_connected(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def send_reply(
        self,
        text: str,
        recipient: str | None = None,
        *,
        meta_type: str | None = None,
        meta_tool: str | None = None,
        meta_attrs: dict[str, str] | None = None,
        meta_payload: object | None = None,
    ):
        """Send a chat message to recipient."""
        to = recipient or self.recipient
        if not to:
            raise ValueError("No recipient specified")
        msg = self.make_message(mto=to, mbody=text, mtype="chat")
        msg["chat_state"] = "active"

        # Optional message metadata extension.
        if meta_type:
            meta = build_message_meta(
                meta_type,
                meta_tool=meta_tool,
                meta_attrs=meta_attrs,
                meta_payload=meta_payload,
            )
            msg.xml.append(meta)

        msg.send()

    def send_typing(self, recipient: str | None = None):
        """Send composing (typing) indicator."""
        to = recipient or self.recipient
        if not to:
            return
        msg = self.make_message(mto=to, mtype="chat")
        msg["chat_state"] = "composing"
        msg.send()

    def _format_exception_for_user(self, exc: BaseException) -> str:
        msg = str(exc).strip()
        if msg:
            return f"Error: {type(exc).__name__}: {msg}"
        return f"Error: {type(exc).__name__}"

    async def guard(
        self,
        coro,
        *,
        recipient: str | None = None,
        context: str | None = None,
    ):
        """Run a coroutine with a single error boundary.

        - Lets internal code raise normally.
        - Catches at the boundary, logs, and sends an error message to the
          relevant recipient.
        """

        try:
            return await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log = getattr(self, "log", logging.getLogger("xmpp"))
            if context:
                log.exception("Unhandled error (%s)", context)
            else:
                log.exception("Unhandled error")
            try:
                self.send_reply(
                    self._format_exception_for_user(exc), recipient=recipient
                )
            except Exception:
                pass
            return None

    def spawn_guarded(
        self,
        coro,
        *,
        recipient: str | None = None,
        context: str | None = None,
    ) -> asyncio.Task:
        """Create a task that reports exceptions to the user."""

        task = asyncio.create_task(
            self.guard(coro, recipient=recipient, context=context)
        )
        return task
