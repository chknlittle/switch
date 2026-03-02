"""Voice call support via Jingle + faster-whisper.

Accepts incoming XMPP voice calls (Jingle), decodes audio via aiortc,
transcribes speech locally with faster-whisper, and feeds the text into
the session as regular messages. Responses come back as XMPP text (no TTS).

Guarded by SWITCH_VOICE_ENABLED=1 — when disabled, nothing is imported
or registered.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from src.voice.audio_buffer import AudioBuffer
from src.voice.jingle import (
    VOICE_DISCO_FEATURES,
    NS_JINGLE,
    build_iq_result,
    build_session_accept,
    build_session_terminate,
    get_jingle_action,
    parse_session_initiate,
    parse_session_terminate,
    parse_transport_info,
)
from src.voice.media import accept_call, jingle_candidate_to_rtc

if TYPE_CHECKING:
    from aiortc import RTCPeerConnection

    from src.core.session_runtime.api import SessionPort

log = logging.getLogger("voice")


def voice_enabled() -> bool:
    """Check if voice call support is enabled via environment."""
    return os.getenv("SWITCH_VOICE_ENABLED", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class VoiceCallManager:
    """Manages voice calls for a single SessionBot.

    Lifecycle:
        1. Created in SessionBot.__init__ (if voice enabled)
        2. register_handlers() called in on_start
        3. Incoming Jingle IQs routed here by the handler
        4. Audio transcribed and enqueued into the session
        5. shutdown() called on bot teardown
    """

    def __init__(
        self,
        bot: Any,  # SessionBot — avoid circular import
        session: "SessionPort",
    ):
        self._bot = bot
        self._session = session
        self._active_calls: dict[str, RTCPeerConnection] = {}  # sid → pc
        self._audio_buffers: dict[str, AudioBuffer] = {}  # sid → buffer
        self._shutting_down = False

    def register_handlers(self) -> None:
        """Register Jingle IQ handler and disco features on the bot."""
        from slixmpp.xmlstream import MatchXPath
        from slixmpp.xmlstream.handler import Callback

        # Register handler for all Jingle IQ stanzas
        self._bot.register_handler(
            Callback(
                "Jingle IQ",
                MatchXPath(f"{{jabber:client}}iq/{{{NS_JINGLE}}}jingle"),
                self._handle_jingle_iq,
            )
        )

        # Register disco features so Conversations shows the call button.
        # xep_0030 must be registered on the bot first.
        try:
            self._bot.register_plugin("xep_0030")
        except Exception:
            pass  # Already registered

        disco = self._bot["xep_0030"]
        for feature in VOICE_DISCO_FEATURES:
            try:
                disco.add_feature(feature)
            except Exception:
                log.debug("Failed to add disco feature: %s", feature)

        log.info("Voice call handlers registered (%d disco features)", len(VOICE_DISCO_FEATURES))

    def _handle_jingle_iq(self, iq: Any) -> None:
        """Dispatch incoming Jingle IQs by action."""
        asyncio.ensure_future(
            self._bot.guard(
                self._async_handle_jingle(iq),
                context="voice.jingle",
            )
        )

    async def _async_handle_jingle(self, iq: Any) -> None:
        """Route Jingle IQ to the appropriate handler."""
        action = get_jingle_action(iq)
        if action is None:
            return

        if action == "session-initiate":
            await self._on_session_initiate(iq)
        elif action == "transport-info":
            await self._on_transport_info(iq)
        elif action == "session-terminate":
            await self._on_session_terminate(iq)
        else:
            # Ack unknown actions to avoid error responses
            reply = build_iq_result(iq)
            reply.send()
            log.debug("Ignored Jingle action: %s", action)

    async def _on_session_initiate(self, iq: Any) -> None:
        """Handle incoming call (Jingle session-initiate)."""
        # Auth check: only accept calls from the session owner
        sender = str(iq["from"].bare)
        if sender != self._bot.xmpp_recipient:
            log.warning(
                "Rejected call from unauthorized sender: %s (expected %s)",
                sender,
                self._bot.xmpp_recipient,
            )
            # Send error IQ
            reply = iq.reply()
            reply["type"] = "error"
            reply.send()
            return

        offer = parse_session_initiate(iq)
        if offer is None:
            log.warning("Failed to parse Jingle session-initiate")
            reply = iq.reply()
            reply["type"] = "error"
            reply.send()
            return

        if offer.sid in self._active_calls:
            log.warning("Duplicate session-initiate for sid=%s", offer.sid)
            reply = build_iq_result(iq)
            reply.send()
            return

        log.info("Incoming voice call from %s (sid=%s)", sender, offer.sid)

        # Ack the initiate IQ immediately
        reply = build_iq_result(iq)
        reply.send()

        # Set up audio buffer with transcription callback
        sid = offer.sid

        def on_segment(pcm: bytes, sample_rate: int) -> None:
            asyncio.ensure_future(self._transcribe_segment(sid, pcm, sample_rate))

        audio_buf = AudioBuffer(on_segment)
        self._audio_buffers[sid] = audio_buf

        # Accept the call via WebRTC
        try:
            pc, jingle_info = await accept_call(
                offer,
                on_audio_frame=audio_buf.push_frame,
            )
        except Exception:
            log.exception("Failed to accept call (sid=%s)", sid)
            self._audio_buffers.pop(sid, None)
            # Send session-terminate with error
            self._send_jingle_terminate(iq["from"], sid, reason="failed-application")
            return

        self._active_calls[sid] = pc

        # Send session-accept IQ
        accept_jingle = build_session_accept(offer, **jingle_info)
        accept_iq = self._bot.make_iq_set(ito=str(iq["from"]))
        accept_iq.xml.append(accept_jingle)
        try:
            await accept_iq.send()
        except Exception:
            log.exception("Failed to send session-accept IQ")
            await self._cleanup_call(sid)
            return

        self._bot.send_reply("[Voice call started]")
        log.info("Voice call accepted (sid=%s)", sid)

    async def _on_transport_info(self, iq: Any) -> None:
        """Handle trickle ICE candidates."""
        reply = build_iq_result(iq)
        reply.send()

        result = parse_transport_info(iq)
        if result is None:
            return

        sid, candidates = result
        pc = self._active_calls.get(sid)
        if pc is None:
            log.debug("transport-info for unknown sid=%s", sid)
            return

        for c in candidates:
            try:
                await pc.addIceCandidate(jingle_candidate_to_rtc(c))
            except Exception:
                log.debug("Failed to add trickle candidate: %s:%s", c.ip, c.port)

    async def _on_session_terminate(self, iq: Any) -> None:
        """Handle remote hangup."""
        reply = build_iq_result(iq)
        reply.send()

        sid = parse_session_terminate(iq)
        if sid is None:
            return

        log.info("Remote hangup (sid=%s)", sid)
        await self._cleanup_call(sid)
        self._bot.send_reply("[Voice call ended]")

    async def _transcribe_segment(
        self, sid: str, pcm: bytes, sample_rate: int
    ) -> None:
        """Transcribe an audio segment and enqueue the text into the session."""
        if self._shutting_down:
            return
        if sid not in self._active_calls:
            return

        from src.voice.transcriber import transcribe

        try:
            text = await transcribe(pcm, sample_rate)
        except Exception:
            log.exception("Transcription failed")
            return

        if not text:
            return

        log.info("Voice transcription: %s", text[:100])
        await self._session.enqueue(
            text,
            None,
            trigger_response=True,
            scheduled=False,
            wait=False,
        )

    def _send_jingle_terminate(
        self, to: Any, sid: str, reason: str = "success"
    ) -> None:
        """Send a Jingle session-terminate IQ."""
        terminate = build_session_terminate(sid, reason=reason)
        term_iq = self._bot.make_iq_set(ito=str(to))
        term_iq.xml.append(terminate)
        try:
            term_iq.send()
        except Exception:
            log.debug("Failed to send session-terminate", exc_info=True)

    async def _cleanup_call(self, sid: str) -> None:
        """Clean up a single call's resources."""
        # Flush remaining audio
        buf = self._audio_buffers.pop(sid, None)
        if buf:
            try:
                buf.flush_remaining()
            except Exception:
                log.debug("Error flushing audio buffer", exc_info=True)

        pc = self._active_calls.pop(sid, None)
        if pc:
            try:
                await pc.close()
            except Exception:
                log.debug("Error closing peer connection", exc_info=True)

    # -----------------------------------------------------------------
    # Public API for SessionBot
    # -----------------------------------------------------------------

    @property
    def active_call_count(self) -> int:
        return len(self._active_calls)

    @property
    def active_call_sids(self) -> list[str]:
        return list(self._active_calls.keys())

    async def hangup_all(self) -> None:
        """Hang up all active calls."""
        sids = list(self._active_calls.keys())
        for sid in sids:
            pc = self._active_calls.get(sid)
            if pc:
                # Try to notify remote side
                # We don't have the remote JID stored, so just close locally
                pass
            await self._cleanup_call(sid)

    async def shutdown(self) -> None:
        """Shut down the voice call manager."""
        self._shutting_down = True
        await self.hangup_all()
        log.info("Voice call manager shut down")
