"""WebRTC media layer — aiortc peer connection with Jingle↔SDP bridge.

Converts Jingle XML signaling into SDP that aiortc understands,
manages the RTCPeerConnection lifecycle, and delivers decoded audio
frames to a callback.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from typing import Any, Callable

from aiortc import (
    RTCConfiguration,
    RTCIceCandidate,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import AudioFrame

from src.voice.jingle import (
    IceCandidate,
    JingleOffer,
    build_session_accept,
)

log = logging.getLogger("voice.media")

# Default STUN server; override via env var.
_DEFAULT_STUN = "stun:stun.l.google.com:19302"


def _ice_servers(turn_user: str = "", turn_pass: str = "") -> list[RTCIceServer]:
    """Build ICE server list from environment + optional credentials.

    Env vars:
        SWITCH_STUN_SERVER  — STUN URI  (default: Google public STUN)
                              Set to "none" to disable (e.g. Tailscale direct routing)
        SWITCH_TURN_SERVER  — TURN URI  (e.g. turn:example.com:3478)
        SWITCH_TURN_USER    — TURN username  (overridden by turn_user param)
        SWITCH_TURN_PASS    — TURN credential (overridden by turn_pass param)
    """
    servers: list[RTCIceServer] = []

    stun = os.getenv("SWITCH_STUN_SERVER", _DEFAULT_STUN).strip()
    if stun.lower() in ("none", "off", ""):
        log.info("STUN disabled — using host candidates only")
    else:
        stun = stun or _DEFAULT_STUN
        servers.append(RTCIceServer(urls=[stun]))

    turn = os.getenv("SWITCH_TURN_SERVER", "").strip()
    if turn and turn.lower() not in ("none", "off"):
        user = turn_user or os.getenv("SWITCH_TURN_USER", "").strip()
        cred = turn_pass or os.getenv("SWITCH_TURN_PASS", "").strip()
        servers.append(RTCIceServer(urls=[turn], username=user, credential=cred))
        log.info("TURN server configured: %s (user=%s)", turn, user or "<none>")

    if not servers:
        log.info("No ICE servers — host candidates only (direct routing)")

    return servers


# ---------------------------------------------------------------------------
# Jingle → SDP conversion
# ---------------------------------------------------------------------------

def _jingle_offer_to_sdp(offer: JingleOffer) -> str:
    """Convert a JingleOffer into an SDP offer string.

    This produces a minimal SDP that aiortc can parse as a remote offer.
    The fiddliest part of the whole voice stack — Conversations' Jingle
    dialect may need iteration here.
    """
    lines: list[str] = [
        "v=0",
        f"o=- {_sdp_session_id(offer.sid)} 2 IN IP4 0.0.0.0",
        "s=-",
        "t=0 0",
        "a=group:BUNDLE audio",
        "a=msid-semantic: WMS *",
    ]

    # Media line — list all payload types
    pt_ids = " ".join(pt_id for pt_id, _, _, _ in offer.codecs) or "111"
    lines.append(f"m=audio 9 UDP/TLS/RTP/SAVPF {pt_ids}")
    lines.append("c=IN IP4 0.0.0.0")

    # ICE credentials
    lines.append(f"a=ice-ufrag:{offer.ice_ufrag}")
    lines.append(f"a=ice-pwd:{offer.ice_pwd}")

    # DTLS fingerprint
    if offer.dtls_fingerprint:
        lines.append(f"a=fingerprint:{offer.dtls_hash} {offer.dtls_fingerprint}")
        lines.append(f"a=setup:{offer.dtls_setup}")

    lines.append("a=mid:audio")
    lines.append("a=sendrecv")
    lines.append("a=rtcp-mux")

    # Codec descriptions
    for pt_id, name, clockrate, channels in offer.codecs:
        ch_str = f"/{channels}" if channels and channels != "1" else ""
        lines.append(f"a=rtpmap:{pt_id} {name}/{clockrate}{ch_str}")

    # ICE candidates
    for c in offer.candidates:
        cand_line = (
            f"a=candidate:{c.foundation} {c.component} {c.protocol} {c.priority} "
            f"{c.ip} {c.port} typ {c.type}"
        )
        if c.rel_addr:
            cand_line += f" raddr {c.rel_addr}"
        if c.rel_port:
            cand_line += f" rport {c.rel_port}"
        cand_line += f" generation {c.generation}"
        lines.append(cand_line)

    return "\r\n".join(lines) + "\r\n"


def _sdp_session_id(sid: str) -> str:
    """Deterministic numeric session ID from Jingle sid string."""
    return str(int(hashlib.sha256(sid.encode()).hexdigest()[:12], 16))


# ---------------------------------------------------------------------------
# SDP → Jingle answer conversion
# ---------------------------------------------------------------------------

def _parse_sdp_answer_for_jingle(
    sdp: str,
) -> tuple[str, str, list[IceCandidate], str, str, str]:
    """Parse our local SDP answer back into Jingle-compatible parts.

    Returns (ice_ufrag, ice_pwd, candidates, dtls_fingerprint, dtls_hash, dtls_setup).
    """
    ice_ufrag = ""
    ice_pwd = ""
    dtls_fingerprint = ""
    dtls_hash = "sha-256"
    dtls_setup = "active"
    candidates: list[IceCandidate] = []

    for line in sdp.splitlines():
        line = line.strip()

        if line.startswith("a=ice-ufrag:"):
            ice_ufrag = line.split(":", 1)[1]
        elif line.startswith("a=ice-pwd:"):
            ice_pwd = line.split(":", 1)[1]
        elif line.startswith("a=fingerprint:"):
            parts = line.split(":", 1)[1].split(None, 1)
            if len(parts) == 2:
                h, fp = parts
                # Prefer sha-256 (what Conversations uses); only take
                # other hashes if sha-256 isn't available.
                if h == "sha-256" or not dtls_fingerprint:
                    dtls_hash, dtls_fingerprint = h, fp
        elif line.startswith("a=setup:"):
            dtls_setup = line.split(":", 1)[1]
        elif line.startswith("a=candidate:"):
            cand = _parse_sdp_candidate(line)
            if cand:
                candidates.append(cand)

    return ice_ufrag, ice_pwd, candidates, dtls_fingerprint, dtls_hash, dtls_setup


def _parse_sdp_candidate(line: str) -> IceCandidate | None:
    """Parse a single SDP a=candidate: line into an IceCandidate."""
    # a=candidate:foundation component protocol priority ip port typ type [extensions]
    m = re.match(
        r"a=candidate:(\S+)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\d+)\s+typ\s+(\S+)(.*)",
        line,
    )
    if not m:
        return None

    foundation, component, protocol, priority, ip, port, ctype = m.groups()[:7]
    rest = m.group(8)

    rel_addr = None
    rel_port = None
    generation = "0"

    rm = re.search(r"raddr\s+(\S+)", rest)
    if rm:
        rel_addr = rm.group(1)
    rm = re.search(r"rport\s+(\d+)", rest)
    if rm:
        rel_port = rm.group(1)
    rm = re.search(r"generation\s+(\d+)", rest)
    if rm:
        generation = rm.group(1)

    return IceCandidate(
        component=component,
        foundation=foundation,
        generation=generation,
        id=uuid.uuid4().hex[:8],
        ip=ip,
        network="0",
        port=port,
        priority=priority,
        protocol=protocol.lower(),
        type=ctype,
        rel_addr=rel_addr,
        rel_port=rel_port,
    )


# ---------------------------------------------------------------------------
# Trickle ICE: Jingle candidate → aiortc RTCIceCandidate
# ---------------------------------------------------------------------------

def jingle_candidate_to_rtc(c: IceCandidate) -> RTCIceCandidate:
    """Convert a Jingle IceCandidate to an aiortc RTCIceCandidate."""
    return RTCIceCandidate(
        component=int(c.component),
        foundation=c.foundation,
        ip=c.ip,
        port=int(c.port),
        priority=int(c.priority),
        protocol=c.protocol,
        type=c.type,
        relatedAddress=c.rel_addr,
        relatedPort=int(c.rel_port) if c.rel_port else None,
    )


# ---------------------------------------------------------------------------
# PeerConnection lifecycle
# ---------------------------------------------------------------------------

# Type for the callback that receives decoded audio frames.
AudioFrameCallback = Callable[[AudioFrame], None]


async def accept_call(
    offer: JingleOffer,
    on_audio_frame: AudioFrameCallback,
    turn_user: str = "",
    turn_pass: str = "",
) -> tuple[RTCPeerConnection, dict[str, Any]]:
    """Accept an incoming Jingle voice call.

    1. Creates an RTCPeerConnection with STUN/TURN config
    2. Sets the remote SDP offer
    3. Creates and sets a local answer
    4. Hooks up the audio track to deliver frames via callback
    5. Returns (pc, jingle_answer_info) where jingle_answer_info has
       the fields needed for build_session_accept()

    The caller is responsible for sending the Jingle session-accept IQ.
    """
    # Enable debug logging for aiortc internals during the call
    for _mod in ("aiortc", "aioice", "aiortc.rtcdtlstransport", "aiortc.rtcpeerconnection"):
        logging.getLogger(_mod).setLevel(logging.DEBUG)

    config = RTCConfiguration(iceServers=_ice_servers(turn_user, turn_pass))
    pc = RTCPeerConnection(configuration=config)

    @pc.on("connectionstatechange")
    def on_connstate():
        log.info("PeerConnection state: %s", pc.connectionState)

    @pc.on("iceconnectionstatechange")
    def on_icestate():
        log.info("ICE connection state: %s", pc.iceConnectionState)

    # Hook audio track
    @pc.on("track")
    def on_track(track):
        if track.kind != "audio":
            return
        log.info("Audio track received: %s", track.id)

        async def _consume():
            from aiortc.mediastreams import MediaStreamError
            frame_count = 0
            try:
                while True:
                    frame = await track.recv()
                    frame_count += 1
                    if frame_count <= 3 or frame_count % 200 == 0:
                        log.info("_consume frame #%d: samples=%s rate=%s format=%s",
                                 frame_count, frame.samples, frame.sample_rate, frame.format.name)
                    try:
                        on_audio_frame(frame)
                    except Exception:
                        log.exception("Error in audio frame callback")
            except MediaStreamError:
                log.info("Audio track ended normally (received %d frames)", frame_count)
            except Exception:
                log.exception("Audio consume loop crashed after %d frames", frame_count)

        import asyncio
        asyncio.ensure_future(_consume())

    # Set remote offer
    sdp_offer = _jingle_offer_to_sdp(offer)
    log.info("Generated SDP offer:\n%s", sdp_offer)
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=sdp_offer, type="offer")
    )

    # Add a silent outbound audio track so the SDP answer says "sendrecv"
    # instead of "recvonly". Conversations won't send RTP until it sees
    # media flowing from us.
    from aiortc.mediastreams import AudioStreamTrack
    silence_track = AudioStreamTrack()
    pc.addTrack(silence_track)

    # Add remote ICE candidates
    for c in offer.candidates:
        try:
            await pc.addIceCandidate(jingle_candidate_to_rtc(c))
        except Exception:
            log.debug("Failed to add initial candidate: %s:%s", c.ip, c.port)

    # Create and set local answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    local_sdp = pc.localDescription.sdp
    log.info("Local SDP answer:\n%s", local_sdp)

    # Parse our answer SDP back into Jingle fields
    (
        local_ufrag,
        local_pwd,
        local_candidates,
        local_dtls_fp,
        local_dtls_hash,
        local_dtls_setup,
    ) = _parse_sdp_answer_for_jingle(local_sdp)

    # Filter to Opus codec only (what Conversations expects)
    accepted_codecs: list[tuple[str, str, str, str]] = []
    for pt_id, name, clockrate, channels in offer.codecs:
        if name.lower() == "opus":
            accepted_codecs.append((pt_id, name, clockrate, channels))
    if not accepted_codecs:
        # Fall back to all offered codecs
        accepted_codecs = list(offer.codecs)

    jingle_info = {
        "local_ufrag": local_ufrag,
        "local_pwd": local_pwd,
        "local_candidates": local_candidates,
        "local_dtls_fingerprint": local_dtls_fp,
        "local_dtls_hash": local_dtls_hash,
        "local_dtls_setup": local_dtls_setup,
        "accepted_codecs": accepted_codecs,
    }

    return pc, jingle_info
