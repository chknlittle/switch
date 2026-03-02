"""Jingle signaling — parse/build XML stanzas for XMPP voice calls.

slixmpp doesn't ship xep_0166/0167 plugins, so we handle raw XML directly.
Conversations (Android) uses standard Jingle with ICE-UDP transport and
Opu codec, which is what we target here.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from slixmpp.xmlstream import ET

log = logging.getLogger("voice.jingle")

# Jingle XML namespaces
NS_JINGLE = "urn:xmpp:jingle:1"
NS_JINGLE_RTP = "urn:xmpp:jingle:apps:rtp:1"
NS_JINGLE_RTP_AUDIO = "urn:xmpp:jingle:apps:rtp:audio"
NS_JINGLE_ICE_UDP = "urn:xmpp:jingle:transports:ice-udp:1"
NS_JINGLE_DTLS = "urn:xmpp:jingle:apps:dtls:0"
NS_JINGLE_RTP_HDREXT = "urn:xmpp:jingle:apps:rtp:rtp-hdrext:0"

# Feature URNs that Conversations checks via disco#info before showing
# the call button.  Register these on the bot's full JID.
OICE_CALL_FEATURES = [
    NS_JINGLE,
    NS_JINGLE_RTP,
    NS_JINGLE_RTP_AUDIO,
    NS_JINGLE_ICE_UDP,
    NS_JINGLE_DTLS,
    "urn:xmpp:jingle:apps:rtp:1",
    "urn:xmpp:jingle:apps:rtp:audio",
    "urn:xmpp:jingle:transports:ice-udp:1",
    "urn:xmpp:jingle:apps:dtls:0",
]

# Deduplicated feature list for disco registration
VOICE_DISCO_FEATURES: list[str] = list(dict.fromkeys(OICE_CALL_FEATURES))


@dataclass
class IceCandidate:
    """A single ICE candidate from a Jingle transport."""

    component: str
    foundation: str
    generation: str
    id: str
    ip: str
    network: str
    port: str
    priority: str
    protocol: str
    type: str
    rel_addr: str | None = None
    rel_port: str | None = None


@dataclass
class JingleOffer:
    """Parsed data from a Jingle session-initiate stanza."""

    sid: str
    initiator: str
    responder: str  # our JID
    content_name: str

    # ICE transport
    ice_ufrag: str
    ice_pwd: str
    candidates: list[IceCandidate] = field(default_factory=list)

    # DTLS
    dtls_hash: str = ""
    dtls_fingerprint: str = ""
    dtls_setup: str = "actpass"

    # Codecs — list of (payload_type, codec_name, clockrate, channels)
    codecs: list[tuple[str, str, str, str]] = field(default_factory=list)


def parse_session_initiate(iq_stanza: Any) -> JingleOffer | None:
    """Parse a Jingle session-initiate IQ into a JingleOffer.

    Returns None if the stanza is not a valid session-initiate or lacks
    audio content.
    """
    jingle_el = iq_stanza.xml.find(f"{{{NS_JINGLE}}}jingle")
    if jingle_el is None:
        return None

    action = jingle_el.get("action", "")
    if action != "session-initiate":
        return None

    sid = jingle_el.get("sid", "")
    initiator = jingle_el.get("initiator", "")
    responder = str(iq_stanza["to"])

    # Find audio content
    content_el = None
    for c in jingle_el.findall(f"{{{NS_JINGLE}}}content"):
        desc = c.find(f"{{{NS_JINGLE_RTP}}}description")
        if desc is not None and desc.get("media") == "audio":
            content_el = c
            break

    if content_el is None:
        log.warning("No audio content in Jingle session-initiate")
        return None

    content_name = content_el.get("name", "audio")

    # Parse codecs from description
    desc_el = content_el.find(f"{{{NS_JINGLE_RTP}}}description")
    codecs: list[tuple[str, str, str, str]] = []
    if desc_el is not None:
        for pt in desc_el.findall(f"{{{NS_JINGLE_RTP}}}payload-type"):
            codecs.append((
                pt.get("id", ""),
                pt.get("name", ""),
                pt.get("clockrate", ""),
                pt.get("channels", "1"),
            ))

    # Parse ICE transport
    transport_el = content_el.find(f"{{{NS_JINGLE_ICE_UDP}}}transport")
    ice_ufrag = ""
    ice_pwd = ""
    candidates: list[IceCandidate] = []
    dtls_hash = ""
    dtls_fingerprint = ""
    dtls_setup = "actpass"

    if transport_el is not None:
        ice_ufrag = transport_el.get("ufrag", "")
        ice_pwd = transport_el.get("pwd", "")

        for cand in transport_el.findall(f"{{{NS_JINGLE_ICE_UDP}}}candidate"):
            candidates.append(IceCandidate(
                component=cand.get("component", "1"),
                foundation=cand.get("foundation", ""),
                generation=cand.get("generation", "0"),
                id=cand.get("id", ""),
                ip=cand.get("ip", ""),
                network=cand.get("network", "0"),
                port=cand.get("port", ""),
                priority=cand.get("priority", ""),
                protocol=cand.get("protocol", "udp"),
                type=cand.get("type", "host"),
                rel_addr=cand.get("rel-addr"),
                rel_port=cand.get("rel-port"),
            ))

        # DTLS fingerprint
        fp_el = transport_el.find(f"{{{NS_JINGLE_DTLS}}}fingerprint")
        if fp_el is not None:
            dtls_hash = fp_el.get("hash", "sha-256")
            dtls_fingerprint = (fp_el.text or "").strip()
            dtls_setup = fp_el.get("setup", "actpass")

    return JingleOffer(
        sid=sid,
        initiator=initiator,
        responder=responder,
        content_name=content_name,
        ice_ufrag=ice_ufrag,
        ice_pwd=ice_pwd,
        candidates=candidates,
        dtls_hash=dtls_hash,
        dtls_fingerprint=dtls_fingerprint,
        dtls_setup=dtls_setup,
        codecs=codecs,
    )


def parse_transport_info(iq_stanza: Any) -> tuple[str, list[IceCandidate]] | None:
    """Parse a Jingle transport-info IQ for trickle ICE candidates.

    Returns (sid, candidates) or None if not a valid transport-info.
    """
    jingle_el = iq_stanza.xml.find(f"{{{NS_JINGLE}}}jingle")
    if jingle_el is None:
        return None

    action = jingle_el.get("action", "")
    if action != "transport-info":
        return None

    sid = jingle_el.get("sid", "")
    candidates: list[IceCandidate] = []

    for content in jingle_el.findall(f"{{{NS_JINGLE}}}content"):
        transport = content.find(f"{{{NS_JINGLE_ICE_UDP}}}transport")
        if transport is None:
            continue
        for cand in transport.findall(f"{{{NS_JINGLE_ICE_UDP}}}candidate"):
            candidates.append(IceCandidate(
                component=cand.get("component", "1"),
                foundation=cand.get("foundation", ""),
                generation=cand.get("generation", "0"),
                id=cand.get("id", ""),
                ip=cand.get("ip", ""),
                network=cand.get("network", "0"),
                port=cand.get("port", ""),
                priority=cand.get("priority", ""),
                protocol=cand.get("protocol", "udp"),
                type=cand.get("type", "host"),
                rel_addr=cand.get("rel-addr"),
                rel_port=cand.get("rel-port"),
            ))

    return (sid, candidates)


def parse_session_terminate(iq_stanza: Any) -> str | None:
    """Parse a Jingle session-terminate IQ. Returns the session ID or None."""
    jingle_el = iq_stanza.xml.find(f"{{{NS_JINGLE}}}jingle")
    if jingle_el is None:
        return None

    action = jingle_el.get("action", "")
    if action != "session-terminate":
        return None

    return jingle_el.get("sid", "")


def get_jingle_action(iq_stanza: Any) -> str | None:
    """Extract the Jingle action from an IQ stanza, or None."""
    jingle_el = iq_stanza.xml.find(f"{{{NS_JINGLE}}}jingle")
    if jingle_el is None:
        return None
    return jingle_el.get("action")


def build_session_accept(
    offer: JingleOffer,
    *,
    local_ufrag: str,
    local_pwd: str,
    local_candidates: list[IceCandidate],
    local_dtls_fingerprint: str,
    local_dtls_hash: str = "sha-256",
    local_dtls_setup: str = "active",
    accepted_codecs: list[tuple[str, str, str, str]] | None = None,
) -> ET.Element:
    """Build a Jingle session-accept IQ payload (the <jingle> element).

    The caller wraps this in an IQ stanza addressed to the initiator.
    """
    codecs = accepted_codecs or offer.codecs

    jingle = ET.Element(f"{{{NS_JINGLE}}}jingle")
    jingle.set("action", "session-accept")
    jingle.set("sid", offer.sid)
    jingle.set("responder", offer.responder)

    content = ET.SubElement(jingle, f"{{{NS_JINGLE}}}content")
    content.set("creator", "initiator")
    content.set("name", offer.content_name)
    content.set("senders", "both")

    # RTP description
    desc = ET.SubElement(content, f"{{{NS_JINGLE_RTP}}}description")
    desc.set("media", "audio")
    for pt_id, name, clockrate, channels in codecs:
        pt = ET.SubElement(desc, f"{{{NS_JINGLE_RTP}}}payload-type")
        pt.set("id", pt_id)
        pt.set("name", name)
        if clockrate:
            pt.set("clockrate", clockrate)
        if channels and channels != "1":
            pt.set("channels", channels)

    # ICE-UDP transport
    transport = ET.SubElement(content, f"{{{NS_JINGLE_ICE_UDP}}}transport")
    transport.set("ufrag", local_ufrag)
    transport.set("pwd", local_pwd)

    # DTLS fingerprint
    fp = ET.SubElement(transport, f"{{{NS_JINGLE_DTLS}}}fingerprint")
    fp.set("hash", local_dtls_hash)
    fp.set("setup", local_dtls_setup)
    fp.text = local_dtls_fingerprint

    # Local ICE candidates
    for c in local_candidates:
        cand = ET.SubElement(transport, f"{{{NS_JINGLE_ICE_UDP}}}candidate")
        cand.set("component", c.component)
        cand.set("foundation", c.foundation)
        cand.set("generation", c.generation)
        cand.set("id", c.id)
        cand.set("ip", c.ip)
        cand.set("network", c.network)
        cand.set("port", c.port)
        cand.set("priority", c.priority)
        cand.set("protocol", c.protocol)
        cand.set("type", c.type)
        if c.rel_addr:
            cand.set("rel-addr", c.rel_addr)
        if c.rel_port:
            cand.set("rel-port", c.rel_port)

    return jingle


def build_session_terminate(sid: str, reason: str = "success") -> ET.Element:
    """Build a Jingle session-terminate payload."""
    jingle = ET.Element(f"{{{NS_JINGLE}}}jingle")
    jingle.set("action", "session-terminate")
    jingle.set("sid", sid)

    reason_el = ET.SubElement(jingle, f"{{{NS_JINGLE}}}reason")
    ET.SubElement(reason_el, f"{{{NS_JINGLE}}}{reason}")

    return jingle


def build_iq_result(iq_stanza: Any) -> Any:
    """Build a bare IQ result reply (acknowledgement)."""
    reply = iq_stanza.reply()
    reply["type"] = "result"
    return reply
