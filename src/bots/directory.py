"""Switch directory service bot.

Implements the Switch hierarchy directory as a standard XMPP client account:

- Answers XEP-0030 disco#items queries for hierarchy nodes.
- Publishes change notifications to ejabberd mod_pubsub (XEP-0060).

This matches "option 1" in the spec: Switch owns the hierarchy model, ejabberd
owns pubsub fanout/storage.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, cast

from slixmpp import JID
from slixmpp.exceptions import IqError, IqTimeout
from slixmpp.plugins.xep_0030.stanza.items import DiscoItems
from slixmpp.xmlstream import ET

from src.db import SessionRepository
from src.utils import BaseXMPPBot


class DirectoryBot(BaseXMPPBot):
    """Directory service that serves hierarchy via disco + pubsub."""

    DISPATCHERS_NODE = "dispatchers"
    ACTIVE_SESSIONS_LIMIT = 200
    ACTIVE_SESSIONS_CACHE_TTL_S = 3.0

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        db,
        xmpp_domain: str,
        dispatchers_config: dict[str, dict[str, Any]],
        pubsub_service_jid: str,
    ):
        super().__init__(jid, password)
        self.log = logging.getLogger(f"directory.{jid}")
        self.sessions = SessionRepository(db)
        self.xmpp_domain = xmpp_domain
        self.dispatchers_config = dispatchers_config
        self.pubsub_service_jid = pubsub_service_jid

        # Session list browsing can generate bursts of disco#items queries.
        # Cache for a short TTL to avoid repeated DB scans.
        self._active_sessions_cache_ts: float = 0.0
        self._active_sessions_cache: list[Any] = []

        # Directory needs disco + pubsub.
        self.register_plugin("xep_0030")
        self.register_plugin("xep_0060")

        self.add_event_handler("session_start", self.on_start)
        self.add_event_handler("disconnected", self.on_disconnected)

    async def on_start(self, event):
        self.send_presence()

        # Dynamic disco handlers.
        disco = cast(Any, self["xep_0030"])
        disco.set_node_handler("get_items", None, None, self._get_items)  # pyright: ignore[reportAttributeAccessIssue]

        # Ensure at least the root node exists as a pubsub node.
        # (Creating nodes may fail depending on ejabberd policy; that is OK.)
        self._ensure_pubsub_node(self.DISPATCHERS_NODE)

        self.log.info(
            "Directory connected (pubsub=%s)",
            self.pubsub_service_jid,
        )
        self.set_connected(True)

    def on_disconnected(self, event):
        self.log.warning("Directory disconnected, reconnecting...")
        self.set_connected(False)
        asyncio.ensure_future(self._reconnect())

    async def _reconnect(self):
        await asyncio.sleep(5)
        self.connect()

    # ---------------------------------------------------------------------
    # XEP-0030: disco#items
    # ---------------------------------------------------------------------

    async def _get_items(self, jid, node, ifrom, data):
        """Return DiscoItems for a requested node."""
        requested = node or self.DISPATCHERS_NODE
        requester_bare = str(getattr(ifrom, "bare", ifrom) or "").split("/", 1)[0]

        if requested == self.DISPATCHERS_NODE:
            return self._items_dispatchers()

        # Direct dispatcher→sessions lookup (no groups indirection).
        if requested.startswith("sessions:"):
            dispatcher_jid = requested[len("sessions:"):]
            return self._items_sessions(dispatcher_jid, owner_jid=requester_bare)

        # Legacy: groups + individuals (kept for backward compat).
        if requested.startswith("groups:"):
            dispatcher_jid = requested[len("groups:") :]
            return self._items_groups(dispatcher_jid)

        if requested.startswith("individuals:"):
            group_jid = requested[len("individuals:") :]
            return self._items_individuals(group_jid, owner_jid=requester_bare)

        if requested.startswith("subagents:"):
            return DiscoItems()

        # Unknown node: return empty list (client treats as empty column).
        return DiscoItems()

    def _items_dispatchers(self) -> DiscoItems:
        items = DiscoItems()
        for idx, (key, cfg) in enumerate(self.dispatchers_config.items()):
            djid = cfg.get("jid")
            label = cfg.get("label") or key
            if not djid:
                continue
            # Encode position + direct flag in node so clients can restore
            # server-defined order (DiscoItems uses a set internally).
            if cfg.get("direct"):
                node = f"{idx}:direct"
            else:
                node = str(idx)
            items.add_item(JID(djid), node=node, name=label)
        return items

    def _is_direct_dispatcher(self, key: str | None) -> bool:
        if not key:
            return False
        cfg = self.dispatchers_config.get(key) or {}
        return bool(cfg.get("direct"))

    def _items_sessions(self, dispatcher_jid: str, owner_jid: str | None = None) -> DiscoItems:
        """Return sessions for a dispatcher directly (no groups indirection)."""
        items = DiscoItems()
        key = self._dispatcher_key_for_jid(dispatcher_jid)

        # Direct dispatchers (e.g. Acorn) have no sessions.
        if self._is_direct_dispatcher(key):
            return items

        now = time.time()
        if (
            self._active_sessions_cache
            and (now - self._active_sessions_cache_ts) < self.ACTIVE_SESSIONS_CACHE_TTL_S
            and not owner_jid
        ):
            sessions = self._active_sessions_cache
        else:
            if owner_jid:
                sessions = self.sessions.list_active_recent_for_owner(
                    owner_jid, limit=self.ACTIVE_SESSIONS_LIMIT
                )
            else:
                sessions = self.sessions.list_active_recent(limit=self.ACTIVE_SESSIONS_LIMIT)
                self._active_sessions_cache = sessions
                self._active_sessions_cache_ts = now

        sessions = self._filter_by_dispatcher(sessions, key)

        active_jids: set[str] = set()
        for s in sessions:
            try:
                items.add_item(JID(s.xmpp_jid), name=s.name)
                active_jids.add(str(JID(s.xmpp_jid).bare))
            except Exception:
                continue

        # Also include up to 10 recent closed sessions.
        if owner_jid:
            closed = self.sessions.list_recent_closed_for_owner(owner_jid, limit=10)
        else:
            closed = self.sessions.list_recent_closed(limit=10)
        closed = self._filter_by_dispatcher(closed, key)
        for s in closed:
            try:
                bare = str(JID(s.xmpp_jid).bare)
                if bare in active_jids:
                    continue
                items.add_item(JID(s.xmpp_jid), node="closed", name=s.name)
            except Exception:
                continue

        return items

    def _filter_by_dispatcher(self, sessions: list, key: str | None) -> list:
        """Filter sessions to those belonging to a dispatcher."""
        if not key:
            return sessions
        cfg = self.dispatchers_config.get(key) or {}
        cfg_jid = cfg.get("jid")
        filtered = []
        for s in sessions:
            if not s.dispatcher_jid:
                continue
            if str(JID(s.dispatcher_jid).bare) != str(JID(cfg_jid).bare):
                continue
            filtered.append(s)
        return filtered

    def _items_groups(self, dispatcher_jid: str) -> DiscoItems:
        items = DiscoItems()
        key = self._dispatcher_key_for_jid(dispatcher_jid)
        if not key:
            return items

        group_jid = self._sessions_group_jid_for_dispatcher(key)
        items.add_item(JID(group_jid), name="Sessions")
        return items

    def _items_individuals(self, group_jid: str, owner_jid: str | None = None) -> DiscoItems:
        items = DiscoItems()
        key = self._dispatcher_key_for_group_jid(group_jid)

        # Default: show active sessions.
        now = time.time()
        if (
            self._active_sessions_cache
            and (now - self._active_sessions_cache_ts) < self.ACTIVE_SESSIONS_CACHE_TTL_S
            and not owner_jid
        ):
            sessions = self._active_sessions_cache
        else:
            if owner_jid:
                sessions = self.sessions.list_active_recent_for_owner(
                    owner_jid, limit=self.ACTIVE_SESSIONS_LIMIT
                )
            else:
                sessions = self.sessions.list_active_recent(limit=self.ACTIVE_SESSIONS_LIMIT)
                self._active_sessions_cache = sessions
                self._active_sessions_cache_ts = now

        # If we can map the group to a dispatcher, filter to that dispatcher.
        if key:
            cfg = self.dispatchers_config.get(key) or {}
            dispatcher_jid = cfg.get("jid")
            filtered = []
            for s in sessions:
                if s.dispatcher_jid:
                    if str(JID(s.dispatcher_jid).bare) != str(JID(dispatcher_jid).bare):
                        continue
                # Sessions without dispatcher_jid shown under all dispatchers.
                filtered.append(s)
            sessions = filtered

        for s in sessions:
            try:
                items.add_item(JID(s.xmpp_jid), name=s.name)
            except Exception:
                # Be defensive: never fail a directory response due to one row.
                continue

        return items

    # ---------------------------------------------------------------------
    # XEP-0060: pubsub notifications
    # ---------------------------------------------------------------------

    def notify_sessions_changed(self, dispatcher_jid: str | None = None) -> None:
        """Publish a change notification for session lists."""
        if dispatcher_jid:
            key = self._dispatcher_key_for_jid(dispatcher_jid)
            if key:
                # New direct node.
                self._publish_sessions_payload(dispatcher_jid)
                # Legacy node (for older clients).
                group_jid = self._sessions_group_jid_for_dispatcher(key)
                self._publish_ping(f"individuals:{group_jid}")
                return

        # Fallback: global refresh.
        self._publish_ping(self.DISPATCHERS_NODE)

    def _publish_sessions_payload(self, dispatcher_jid: str) -> None:
        """Publish a fat notification with the full session list inline."""
        node = f"sessions:{dispatcher_jid}"
        try:
            self._ensure_pubsub_node(node)
            items_disco = self._items_sessions(dispatcher_jid)
            payload = ET.Element("switch")
            payload.set("event", "sessions")
            payload.set("ts", str(int(time.time())))
            payload.set("dispatcher", dispatcher_jid)
            for item_tuple in items_disco["items"]:
                # items are (jid, node, name) tuples
                jid_val, node_val, name_val = item_tuple
                session_el = ET.SubElement(payload, "session")
                session_el.set("jid", str(jid_val))
                if node_val == "closed":
                    session_el.set("status", "closed")
                if name_val:
                    session_el.set("name", name_val)
            pubsub = cast(Any, self["xep_0060"])
            pubsub.publish(  # pyright: ignore[reportAttributeAccessIssue]
                self.pubsub_service_jid,
                node,
                id=str(uuid.uuid4()),
                payload=payload,
            )
        except (IqTimeout, IqError) as exc:
            self.log.warning("PubSub publish failed for %s: %s", node, exc)
        except Exception as exc:
            self.log.debug("Failed to publish sessions payload for %s: %s", node, exc)

    def _publish_ping(self, node: str) -> None:
        try:
            self._ensure_pubsub_node(node)
            payload = ET.Element("switch")
            payload.set("event", "update")
            payload.set("ts", str(int(time.time())))
            pubsub = cast(Any, self["xep_0060"])
            pubsub.publish(  # pyright: ignore[reportAttributeAccessIssue]
                self.pubsub_service_jid,
                node,
                id=str(uuid.uuid4()),
                payload=payload,
            )
        except IqTimeout:
            self.log.warning(
                "PubSub publish timed out (node=%s service=%s)",
                node,
                self.pubsub_service_jid,
            )
        except IqError as exc:
            try:
                condition = exc.iq["error"]["condition"]
                text = exc.iq["error"]["text"]
            except Exception:
                condition = "unknown"
                text = ""
            self.log.warning(
                "PubSub publish failed (node=%s service=%s condition=%s %s)",
                node,
                self.pubsub_service_jid,
                condition,
                text,
            )
        except Exception as exc:
            self.log.debug("Failed to publish pubsub update for %s: %s", node, exc)

    def _ensure_pubsub_node(self, node: str) -> None:
        pubsub = cast(Any, self["xep_0060"])
        try:
            result = pubsub.create_node(  # pyright: ignore[reportAttributeAccessIssue]
                self.pubsub_service_jid,
                node,
            )
        except IqTimeout:
            self.log.warning(
                "PubSub create_node timed out (node=%s service=%s)",
                node,
                self.pubsub_service_jid,
            )
            return
        except IqError as exc:
            # Some slixmpp versions raise synchronously.
            try:
                condition = exc.iq["error"]["condition"]
                text = exc.iq["error"]["text"]
            except Exception:
                condition = "unknown"
                text = ""
            if condition != "conflict":
                self.log.warning(
                    "PubSub create_node failed (node=%s service=%s condition=%s %s)",
                    node,
                    self.pubsub_service_jid,
                    condition,
                    text,
                )
            return
        except Exception:
            return

        # Other slixmpp versions return an awaitable/Future; ensure we always
        # consume exceptions so we don't get "Future exception was never retrieved".
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            task = asyncio.ensure_future(result)

            def _done(t: asyncio.Future):
                try:
                    t.result()
                except IqError as exc:
                    try:
                        condition = exc.iq["error"]["condition"]
                        text = exc.iq["error"]["text"]
                    except Exception:
                        condition = "unknown"
                        text = ""
                    if condition != "conflict":
                        self.log.warning(
                            "PubSub create_node failed (node=%s service=%s condition=%s %s)",
                            node,
                            self.pubsub_service_jid,
                            condition,
                            text,
                        )
                except IqTimeout:
                    self.log.warning(
                        "PubSub create_node timed out (node=%s service=%s)",
                        node,
                        self.pubsub_service_jid,
                    )
                except Exception:
                    # Best-effort.
                    return

            task.add_done_callback(_done)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _dispatcher_key_for_jid(self, dispatcher_jid: str) -> str | None:
        bare = str(JID(dispatcher_jid).bare)
        for key, cfg in self.dispatchers_config.items():
            if str(JID(cfg.get("jid", "")).bare) == bare:
                return key
        return None

    def _sessions_group_jid_for_dispatcher(self, dispatcher_key: str) -> str:
        return f"sessions-{dispatcher_key}@{self.xmpp_domain}"

    def _dispatcher_key_for_group_jid(self, group_jid: str) -> str | None:
        try:
            local = str(JID(group_jid).user)
        except Exception:
            return None
        if not local.startswith("sessions-"):
            return None
        key = local[len("sessions-") :]
        return key if key in self.dispatchers_config else None
