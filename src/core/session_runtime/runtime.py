"""SessionRuntime.

This is the single place that owns:
- serialization (message queue)
- cancellation (drop queued + cancel in-flight)
- runner orchestration and question handling

It intentionally depends only on ports, not concrete XMPP/DB/runners.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from src.attachments import Attachment
from src.runners import Question, Runner
from src.runners.opencode.config import OpenCodeConfig

from src.core.session_runtime.ports import (
    AttachmentPromptPort,
    HistoryPort,
    MessageStorePort,
    ReplyPort,
    RunnerFactoryPort,
    SessionState,
    SessionStorePort,
    TypingPort,
)


@dataclass(frozen=True)
class _WorkItem:
    generation: int
    body: str
    attachments: list[Attachment] | None
    trigger_response: bool
    scheduled: bool
    done: asyncio.Future[None] | None = None
    enqueued_at: float = field(default_factory=time.monotonic)


class SessionRuntime:
    def __init__(
        self,
        *,
        session_name: str,
        working_dir: str,
        output_dir: Path,
        sessions: SessionStorePort,
        messages: MessageStorePort,
        reply: ReplyPort,
        typing: TypingPort,
        runner_factory: RunnerFactoryPort,
        history: HistoryPort,
        prompt: AttachmentPromptPort,
        infer_meta_tool_from_summary: Callable[[str], str | None],
        on_processing_changed: Callable[[bool], None] | None = None,
    ):
        self.session_name = session_name
        self.working_dir = working_dir
        self.output_dir = output_dir
        self._sessions = sessions
        self._messages = messages
        self._reply = reply
        self._typing = typing
        self._runner_factory = runner_factory
        self._history = history
        self._prompt = prompt
        self._infer_meta_tool_from_summary = infer_meta_tool_from_summary
        self._on_processing_changed = on_processing_changed

        self._generation = 0
        self._queue: asyncio.Queue[_WorkItem] = asyncio.Queue()
        self._task: asyncio.Task | None = None

        self.processing = False
        self.shutting_down = False

        self.runner: Runner | None = None
        self._run_task: asyncio.Task | None = None

        self._pending_question_answers: dict[str, asyncio.Future] = {}

    def ensure_running(self) -> None:
        if self.shutting_down:
            return
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop())

    def pending_count(self) -> int:
        return self._queue.qsize()

    async def enqueue(
        self,
        body: str,
        attachments: list[Attachment] | None,
        *,
        trigger_response: bool,
        scheduled: bool,
        wait: bool,
    ) -> None:
        if self.shutting_down:
            return

        done: asyncio.Future[None] | None = None
        if wait:
            done = asyncio.get_running_loop().create_future()

        item = _WorkItem(
            generation=self._generation,
            body=body,
            attachments=list(attachments) if attachments else None,
            trigger_response=trigger_response,
            scheduled=scheduled,
            done=done,
        )
        await self._queue.put(item)
        self.ensure_running()

        if done is not None:
            await done

    def cancel_queued(self) -> bool:
        """Drop queued items; ignore anything already dequeued."""
        self._generation += 1
        dropped_any = False
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                dropped_any = True
                if item.done and not item.done.done():
                    item.done.set_exception(asyncio.CancelledError())
        return dropped_any

    def cancel_operations(self, *, notify: bool = False) -> bool:
        cancelled_any = False

        if self.cancel_queued():
            cancelled_any = True

        if self.runner and self.processing:
            cancelled_any = True
            self.runner.cancel()

        if self._run_task and not self._run_task.done():
            cancelled_any = True
            self._run_task.cancel()

        # Best-effort: unblock any waiting question futures.
        for fut in list(self._pending_question_answers.values()):
            if fut and not fut.done():
                fut.cancel()

        if notify and cancelled_any:
            self._reply.send_reply("Cancelling current work...")

        return cancelled_any

    def shutdown(self) -> None:
        if self.shutting_down:
            return
        self.shutting_down = True
        self.cancel_operations(notify=False)
        task = self._task
        self._task = None
        if task and not task.done():
            task.cancel()
        self.cancel_queued()

    def answer_question(self, answer: object, *, request_id: str | None = None) -> bool:
        if not self._pending_question_answers:
            return False
        rid = request_id or list(self._pending_question_answers.keys())[-1]
        fut = self._pending_question_answers.get(rid)
        if fut and not fut.done():
            fut.set_result(answer)
            return True
        return False

    def _set_processing(self, active: bool) -> None:
        self.processing = active
        if self._on_processing_changed:
            try:
                self._on_processing_changed(active)
            except Exception:
                pass
        if active:
            self._typing.start()
        else:
            self._typing.stop()

    async def _loop(self) -> None:
        try:
            while not self.shutting_down:
                item = await self._queue.get()
                if item.generation != self._generation:
                    if item.done and not item.done.done():
                        item.done.set_exception(asyncio.CancelledError())
                    continue

                if item.trigger_response:
                    self._set_processing(True)

                try:
                    await self._process_one(item)
                except asyncio.CancelledError:
                    if self.shutting_down:
                        raise
                finally:
                    if item.trigger_response:
                        self._set_processing(False)
                    if item.done and not item.done.done():
                        item.done.set_result(None)
        except asyncio.CancelledError:
            return

    async def _process_one(self, item: _WorkItem) -> None:
        session = self._sessions.get(self.session_name)
        if not session:
            self._reply.send_reply("Session not found in database.")
            return

        self._sessions.update_last_active(self.session_name)

        body_for_history = self._prompt.augment_prompt(item.body, item.attachments)
        self._history.append_to_history(body_for_history, self.working_dir, session.claude_session_id)
        self._history.log_activity(item.body, session=self.session_name, source="xmpp")
        self._messages.add(self.session_name, "user", body_for_history, session.active_engine)

        if not item.trigger_response:
            return

        engine = (session.active_engine or "opencode").strip().lower()
        self._run_task = asyncio.create_task(
            self._run_engine(engine=engine, session=session, prompt=body_for_history)
        )
        try:
            await self._run_task
        finally:
            self._run_task = None

    async def _run_engine(self, *, engine: str, session: SessionState, prompt: str) -> None:
        if engine == "claude":
            await self._run_claude(session, prompt)
            return
        if engine == "opencode":
            await self._run_opencode(session, prompt)
            return
        self._reply.send_reply(f"Unknown engine '{engine}'.")

    async def _run_claude(self, session: SessionState, prompt: str) -> None:
        self.runner = self._runner_factory.create(
            "claude",
            working_dir=self.working_dir,
            output_dir=self.output_dir,
            session_name=self.session_name,
        )

        response_parts: list[str] = []
        tool_summaries: list[str] = []
        last_progress_at = 0

        async for event_type, content in self.runner.run(prompt, session.claude_session_id):
            if self.shutting_down:
                return

            if event_type == "session_id" and isinstance(content, str) and content:
                self._sessions.update_claude_session_id(self.session_name, content)
            elif event_type == "text" and isinstance(content, str):
                response_parts = [content]
            elif event_type == "tool" and isinstance(content, str):
                tool_summaries.append(content)
                if len(tool_summaries) - last_progress_at >= 8:
                    last_progress_at = len(tool_summaries)
                    self._reply.send_reply(
                        f"... {' '.join(tool_summaries[-3:])}",
                        meta_type="tool",
                        meta_tool=self._infer_meta_tool_from_summary(content),
                    )
                    self._typing.maybe_send(min_interval_s=5.0)
            elif event_type == "result":
                self._send_result(tool_summaries, response_parts, content, engine="claude")
            elif event_type == "error":
                self._typing.stop()
                self._reply.send_reply(f"Error: {content}")
            elif event_type == "cancelled":
                self._typing.stop()
                self._reply.send_reply("Cancelled.")

    async def _run_opencode(self, session: SessionState, prompt: str) -> None:
        question_callback = self._create_question_callback()
        self.runner = self._runner_factory.create(
            "opencode",
            working_dir=self.working_dir,
            output_dir=self.output_dir,
            session_name=self.session_name,
            opencode_config=OpenCodeConfig(
                model=session.model_id,
                reasoning_mode=session.reasoning_mode,
                agent=session.opencode_agent,
                question_callback=question_callback,
            ),
        )

        response_parts: list[str] = []
        tool_summaries: list[str] = []
        accumulated = ""
        last_progress_at = 0

        async for event_type, content in self.runner.run(prompt, session.opencode_session_id):
            if self.shutting_down:
                return

            if event_type == "session_id" and isinstance(content, str) and content:
                self._sessions.update_opencode_session_id(self.session_name, content)
            elif event_type == "text" and isinstance(content, str):
                accumulated += content
                response_parts = [accumulated]
            elif event_type == "tool" and isinstance(content, str):
                tool_summaries.append(content)
                is_bash = content.startswith("[tool:bash")
                if is_bash or len(tool_summaries) == 1:
                    last_progress_at = len(tool_summaries)
                    self._reply.send_reply(
                        f"... {content}",
                        meta_type="tool",
                        meta_tool=self._infer_meta_tool_from_summary(content),
                    )
                    self._typing.maybe_send(min_interval_s=5.0)
                elif len(tool_summaries) - last_progress_at >= 8:
                    last_progress_at = len(tool_summaries)
                    self._reply.send_reply(
                        f"... {' '.join(tool_summaries[-3:])}",
                        meta_type="tool",
                        meta_tool=self._infer_meta_tool_from_summary(tool_summaries[-1]),
                    )
                    self._typing.maybe_send(min_interval_s=5.0)
            elif event_type == "question" and isinstance(content, Question):
                # The runner handles question flow via the callback.
                pass
            elif event_type == "result":
                self._send_result(tool_summaries, response_parts, content, engine="opencode")
            elif event_type == "error":
                self._typing.stop()
                self._reply.send_reply(f"Error: {content}")
            elif event_type == "cancelled":
                self._typing.stop()
                self._reply.send_reply("Cancelled.")

    def _send_result(
        self,
        tool_summaries: list[str],
        response_parts: list[str],
        stats: object,
        *,
        engine: str,
    ) -> None:
        self._typing.stop()

        parts: list[str] = []
        if tool_summaries:
            tools = " ".join(tool_summaries[:5])
            if len(tool_summaries) > 5:
                tools += f" +{len(tool_summaries) - 5}"
            parts.append(tools)
        if response_parts:
            parts.append(response_parts[-1])

        meta_type = None
        meta_attrs: dict[str, str] | None = None
        if isinstance(stats, dict):
            meta_type = "run-stats"
            meta_attrs = {str(k): str(v) for k, v in stats.items() if v is not None}

        self._reply.send_reply("\n\n".join([p for p in parts if p]), meta_type=meta_type, meta_attrs=meta_attrs)
        self._messages.add(
            self.session_name,
            "assistant",
            response_parts[-1] if response_parts else "",
            engine,
        )

    def _create_question_callback(self) -> Callable[[Question], Awaitable[list[list[str]]]]:
        async def question_callback(question: Question) -> list[list[str]]:
            question_text = self._format_question(question)
            self._reply.send_reply(
                question_text,
                meta_type="question",
                meta_tool="question",
                meta_attrs={
                    "version": "1",
                    "engine": "opencode",
                    "request_id": question.request_id,
                    "question_count": str(len(question.questions or [])),
                },
                meta_payload={
                    "version": 1,
                    "engine": "opencode",
                    "request_id": question.request_id,
                    "questions": question.questions,
                },
            )

            fut: asyncio.Future = asyncio.get_event_loop().create_future()
            self._pending_question_answers[question.request_id] = fut
            try:
                answer = await asyncio.wait_for(fut, timeout=300)
                return self._parse_question_answer(question, answer)
            except asyncio.TimeoutError:
                self._reply.send_reply("[Question timed out - proceeding without answer]")
                raise
            finally:
                self._pending_question_answers.pop(question.request_id, None)

        return question_callback

    def _parse_question_answer(self, question: Question, answer: object) -> list[list[str]]:
        if isinstance(answer, list):
            return answer  # type: ignore[return-value]

        text = str(answer or "").strip()
        qs = question.questions or []
        if not qs:
            return []

        segments: list[str] = []
        if "\n" in text:
            segments = [s.strip() for s in text.splitlines() if s.strip()]
        if not segments and ";" in text and len(qs) > 1:
            segments = [s.strip() for s in text.split(";") if s.strip()]
        if not segments:
            segments = [text]

        answers: list[list[str]] = []
        for idx, q in enumerate(qs):
            seg = segments[idx] if idx < len(segments) else (segments[0] if segments else "")
            options = q.get("options") if isinstance(q, dict) else None
            if not isinstance(options, list) or not options:
                answers.append([seg] if seg else [])
                continue

            labels: list[str] = []
            for opt in options:
                if isinstance(opt, dict):
                    lab = str(opt.get("label", "") or "").strip()
                    if lab:
                        labels.append(lab)

            chosen: list[str] = []
            seg_norm = seg.strip().lower()
            direct = next((lab for lab in labels if lab.lower() == seg_norm), None)
            if direct:
                answers.append([direct])
                continue

            for tok in re.split(r"[\s,]+", seg.strip()):
                if not tok:
                    continue
                if tok.isdigit():
                    n = int(tok)
                    if 1 <= n <= len(labels):
                        chosen.append(labels[n - 1])
                    continue
                match = next((lab for lab in labels if lab.lower() == tok.lower()), None)
                if match:
                    chosen.append(match)

            seen: set[str] = set()
            chosen = [x for x in chosen if not (x in seen or seen.add(x))]
            answers.append(chosen)

        return answers

    def _format_question(self, question: Question) -> str:
        parts: list[str] = []
        parts.append("[Question]")
        for q_idx, q in enumerate(question.questions or [], 1):
            if not isinstance(q, dict):
                continue
            header = str(q.get("header", "") or "").strip()
            text = str(q.get("question", "") or "").strip()
            options = q.get("options", [])

            if header:
                parts.append(f"{q_idx}) {header}")
            elif len(question.questions or []) > 1:
                parts.append(f"{q_idx})")
            if text:
                parts.append(text)

            if isinstance(options, list) and options:
                parts.append("Options:")
                for i, opt in enumerate(options, 1):
                    if not isinstance(opt, dict):
                        continue
                    label = str(opt.get("label", f"Option {i}") or f"Option {i}").strip()
                    desc = str(opt.get("description", "") or "").strip()
                    parts.append(f"  {i}) {label}" + (f" - {desc}" if desc else ""))

        parts.append("Reply with option number(s) (e.g., '1' or '1,2') or label text.")
        return "\n".join([p for p in parts if p])
