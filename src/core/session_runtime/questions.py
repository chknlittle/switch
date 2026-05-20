"""Question handling for interactive runner prompts."""

from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable, cast

from src.core.session_runtime.api import OutboundMessage
from src.runners import Question


class QuestionHandlerMixin:
    def _create_question_callback(
        self,
        engine: str = "pi",
    ) -> Callable[[Question], Awaitable[list[list[str]]]]:
        async def question_callback(question: Question) -> list[list[str]]:
            question_text = self._format_question(question)
            await self._emit(
                OutboundMessage(
                    question_text,
                    meta_type="question",
                    meta_tool="question",
                    meta_attrs={
                        "version": "1",
                        "engine": engine,
                        "request_id": question.request_id,
                        "question_count": str(len(question.questions or [])),
                    },
                    meta_payload={
                        "version": 1,
                        "engine": engine,
                        "request_id": question.request_id,
                        "questions": question.questions,
                    },
                )
            )

            fut: asyncio.Future = asyncio.get_event_loop().create_future()
            self._pending_question_answers[question.request_id] = fut
            try:
                answer = await asyncio.wait_for(fut, timeout=300)
                return self._parse_question_answer(question, answer)
            except asyncio.TimeoutError:
                await self._emit(
                    OutboundMessage("[Question timed out - proceeding without answer]")
                )
                raise
            finally:
                self._pending_question_answers.pop(question.request_id, None)

        return question_callback

    def _parse_question_answer(
        self, question: Question, answer: object
    ) -> list[list[str]]:
        if (
            isinstance(answer, list)
            and all(isinstance(item, list) for item in answer)
            and all(
                isinstance(choice, str)
                for item in answer
                for choice in (item if isinstance(item, list) else [])
            )
        ):
            return cast(list[list[str]], answer)

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
            seg = (
                segments[idx]
                if idx < len(segments)
                else (segments[0] if segments else "")
            )
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
                match = next(
                    (lab for lab in labels if lab.lower() == tok.lower()), None
                )
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
                    label = str(
                        opt.get("label", f"Option {i}") or f"Option {i}"
                    ).strip()
                    desc = str(opt.get("description", "") or "").strip()
                    parts.append(f"  {i}) {label}" + (f" - {desc}" if desc else ""))

        parts.append("Reply with option number(s) (e.g., '1' or '1,2') or label text.")
        return "\n".join([p for p in parts if p])
