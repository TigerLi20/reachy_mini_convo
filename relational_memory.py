"""Session identity binding and scoped memory operations for the robot runtime."""

from dataclasses import dataclass
import json
import logging
import re
import threading
import uuid
from time import monotonic

from relational_store import normalized

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryScope:
    person_id: str
    session_id: str
    epoch: int


class RelationalMemory:
    """Never accept a person ID from an LLM or infer the speaker in a group."""

    def __init__(self, store, recognition, models, worker) -> None:
        self.store, self.recognition, self.models, self.worker = store, recognition, models, worker
        self.lock = threading.RLock()
        self.epoch = 0
        self.runtime_id = str(uuid.uuid4())
        self.backend_epoch = -1
        self.person_id = None
        self.session_id = None
        self.name = None
        self.pending = None
        self.pending_delete = None
        self.capture = None
        self.last_user_turn = None
        self.uncertain_at = None
        self.previous_mode = "unknown"
        self.previous_verified = None
        self.notice = "startup"
        self.generation_turns = 0
        self.compaction_requested = False
        self.reply_completed = False
        self.closing = False

    def scene(self) -> dict:
        if self.recognition is None:
            return {"state": "unknown", "reason": "recognition_unavailable", "face_count": 0}
        return self.recognition.snapshot()

    def _reset(self, reason: str, unbind=False) -> None:
        if unbind and self.person_id is not None:
            self.store.end(self.person_id, self.session_id)
            self.worker.submit(self.person_id, self.session_id, final=True)
            self.person_id = self.session_id = self.name = None
        self.epoch += 1
        self.notice = reason
        self.capture = self.pending = self.last_user_turn = None
        self.pending_delete = None
        self.generation_turns = 0
        self.compaction_requested = False
        self.reply_completed = False
        logger.info("MEMORY context_reset reason=%s epoch=%d", reason, self.epoch)

    def tick(self) -> dict:
        scene = self.scene()
        now = monotonic()
        with self.lock:
            if self.closing:
                return scene
            group = scene.get("face_count", 0) > 1 or scene.get("reason") == "multiple_faces"
            known = scene.get("state") == "known" and scene.get("face_count") == 1
            mode = "group" if group else "single" if known else "unknown"
            if self.person_id is not None:
                changed = known and scene["person_id"] != self.person_id
                if group or changed:
                    self._reset("multiple_people" if group else "person_changed", unbind=True)
                elif not known:
                    self.uncertain_at = now if self.uncertain_at is None else self.uncertain_at
                    if now - self.uncertain_at > 1.5:
                        self._reset("identity_lost", unbind=True)
                else:
                    self.uncertain_at = None
            elif ((mode == "group") != (self.previous_mode == "group") or
                  (known and self.previous_verified is not None and self.previous_verified != scene["person_id"])):
                self._reset("multiple_people" if group else "confirm_new_person")
            if known:
                self.previous_verified = scene["person_id"]
            if group:
                self.previous_verified = None
                self.pending = None
            self.previous_mode = mode
            return scene

    def scope(self) -> MemoryScope | None:
        scene = self.tick()
        with self.lock:
            if (not self.closing and self.person_id and self.backend_epoch == self.epoch and scene.get("face_count") == 1
                    and scene.get("state") == "known" and scene.get("person_id") == self.person_id):
                return MemoryScope(self.person_id, self.session_id, self.epoch)
            return None

    def valid(self, scope: MemoryScope | None) -> bool:
        return scope is not None and scope == self.scope()

    def output_allowed(self, epoch: int | None = None) -> bool:
        self.tick()
        with self.lock:
            return not self.closing and self.backend_epoch == self.epoch and (epoch is None or epoch == self.epoch)

    def attach_backend(self) -> int:
        self.tick()
        with self.lock:
            self.backend_epoch = self.epoch
            self.generation_turns = 0
            return self.epoch

    def session(self, confirmed: bool, transcript_seq: int, transcript: str) -> dict:
        scene = self.tick()
        with self.lock:
            if self.scope() is not None:
                return {"status": "bound", "name": self.name}
            if scene.get("face_count") != 1 or scene.get("state") != "known":
                return {"status": "temporary_only", "instruction": "Personal memory is unavailable until exactly one enrolled person is recognized. Chat generically; never reveal or save private memories."}
            pending = self.pending
            if (not confirmed or pending is None or pending["person_id"] != scene["person_id"]
                    or monotonic() - pending["at"] > 60):
                self.pending = {"person_id": scene["person_id"], "seq": transcript_seq, "at": monotonic()}
                return {"status": "confirmation_required", "name": scene["name"],
                        "instruction": f'Ask whether to continue as {scene["name"]} and use their personal memories. Wait for a new explicit yes.'}
            yes = re.match(r"^(yes|yeah|yep|correct|that['’]?s right|please do)\b", transcript.strip(), re.I)
            no = re.search(r"\b(no|not|wrong|instead|don['’]?t)\b", transcript, re.I)
            if transcript_seq <= pending["seq"] or not yes or no:
                return {"status": "confirmation_required", "instruction": "No new explicit yes was detected; no private history has been loaded."}
            self.person_id, self.name = scene["person_id"], scene["name"]
            self.session_id = self.store.start(self.person_id)
            self.uncertain_at = None
            self._reset("memory_bound")
            return {"status": "bound_restarting", "instruction": "A fresh private context will start automatically."}

    def speech_started(self, epoch: int) -> None:
        with self.lock:
            self.capture = self.scope() if epoch == self.epoch else None
            self.reply_completed = False

    def transcript(self, epoch: int, role: str, text: str) -> None:
        with self.lock:
            scope = self.capture if role == "user" else self.scope()
            if epoch != self.epoch or not self.valid(scope):
                return
            turn = self.store.append(scope.person_id, scope.session_id, role, text)
            if role == "user":
                self.last_user_turn = turn
                self.generation_turns += 1
            elif self.last_user_turn is not None:
                self.reply_completed = True
            if self.generation_turns >= 8:
                self.worker.submit(scope.person_id, scope.session_id)
                self.compaction_requested = True

    def compact_if_ready(self) -> None:
        with self.lock:
            scope = self.scope()
            if scope and self.reply_completed and self.compaction_requested and not self.worker.busy(scope.person_id, scope.session_id):
                self._reset("context_compacted")

    def context(self) -> dict:
        with self.lock:
            scope = self.scope()
            if scope is None:
                return {"mode": "group" if self.previous_mode == "group" else "temporary",
                        "instruction": "No personal history is available. Confirm one recognized person's memory session before any private recall or saving."}
            context = self.store.context(scope.person_id, scope.session_id)
            context["name"] = self.name
            context["mode"] = "private"
            return context

    def operation(self, action: str, *, key="", value="", quote="", query="", confirmed=False,
                  expected_scope=None, source_turn=...) -> dict:
        scope = self.scope()
        if scope is None or expected_scope is not None and scope != expected_scope:
            return {"status": "blocked", "instruction": "Memory is paused. Use a fresh single-person identity check and memory-session confirmation; never guess whose memories to use."}
        with self.lock:
            source_turn = self.last_user_turn if source_turn is ... else source_turn
        if action in ("recall", "remember", "correct"):
            vector = self.models.embed(query if action == "recall" else f"{key}: {value}")
        else:
            vector = None
        # Scope is checked again after model work, which may run in a worker thread.
        with self.lock:
            if not self.valid(scope):
                return {"status": "blocked", "instruction": "The person/session changed while processing. Nothing was read or saved."}
            try:
                if action == "review":
                    result = {"status": "ok", "facts": self.store.review(scope.person_id),
                              "current_user_turn": self.store.turns(scope.person_id, scope.session_id, 2)}
                elif action == "recall":
                    result = {"status": "ok", "memories": self.store.recall(scope.person_id, query, vector)}
                elif action in ("remember", "correct"):
                    result = self.store.remember(scope.person_id, scope.session_id, source_turn,
                                                 key, value, quote, vector, correction=action == "correct")
                    if result["status"] == "corrected":
                        self._rebuild(scope.person_id)
                        self._reset("memory_corrected")
                elif action == "forget":
                    result = self.store.forget(scope.person_id, key)
                    if result["status"] == "forgotten":
                        self._rebuild(scope.person_id)
                        self._reset("memory_deleted")
                elif action == "forget_episode":
                    details = self.store.episode_details(scope.person_id, query)
                    if details is None:
                        result = {"status": "not_found", "instruction": "Recall the episode first; no episode was deleted."}
                    elif (not confirmed or self.pending_delete is None or self.pending_delete[0] != query
                          or monotonic() - self.pending_delete[2] > 60):
                        self.pending_delete = (query, self.last_user_turn, monotonic())
                        result = {"status": "confirmation_required", "source_messages": details["source_messages"],
                                  "derived_facts": details["derived_facts"],
                                  "instruction": "Explain that this episode deletion also removes these source messages and derived facts. Ask for a new explicit yes before confirming."}
                    else:
                        rows = self.store.turns(scope.person_id, scope.session_id, 4)
                        latest = next((r for r in reversed(rows) if r["role"] == "user"), None)
                        affirmative = latest and latest["id"] != self.pending_delete[1] and re.match(
                            r"^(yes|yeah|yep|correct|please do)\b", latest["text"].strip(), re.I)
                        negative = latest and re.search(r"\b(no|not|wrong|don['’]?t)\b", latest["text"], re.I)
                        if not affirmative or negative:
                            result = {"status": "confirmation_required", "instruction": "No new explicit yes detected; nothing was deleted."}
                        else:
                            result = self.store.forget_episode(scope.person_id, query)
                            self._rebuild(scope.person_id)
                            self._reset("memory_deleted")
                elif action == "forget_all":
                    # Require an actual user request rather than model initiative.
                    rows = self.store.turns(scope.person_id, scope.session_id, 4)
                    latest = next((r["text"] for r in reversed(rows) if r["role"] == "user"), "")
                    request = normalized(latest)
                    if not re.match(r"^(please |can you |could you |i want you to )?(forget|delete|erase|clear)\b.*\b(all|everything)\b", request):
                        raise ValueError("Ask the user to explicitly request forgetting all conversation memories")
                    result = self.store.forget_all(scope.person_id)
                    self.session_id = self.store.start(scope.person_id)
                    self._reset("all_memories_deleted")
                else:
                    raise ValueError("Unknown memory action")
            except ValueError as exc:
                result = {"status": "not_saved", "reason": str(exc),
                          "instruction": "Clarify with the user or review the current transcript; never claim an unsuccessful change worked."}
            result["_memory_epoch"] = self.epoch
            logger.info("MEMORY operation=%s status=%s", action, result["status"])
            return result

    def _rebuild(self, person_id: str) -> None:
        for owner, session in self.store.recoverable_sessions():
            if owner == person_id:
                self.worker.submit(owner, session, final=True)

    def finish(self) -> None:
        with self.lock:
            if self.person_id:
                self.store.end(self.person_id, self.session_id)
                self.worker.submit(self.person_id, self.session_id, final=True)
