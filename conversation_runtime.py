"""Timing and farewell playback coordination for phase 2."""

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
from time import monotonic


class ConversationTiming(logging.Handler):
    """Observe the pinned app's realtime event logs without changing its backend."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.turn = 0
        self.speech_end = None
        self.first_audio = False
        self.response_active = False
        self.user_speaking = False
        self.awaiting_response = False
        self.activity_at = monotonic()
        self.user_transcript = ""
        self.transcript_seq = 0
        self.response_number = 0
        self.last_audio_at = None
        self.playback_until = 0.0
        self.last_push_at = 0.0
        self.ending = False
        self.farewell_task = None
        self.state_lock = threading.RLock()

    def write_event(self, event: str, **fields) -> None:
        now = monotonic()
        entry = {"event": event, "turn": self.turn,
                 "time_utc": datetime.now(timezone.utc).isoformat(),
                 "monotonic_s": now, **fields}
        if self.speech_end is not None:
            entry["since_speech_end_ms"] = round((now - self.speech_end) * 1000, 1)
        with self.path.open("a") as log:
            log.write(json.dumps(entry) + "\n")
        print("TIMING " + json.dumps(entry), flush=True)

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("User transcript: "):
            with self.state_lock:
                self.user_transcript = message[len("User transcript: "):]
                self.transcript_seq += 1
            return
        prefix = "Realtime event: "
        if not message.startswith(prefix):
            return
        event = message[len(prefix):]
        with self.state_lock:
            if event == "input_audio_buffer.speech_started":
                self.user_speaking = True
                self.awaiting_response = False
                self.activity_at = monotonic()
            elif event == "input_audio_buffer.speech_stopped":
                self.user_speaking = False
                self.awaiting_response = True
                self.activity_at = monotonic()
                self.turn += 1
                self.speech_end = monotonic()
                self.first_audio = False
                self.write_event("speech_ended", measurement="backend_vad_event_received")
            elif event == "response.created":
                self.activity_at = monotonic()
                self.response_number += 1
                self.response_active = True
                self.last_audio_at = None
                self.first_audio = False
                self.write_event("processing_started", measurement="backend_response_created_received")
            elif event == "response.output_audio.delta":
                self.activity_at = monotonic()
                self.last_audio_at = monotonic()
            elif event == "response.done":
                self.response_active = False
                self.awaiting_response = False

    def request_farewell(self, sleep_callback) -> dict:
        """Arrange sleep after the tool's next spoken response finishes."""
        if sleep_callback is None:
            return {"error": "Sleep is unavailable in this runtime."}
        if self.farewell_task is not None and not self.farewell_task.done():
            return {"status": "already_requested", "instruction": "Finish your brief farewell; do not ask a question."}
        self.farewell_task = asyncio.create_task(
            self.finish(sleep_callback, after_response=self.response_number),
            name="farewell-then-sleep",
        )
        return {"status": "farewell_requested",
                "instruction": "Now say a short warm farewell aloud. Do not call another tool or ask a question. "
                               "Reachy will go to sleep after your audio finishes."}

    async def cancel_farewell(self) -> None:
        if self.farewell_task is not None and not self.farewell_task.done():
            self.farewell_task.cancel()
            try:
                await self.farewell_task
            except asyncio.CancelledError:
                pass
        self.farewell_task = None

    async def finish(self, sleep_callback, after_response: int | None = None) -> dict:
        """Wait for response delivery and queued playback before sleeping."""
        if sleep_callback is None:
            return {"error": "Sleep is unavailable in this runtime."}
        with self.state_lock:
            if self.ending:
                return {"status": "already_requested"}
            self.ending = True
        try:
            deadline = monotonic() + 45
            while monotonic() < deadline:
                with self.state_lock:
                    now = monotonic()
                    final_response = after_response is None or self.response_number > after_response
                    ready = (final_response and not self.response_active and self.last_audio_at is not None
                             and self.last_push_at >= self.last_audio_at
                             and now >= self.playback_until + 0.5
                             and now - self.last_push_at >= 1.0)
                    no_farewell = final_response and not self.response_active and self.last_audio_at is None
                if no_farewell:
                    self.write_event("farewell_failed", reason="No farewell audio received; staying awake")
                    return {"error": "No farewell audio received; robot remains awake."}
                if ready:
                    self.write_event("farewell_playback_finished", measurement="estimated_queue_drain")
                    self.write_event("sleep_requested")
                    result = await asyncio.to_thread(sleep_callback)
                    self.write_event("sleep_result", result=result)
                    return result
                await asyncio.sleep(0.05)
            self.write_event("farewell_failed", reason="Timed out waiting for audio; staying awake")
            return {"error": "Timed out waiting for farewell playback; robot remains awake."}
        finally:
            self.ending = False


class TimedMedia:
    """Track output queue duration while preserving the existing media routing."""

    def __init__(self, media, timing: ConversationTiming, recognition=None) -> None:
        self.media = media
        self.timing = timing
        self.recognition = recognition
        self.relational = None

    def __getattr__(self, name):
        return getattr(self.media, name)

    def push_audio_sample(self, samples) -> None:
        if not samples.size:
            return
        if self.relational is not None and not self.relational.output_allowed():
            return
        self.media.push_audio_sample(samples)
        now = monotonic()
        rate = self.media.get_output_audio_samplerate()
        with self.timing.state_lock:
            self.timing.playback_until = max(now, self.timing.playback_until) + len(samples) / rate
            self.timing.last_push_at = now
            if not self.timing.first_audio:
                self.timing.first_audio = True
                self.timing.write_event("first_reply_audio_submitted",
                                        measurement="speaker_queue_submission")

    async def finish_conversation(self, sleep_callback) -> dict:
        if self.relational is not None:
            origin = self.current_origin()
            if origin is None:
                return {"status": "blocked", "instruction": "This conversation has changed; do not end another session."}
            callback = sleep_callback

            def sleep_callback():
                self.relational.tick()
                if origin.epoch != self.relational.epoch:
                    return {"status": "cancelled_context_changed"}
                return callback()
        return self.timing.request_farewell(sleep_callback)

    async def person_identity(self, reject_current: bool = False) -> dict:
        if self.relational is not None and self.current_origin() is None:
            return {"state": "unknown", "reason": "context_changed"}
        if self.recognition is None:
            return {"state": "unknown", "reason": "recognition_disabled_or_unavailable"}
        if reject_current:
            result = self.recognition.reject_current()
            if self.relational is not None:
                self.relational.reject_identity()
            return result
        deadline = monotonic() + 6
        while True:
            if self.relational is not None:
                if self.current_origin() is None:
                    return {"state": "unknown", "reason": "context_changed"}
                confirmed = self.relational.confirmed_identity()
                if confirmed is not None:
                    return confirmed
            result = self.recognition.snapshot()
            if result["state"] == "known" or monotonic() >= deadline:
                if result["state"] == "unknown":
                    result["instruction"] = (
                        "Do not guess a name. Explain that you cannot get a confident face match "
                        "from this view. Ask the person to face the camera in good light and "
                        "hold still briefly for another check, or offer to use a name they tell you."
                    )
                return result
            await asyncio.sleep(.1)

    async def enroll_person(self, name: str, confirmed: bool) -> dict:
        origin = self.current_origin() if self.relational is not None else None
        if self.relational is not None and origin is None:
            return {"status": "blocked", "reason": "context_changed"}
        if self.recognition is None:
            return {"status": "unavailable", "instruction": "Face enrollment is unavailable; do not claim to save a name."}
        with self.timing.state_lock:
            seq, transcript = ((origin.transcript_seq, origin.transcript) if origin else
                               (self.timing.transcript_seq, self.timing.user_transcript))
        try:
            return self.recognition.enrollment(name, confirmed, seq, transcript)
        except ValueError as exc:
            logging.getLogger(__name__).warning("Identity enrollment declined: %s", exc)
            return {"status": "not_saved", "reason": str(exc)}

    async def memory_session(self, confirmed: bool) -> dict:
        if self.relational is None:
            return {"status": "disabled"}
        origin = self.current_origin()
        if origin is None:
            return {"status": "blocked", "reason": "context_changed"}
        return self.relational.session(confirmed, origin.transcript_seq, origin.transcript)

    async def memory_operation(self, action: str, **kwargs) -> dict:
        if self.relational is None:
            return {"status": "disabled"}
        origin = self.current_origin()
        if origin is None or origin.scope is None:
            return {"status": "blocked", "instruction": "A confirmed current-person memory session is required."}
        return await asyncio.to_thread(self.relational.operation, action, expected_scope=origin.scope,
                                       source_turn=origin.source_turn, **kwargs)

    def current_origin(self):
        from memory_origin import tool_origin
        self.relational.tick()
        origin = tool_origin.get()
        if origin is None or origin.runtime_id != self.relational.runtime_id or origin.epoch != self.relational.epoch:
            return None
        return origin

    def reset_output_timing(self) -> None:
        with self.timing.state_lock:
            self.timing.playback_until = monotonic()
            self.timing.last_push_at = 0
            self.timing.response_active = self.timing.awaiting_response = False
        self.media.reset_echo_gate()
