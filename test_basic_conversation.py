import asyncio
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from conversation_runtime import ConversationTiming, TimedMedia

ROOT = Path(__file__).resolve().parent


def event(timing, name):
    timing.emit(logging.LogRecord("realtime", logging.DEBUG, "", 0,
                                 "Realtime event: %s", (name,), None))


class ConversationTests(unittest.TestCase):
    def test_turn_timings_and_audio_submission(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("conversation_runtime.monotonic", return_value=100) as clock:
            timing = ConversationTiming(Path(directory) / "timings.jsonl")
            speaker = Mock()
            speaker.get_output_audio_samplerate.return_value = 16000
            media = TimedMedia(speaker, timing)
            event(timing, "input_audio_buffer.speech_stopped")
            clock.return_value = 100.2
            event(timing, "response.created")
            event(timing, "response.output_audio.delta")
            clock.return_value = 100.4
            media.push_audio_sample(np.ones(1600, dtype=np.float32))
            media.push_audio_sample(np.ones(1600, dtype=np.float32))
            entries = [json.loads(line) for line in timing.path.read_text().splitlines()]
            self.assertEqual([entry["event"] for entry in entries],
                             ["speech_ended", "processing_started", "first_reply_audio_submitted"])
            self.assertEqual(entries[-1]["since_speech_end_ms"], 400)
            self.assertAlmostEqual(timing.playback_until, 100.6)
            self.assertTrue(all(entry["turn"] == 1 for entry in entries))

    def test_farewell_waits_for_delivery_and_speaker_queue(self):
        async def exercise(timing, media, clock):
            sleeper = Mock(return_value={"status": "sleeping"})
            event(timing, "response.created")
            event(timing, "response.output_audio.delta")
            task = asyncio.create_task(timing.finish(sleeper))
            await asyncio.sleep(0)
            sleeper.assert_not_called()
            event(timing, "response.done")
            await asyncio.sleep(0.06)
            sleeper.assert_not_called()  # Backend complete, but audio not submitted yet.
            media.push_audio_sample(np.ones(32000, dtype=np.float32))
            clock.return_value = 101.0
            await asyncio.sleep(0.06)
            sleeper.assert_not_called()  # Two seconds remain queued.
            clock.return_value = 102.6
            result = await asyncio.wait_for(task, timeout=1)
            self.assertEqual(result["status"], "sleeping")
            sleeper.assert_called_once()

        with tempfile.TemporaryDirectory() as directory, \
             patch("conversation_runtime.monotonic", return_value=100) as clock:
            timing = ConversationTiming(Path(directory) / "timings.jsonl")
            speaker = Mock()
            speaker.get_output_audio_samplerate.return_value = 16000
            asyncio.run(exercise(timing, TimedMedia(speaker, timing), clock))

    def test_sleep_refused_without_farewell_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            timing = ConversationTiming(Path(directory) / "timings.jsonl")
            sleeper = Mock()
            result = asyncio.run(timing.finish(sleeper))
            self.assertIn("error", result)
            sleeper.assert_not_called()

    def test_end_tool_waits_for_next_spoken_response(self):
        async def exercise(timing, media, clock):
            sleeper = Mock(return_value={"status": "sleeping"})
            event(timing, "response.created")
            result = await media.finish_conversation(sleeper)
            self.assertEqual(result["status"], "farewell_requested")
            event(timing, "response.done")
            await asyncio.sleep(0.06)
            sleeper.assert_not_called()
            event(timing, "response.created")
            event(timing, "response.output_audio.delta")
            media.push_audio_sample(np.ones(16000, dtype=np.float32))
            event(timing, "response.done")
            await asyncio.sleep(0.06)
            sleeper.assert_not_called()
            clock.return_value = 102
            await asyncio.wait_for(timing.farewell_task, 1)
            sleeper.assert_called_once()

        with tempfile.TemporaryDirectory() as directory, \
             patch("conversation_runtime.monotonic", return_value=100) as clock:
            timing = ConversationTiming(Path(directory) / "timings.jsonl")
            speaker = Mock()
            speaker.get_output_audio_samplerate.return_value = 16000
            asyncio.run(exercise(timing, TimedMedia(speaker, timing), clock))

    def test_profile_loads_without_memory_tools_or_previous_facts(self):
        # A fresh process mirrors the launcher's import/environment boundary.
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env.update({"REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY": str(ROOT / "profiles"),
                        "REACHY_MINI_CUSTOM_PROFILE": "basic_conversation",
                        "REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY": str(ROOT / "conversation_tools"),
                        "AUTOLOAD_EXTERNAL_TOOLS": "false", "XDG_DATA_HOME": directory})
            code = """
import sys
from reachy_mini_conversation_app.tools.core_tools import initialize_tools, get_tools
from reachy_mini_conversation_app.prompts import get_session_instructions, get_session_greeting_prompt
initialize_tools(sys.argv[1])
assert set(get_tools()) == {'finish_conversation', 'person_identity', 'enroll_person', 'task_status', 'task_cancel'}, get_tools()
assert 'no persistent memory' in get_session_instructions(sys.argv[1])
assert 'Nice to meet you' in get_session_greeting_prompt()
from reachy_mini_conversation_app.memory import list_memory_facts
assert list_memory_facts(sys.argv[1]) == []
"""
            result = subprocess.run([str(ROOT / ".venv/bin/python"), "-c", code, directory],
                                    cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(list(Path(directory).rglob("memory.v1.json")))


if __name__ == "__main__":
    unittest.main()
