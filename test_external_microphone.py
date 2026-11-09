import unittest
from unittest.mock import Mock, patch

import numpy as np

from external_microphone import ExternalMicrophoneMedia, select_input


class ExternalMicrophoneTests(unittest.TestCase):
    def test_processed_channel_and_reachy_output_are_independent(self):
        device = {"name": "ReSpeaker 4 Mic Array", "max_input_channels": 6}
        speaker = Mock()
        speaker.get_output_audio_samplerate.return_value = 16000
        stream = Mock(read_available=3)
        samples = np.array([[0.1, 1, 2, 3, 4, 5]] * 3, dtype=np.float32)
        stream.read.return_value = (samples, False)
        with patch("external_microphone.sd.query_devices", return_value=[device]), \
             patch("external_microphone.sd.check_input_settings"), \
             patch("external_microphone.sd.InputStream", return_value=stream) as factory:
            media = ExternalMicrophoneMedia(speaker, "respeaker")
            media.start_recording()
            factory.assert_called_once_with(device=0, samplerate=16000,
                                            channels=6, dtype="float32")
            np.testing.assert_array_equal(media.get_audio_sample(), samples[:, :1])
            media.push_audio_sample(samples[:, :1])
            speaker.push_audio_sample.assert_called_once()
            speaker.start_recording.assert_not_called()
            media.stop_recording()
            self.assertIsNone(media.get_audio_sample())
            media.close()
            stream.close.assert_called_once()
            speaker.close.assert_called_once()

    def test_missing_or_ambiguous_input_never_falls_back(self):
        devices = [{"name": "MacBook Microphone", "max_input_channels": 1},
                   {"name": "reSpeaker A", "max_input_channels": 1},
                   {"name": "reSpeaker B", "max_input_channels": 6}]
        with patch("external_microphone.sd.query_devices", return_value=devices):
            for selector in ["respeaker", "missing", "9"]:
                with self.assertRaises(ValueError):
                    select_input(selector)
            self.assertEqual(select_input("2"), (2, devices[2]))
            self.assertEqual(select_input("respeaker a"), (1, devices[1]))

    def test_invalid_channel_rejected_before_recording(self):
        with patch("external_microphone.sd.query_devices", return_value=[
                {"name": "reSpeaker", "max_input_channels": 1}]):
            with self.assertRaises(ValueError):
                ExternalMicrophoneMedia(Mock(), "respeaker", channel=1)

    def test_overflow_preserves_audio_and_subsequent_capture(self):
        samples = np.array([[0.2], [0.3]], dtype=np.float32)
        stream = Mock(read_available=2)
        stream.read.side_effect = [(samples, True), (samples, True), (samples, False)]
        with patch("external_microphone.sd.query_devices", return_value=[
                {"name": "reSpeaker", "max_input_channels": 1}]), \
             patch("external_microphone.sd.check_input_settings"), \
             patch("external_microphone.sd.InputStream", return_value=stream):
            media = ExternalMicrophoneMedia(Mock(), "respeaker")
            media.start_recording()
            try:
                with self.assertLogs("external_microphone", level="WARNING") as logs:
                    for _ in range(3):
                        np.testing.assert_array_equal(media.get_audio_sample(), samples)
                self.assertEqual(len(logs.output), 1)
                self.assertIn("continuing capture", logs.output[0])
                stream.close.assert_not_called()
            finally:
                media.close()

    def test_speaker_echo_suppressed_until_queued_audio_and_tail_finish(self):
        speaker = Mock()
        speaker.get_output_audio_samplerate.return_value = 16000
        stream = Mock(read_available=2)
        voice = np.array([[0.2], [0.3]], dtype=np.float32)
        stream.read.return_value = (voice, False)
        with patch("external_microphone.sd.query_devices", return_value=[
                {"name": "reSpeaker", "max_input_channels": 1}]), \
             patch("external_microphone.sd.check_input_settings"), \
             patch("external_microphone.sd.InputStream", return_value=stream), \
             patch("external_microphone.time.monotonic", return_value=100.0) as clock:
            media = ExternalMicrophoneMedia(speaker, "respeaker")
            media.start_recording()
            try:
                np.testing.assert_array_equal(media.get_audio_sample(), voice)
                # Two queued seconds, even though the backend supplies both immediately.
                media.push_audio_sample(np.ones(16000, dtype=np.float32))
                media.push_audio_sample(np.ones(16000, dtype=np.float32))
                for now in [100.0, 101.5, 102.4]:
                    clock.return_value = now
                    np.testing.assert_array_equal(media.get_audio_sample(), np.zeros((2, 1)))
                clock.return_value = 102.6
                np.testing.assert_array_equal(media.get_audio_sample(), voice)
                self.assertEqual(speaker.push_audio_sample.call_count, 2)
                self.assertEqual(stream.read.call_count, 5)
            finally:
                media.close()


if __name__ == "__main__":
    unittest.main()
