import json
import logging
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from attention import AttentionController, AttentionPolicy, AttentiveRobot
from conversation_runtime import ConversationTiming
from sound_direction import SoundDirection


class AttentionTests(unittest.TestCase):
    def test_multiple_faces_are_tracked_but_never_named_or_enrolled(self):
        with tempfile.TemporaryDirectory() as directory, patch("attention.monotonic", return_value=100):
            timing = ConversationTiming(Path(directory) / "timing.jsonl")
            robot = Mock()
            robot.media.get_frame.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
            robot.look_at_image.return_value = np.eye(4)
            vision, recognition = Mock(), Mock()
            vision.prepare.return_value = (robot.media.get_frame.return_value, (1.0, 1.0))
            vision.detect.return_value = [SimpleNamespace(bbox=(100, 70, 80, 100)),
                                          SimpleNamespace(bbox=(200, 70, 70, 90))]
            controller = AttentionController(robot, timing, Path(directory) / "attention.jsonl",
                                             recognition=recognition, vision=vision)
            controller.stop_event = Mock()
            controller.stop_event.is_set.side_effect = [False, True]
            controller._sense()
            recognition.invalidate.assert_called_once_with("multiple_faces")
            vision.embed.assert_not_called()
            self.assertIsNotNone(controller.policy.face)

    def test_sensor_loop_tracks_face_without_writing_motors_or_chasing_speaker(self):
        with tempfile.TemporaryDirectory() as directory, patch("attention.monotonic", return_value=100), \
             patch("reachy_mini.vision.face_detector.FaceDetector") as detector:
            timing = ConversationTiming(Path(directory) / "timing.jsonl")
            timing.last_push_at = 100
            timing.playback_until = 102
            robot = Mock()
            robot.media.get_frame.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler("z", 20, degrees=True).as_matrix()
            robot.look_at_image.return_value = pose
            detector.return_value.detect.return_value = [SimpleNamespace(bbox=(100, 70, 80, 100))]
            sound = Mock()
            controller = AttentionController(robot, timing, Path(directory) / "attention.jsonl", sound)
            controller.stop_event = Mock()
            controller.stop_event.is_set.side_effect = [False, True]
            controller._sense()
            robot.look_at_image.assert_called_once_with(140, 120, perform_movement=False)
            self.assertAlmostEqual(controller.policy.face[0], math.radians(20))
            sound.read.assert_not_called()
            robot.set_target.assert_not_called()

    def test_camera_failure_still_allows_sound_attention(self):
        with tempfile.TemporaryDirectory() as directory, patch("attention.monotonic", return_value=100), \
             patch("reachy_mini.vision.face_detector.FaceDetector"):
            timing = ConversationTiming(Path(directory) / "timing.jsonl")
            robot = Mock()
            robot.media.get_frame.side_effect = RuntimeError("Camera disconnected")
            sound = Mock()
            sound.read.return_value = (math.radians(30), True)
            controller = AttentionController(robot, timing, Path(directory) / "attention.jsonl", sound)
            controller.stop_event = Mock()
            controller.stop_event.is_set.side_effect = [False, True]
            with self.assertLogs("attention", "WARNING"):
                controller._sense()
            self.assertAlmostEqual(controller.policy.sound_yaw, math.radians(30))
            self.assertEqual(controller.policy.step(100, .1, False, False, False)[0], "listening")

    def test_all_states_and_face_loss_return_to_idle(self):
        policy = AttentionPolicy()
        self.assertEqual(policy.step(10, .1, False, False, False)[0], "idle")
        policy.observe_face(.3, -.1, 10)
        self.assertEqual(policy.step(10, .1, False, False, False)[0], "attending")
        self.assertEqual(policy.step(10, .1, False, True, False)[0], "listening")
        self.assertEqual(policy.step(10, .1, False, False, True)[0], "thinking")
        self.assertEqual(policy.step(10, .1, True, True, True)[0], "speaking")
        self.assertEqual(policy.step(11, .1, False, False, False)[0], "attending")
        self.assertEqual(policy.step(12, .1, False, False, False)[0], "idle")
        for _ in range(100):
            policy.step(20, .1, False, False, False)
        self.assertAlmostEqual(policy.yaw, 0, places=5)
        self.assertAlmostEqual(policy.pitch, 0, places=5)

    def test_face_refines_sound_and_sound_recovers_when_face_lost(self):
        policy = AttentionPolicy()
        policy.observe_sound(math.radians(-30), True, 10)
        self.assertEqual(policy.step(10, .1, False, False, False)[0], "listening")
        self.assertLess(policy.yaw, 0)
        policy.observe_face(math.radians(20), 0, 10)
        for _ in range(20):
            policy.step(10.5, .1, False, False, False)
        self.assertGreater(policy.yaw, 0)
        policy.observe_sound(math.radians(-30), True, 12)
        for _ in range(20):
            policy.step(12, .1, False, False, False)
        self.assertLess(policy.yaw, 0)
        self.assertEqual(policy.step(15, .1, False, False, False)[0], "idle")

    def test_unvoiced_or_invalid_observations_do_not_create_targets(self):
        policy = AttentionPolicy()
        policy.observe_sound(1, False, 10)
        policy.observe_sound(float("nan"), True, 10)
        policy.observe_face(float("nan"), 0, 10)
        self.assertEqual(policy.step(10, .1, False, False, False), ("idle", 0, 0))

    def test_sudden_target_changes_respect_speed_and_angle_limits(self):
        policy = AttentionPolicy()
        policy.observe_face(3, -3, 10)
        previous = (0, 0)
        for _ in range(100):
            _, yaw, pitch = policy.step(10, .02, False, False, False)
            self.assertLessEqual(abs(yaw - previous[0]), math.radians(30) * .02 + 1e-9)
            self.assertLessEqual(abs(pitch - previous[1]), math.radians(30) * .02 + 1e-9)
            self.assertLessEqual(abs(yaw), math.radians(90))
            self.assertLessEqual(abs(pitch), math.radians(20))
            previous = yaw, pitch
        policy.observe_face(-3, 3, 10)
        _, yaw, pitch = policy.step(10, 20, False, False, False)
        self.assertLessEqual(abs(yaw - previous[0]), math.radians(3) + 1e-9)
        self.assertLessEqual(abs(pitch - previous[1]), math.radians(3) + 1e-9)

    def test_sound_targets_reach_both_90_degree_limits(self):
        for direction in [-1, 1]:
            policy = AttentionPolicy()
            for tick in range(600):
                now = tick * .02
                policy.observe_sound(direction * math.radians(120), True, now)
                policy.step(now, .02, False, False, False)
            self.assertAlmostEqual(math.degrees(policy.yaw), direction * 90, places=4)

    def test_proxy_keeps_one_motor_writer_and_stops_before_sleep(self):
        robot = Mock()
        controller = Mock()
        controller.orient.return_value = np.eye(4)
        proxy = AttentiveRobot(robot, controller)
        proxy.set_target(head=np.eye(4), antennas=[.1, -.1], body_yaw=0)
        robot.set_target.assert_called_once()
        self.assertEqual(robot.set_target.call_args.kwargs["antennas"], [.1, -.1])
        controller.orient.assert_called_once()
        proxy.enable_wobbling()
        robot.enable_wobbling.assert_not_called()
        order = Mock()
        order.attach_mock(controller.stop, "stop")
        order.attach_mock(robot.goto_sleep, "sleep")
        proxy.goto_sleep()
        self.assertEqual([call[0] for call in order.mock_calls], ["stop", "sleep"])

    def test_controller_uses_actual_playback_and_expires_backend_flags(self):
        with tempfile.TemporaryDirectory() as directory, patch("attention.monotonic", return_value=100) as clock:
            timing = ConversationTiming(Path(directory) / "timing.jsonl")
            robot = Mock()
            robot.get_current_head_pose.return_value = np.eye(4)
            controller = AttentionController(robot, timing, Path(directory) / "attention.jsonl")
            timing.user_speaking = True
            timing.activity_at = 100
            controller.orient(np.eye(4))
            self.assertEqual(controller.policy.state, "listening")
            timing.user_speaking = False
            timing.response_active = True
            controller.orient(np.eye(4))
            self.assertEqual(controller.policy.state, "thinking")
            timing.response_active = False  # Generation done; output still playing.
            timing.last_push_at = 100
            timing.playback_until = 102
            controller.orient(np.eye(4))
            self.assertEqual(controller.policy.state, "speaking")
            clock.return_value = 103
            timing.user_speaking = True
            timing.activity_at = 100
            controller.orient(np.eye(4))
            clock.return_value = 130
            controller.orient(np.eye(4))
            self.assertEqual(controller.policy.state, "idle")
            states = [json.loads(line)["state"] for line in controller.path.read_text().splitlines()]
            self.assertEqual(states, ["listening", "thinking", "speaking", "listening", "idle"])

    def test_neutral_orientation_does_not_remove_breathing_translation(self):
        with tempfile.TemporaryDirectory() as directory, patch("attention.monotonic", return_value=100):
            timing = ConversationTiming(Path(directory) / "timing.jsonl")
            robot = Mock()
            robot.get_current_head_pose.return_value = np.eye(4)
            controller = AttentionController(robot, timing, Path(directory) / "attention.jsonl")
            pose = np.eye(4)
            pose[2, 3] = .005
            np.testing.assert_allclose(controller.orient(pose)[:3, 3], pose[:3, 3])
            robot.set_target.assert_not_called()  # Only the app's motor loop writes.
            controller.stop()
            np.testing.assert_array_equal(controller.orient(pose), pose)

    def test_realtime_events_supply_attention_state(self):
        with tempfile.TemporaryDirectory() as directory:
            timing = ConversationTiming(Path(directory) / "timing.jsonl")
            for event, speaking, waiting in [("input_audio_buffer.speech_started", True, False),
                                            ("input_audio_buffer.speech_stopped", False, True),
                                            ("response.done", False, False)]:
                timing.emit(logging.LogRecord("realtime", logging.DEBUG, "", 0,
                                             "Realtime event: %s", (event,), None))
                self.assertEqual((timing.user_speaking, timing.awaiting_response), (speaking, waiting))


class DirectionTests(unittest.TestCase):
    def test_xvf3000_protocol_and_fixed_angle_mapping(self):
        device = Mock()
        device.ctrl_transfer.side_effect = [struct.pack("<ii", 0, 0), struct.pack("<ii", 0, 0),
                                           struct.pack("<ii", 1, 0), struct.pack("<ii", 30, 0),
                                           struct.pack("<ii", 1, 0), struct.pack("<ii", 330, 0)]
        with patch("sound_direction.usb.core.find", return_value=[device]) as find, \
             patch("sound_direction.get_libusb1_backend", return_value=Mock()), \
             patch("sound_direction.usb.util.dispose_resources") as dispose:
            sound = SoundDirection()
            self.assertEqual(find.call_args.kwargs["idProduct"], 0x0018)
            self.assertAlmostEqual(sound.read()[0], math.radians(30))
            angle, voice = sound.read()
            self.assertAlmostEqual(angle, math.radians(-30))
            self.assertTrue(voice)
            calls = device.ctrl_transfer.call_args_list
            self.assertEqual(calls[0].args, (0xC0, 0, 0xE0, 19, 8))
            self.assertEqual(calls[1].args, (0xC0, 0, 0xC0, 21, 8))
            sound.close()
            dispose.assert_called_once_with(device)

    def test_ambiguous_arrays_are_not_silently_selected(self):
        with patch("sound_direction.usb.core.find", return_value=[Mock(), Mock()]):
            with self.assertRaises(RuntimeError):
                SoundDirection()


if __name__ == "__main__":
    unittest.main()
