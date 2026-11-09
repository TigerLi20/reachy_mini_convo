"""Face/sound attention with one coordinated path to the robot's motor targets."""

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import threading
from time import monotonic

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


class AttentionPolicy:
    """Fuse fresh observations, expire lost targets, and limit orientation speed."""

    def __init__(self) -> None:
        self.face = None
        self.face_at = float("-inf")
        self.sound_yaw = 0.0
        self.sound_at = float("-inf")
        self.voice_at = float("-inf")
        self.yaw = 0.0
        self.pitch = 0.0
        self.state = "idle"

    def observe_face(self, yaw: float, pitch: float, now: float) -> None:
        if math.isfinite(yaw) and math.isfinite(pitch):
            self.face = (yaw, pitch)
            self.face_at = now

    def observe_sound(self, yaw: float, voice: bool, now: float) -> None:
        if voice and math.isfinite(yaw):
            self.sound_yaw = yaw
            self.sound_at = self.voice_at = now

    def step(self, now: float, dt: float, speaking: bool, listening: bool, thinking: bool):
        face = self.face is not None and now - self.face_at < 1.2
        sound = now - self.sound_at < 2.0
        if speaking:
            self.state = "speaking"
        elif listening or now - self.voice_at < 0.35:
            self.state = "listening"
        elif thinking:
            self.state = "thinking"
        elif face or sound:
            self.state = "attending"
        else:
            self.state = "idle"
        # Face refines gaze; sound supplies a bearing when the person is out of view.
        target = self.face if face else (self.sound_yaw, 0.0) if sound else (0.0, 0.0)
        yaw = float(np.clip(target[0], -math.radians(90), math.radians(90)))
        pitch = float(np.clip(target[1], -math.radians(20), math.radians(15)))
        # Exponential settling plus a hard 30 degrees/second speed bound.
        dt = max(0.0, min(dt, 0.1))
        alpha = 1 - math.exp(-dt / 0.25)
        limit = math.radians(30) * dt
        self.yaw += float(np.clip((yaw - self.yaw) * alpha, -limit, limit))
        self.pitch += float(np.clip((pitch - self.pitch) * alpha, -limit, limit))
        return self.state, self.yaw, self.pitch


class AttentionController:
    """Collect observations off-thread; the existing movement loop applies them."""

    def __init__(self, robot, timing, path: Path, sound=None, recognition=None, vision=None) -> None:
        self.robot = robot
        self.timing = timing
        self.path = path
        self.sound = sound
        self.recognition = recognition
        self.vision = vision
        self.last_embedding_at = float("-inf")
        self.policy = AttentionPolicy()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.enabled = True
        self.last_tick = None
        self.last_state = None
        self.last_log_at = 0.0
        self.previous_center = None
        self.previous_face_at = float("-inf")
        self.last_sensor_warning = float("-inf")

    def start(self) -> None:
        # Eliminate independent daemon orientation writers left by previous apps.
        self.robot.stop_head_tracking()
        self.robot.disable_wobbling()
        self.thread = threading.Thread(target=self._sense, name="attention-sensors", daemon=True)
        self.thread.start()

    def _sense(self) -> None:
        detector = None
        try:
            if self.vision is not None:
                detector = self.vision
            else:
                from reachy_mini.vision.face_detector import FaceDetector
                detector = FaceDetector()
        except Exception:
            logger.exception("Face detector unavailable; sound attention remains enabled")
        while not self.stop_event.is_set():
            now = monotonic()
            try:
                if detector is not None:
                    frame = self.robot.media.get_frame()
                    if frame is not None:
                        if self.vision is not None:
                            small, scale = self.vision.prepare(frame)
                        else:
                            scale = max(1, math.ceil(frame.shape[1] / 320))
                            small = np.ascontiguousarray(frame[::scale, ::scale])
                        faces = detector.detect(small)
                        if self.recognition is not None:
                            self.recognition.report_scene(len(faces))
                            if len(faces) != 1:
                                if faces:
                                    self.recognition.invalidate("multiple_faces")
                                else:
                                    self.recognition.invalidate("no_face", transient=True)
                            elif monotonic() - self.last_embedding_at >= .4:
                                self.last_embedding_at = monotonic()
                                try:
                                    embedding = self.vision.embed(frame, faces[0], scale)
                                    self.recognition.observe(embedding)
                                except ValueError as exc:
                                    self.recognition.invalidate(str(exc), transient=True)
                        if faces:
                            centers = [((f.bbox[0] + f.bbox[2] / 2) / small.shape[1],
                                        (f.bbox[1] + f.bbox[3] / 2) / small.shape[0]) for f in faces]
                            if self.previous_center is not None and now - self.previous_face_at < 1.2:
                                index = min(range(len(faces)), key=lambda i:
                                            math.dist(centers[i], self.previous_center))
                            else:
                                index = max(range(len(faces)), key=lambda i: faces[i].bbox[2] * faces[i].bbox[3])
                            center = centers[index]
                            self.previous_center, self.previous_face_at = center, now
                            height, width = frame.shape[:2]
                            u = int(np.clip(center[0] * width, 1, width - 1))
                            v = int(np.clip(center[1] * height, 1, height - 1))
                            pose = self.robot.look_at_image(u, v, perform_movement=False)
                            _, pitch, yaw = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")
                            with self.lock:
                                self.policy.observe_face(yaw, pitch, monotonic())
                    elif self.recognition is not None:
                        self.recognition.report_scene(0)
                        self.recognition.invalidate("no_frame", transient=True)
            except Exception as exc:
                if self.recognition is not None:
                    self.recognition.invalidate("camera_unavailable")
                self._warn("Face tracking", exc)
            try:
                with self.timing.state_lock:
                    speaking = monotonic() < self.timing.playback_until + 0.5
                if self.sound is not None and not speaking:
                    reading = self.sound.read()
                    if reading is not None:
                        with self.lock:
                            self.policy.observe_sound(*reading, monotonic())
            except Exception as exc:
                self._warn("Sound direction", exc)
            self.stop_event.wait(0.1)

    def _warn(self, source, exc) -> None:
        if monotonic() - self.last_sensor_warning > 10:
            logger.warning("%s unavailable: %s; continuing with remaining observations", source, exc)
            self.last_sensor_warning = monotonic()

    def orient(self, head):
        now = monotonic()
        with self.timing.state_lock:
            speaking = self.timing.last_push_at > 0 and now < self.timing.playback_until + 0.5
            listening = self.timing.user_speaking and now - self.timing.activity_at < 15
            thinking = (self.timing.response_active or self.timing.awaiting_response) and now - self.timing.activity_at < 20
        with self.lock:
            if not self.enabled:
                return head
            if self.last_tick is None:
                # Seed from the actual pose after the app has woken the robot.
                actual = self.robot.get_current_head_pose()
                _, self.policy.pitch, self.policy.yaw = Rotation.from_matrix(actual[:3, :3]).as_euler("xyz")
                self.last_tick = now
            state, yaw, pitch = self.policy.step(now, now - self.last_tick, speaking, listening, thinking)
            self.last_tick = now
            result = head.copy()
            result[:3, :3] = Rotation.from_euler("xyz", [0, pitch, yaw]).as_matrix()
            if state != self.last_state or now - self.last_log_at >= 1:
                entry = {"time_utc": datetime.now(timezone.utc).isoformat(), "monotonic_s": now,
                         "state": state, "yaw_degrees": round(math.degrees(yaw), 1),
                         "pitch_degrees": round(math.degrees(pitch), 1),
                         "face_visible": now - self.policy.face_at < 0.4,
                         "target_source": "face" if now - self.policy.face_at < 1.2
                         else "sound" if now - self.policy.sound_at < 2 else "neutral"}
                with self.path.open("a") as log:
                    log.write(json.dumps(entry) + "\n")
                if state != self.last_state:
                    logger.info("Attention: %s (target=%s)", state, entry["target_source"])
                self.last_state, self.last_log_at = state, now
            return result

    def stop(self) -> None:
        with self.lock:
            self.enabled = False
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)


class AttentiveRobot:
    """Compose attention into the app's existing motor writer and sleep lifecycle."""

    def __init__(self, robot, attention: AttentionController) -> None:
        self.robot = robot
        self.attention = attention

    def __getattr__(self, name):
        return getattr(self.robot, name)

    def set_target(self, head=None, antennas=None, body_yaw=None) -> None:
        if head is not None:
            head = self.attention.orient(head)
        self.robot.set_target(head=head, antennas=antennas, body_yaw=body_yaw)

    def enable_wobbling(self) -> None:
        logger.info("Attention controls head orientation; speech wobbling remains disabled")

    def goto_sleep(self, *args, **kwargs):
        self.attention.stop()
        return self.robot.goto_sleep(*args, **kwargs)
