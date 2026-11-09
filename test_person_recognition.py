import asyncio
import json
import logging
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid

import numpy as np

from conversation_runtime import ConversationTiming, TimedMedia
from person_recognition import IdentityStore, PersonRecognition, MODEL_ID
from face_embeddings import FaceEmbeddings, RecognitionFace


def vector(axis=0):
    value = np.zeros(128, dtype=np.float32)
    value[axis] = 1
    return value


def fill(recognition, embedding, start=0, count=8):
    for i in range(count):
        recognition.observe(embedding, now=start + i * .4)


class RecognitionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "people" / "identities.json"
        self.store = IdentityStore(self.path)

    def test_fresh_session_recognizes_same_id_without_other_memory(self):
        person_id = self.store.enroll("Taylor", [vector()] * 8)
        self.assertEqual(str(uuid.UUID(person_id)), person_id)
        fresh_store = IdentityStore(self.path)
        recognition = PersonRecognition(fresh_store)
        with patch("person_recognition.monotonic", return_value=3):
            fill(recognition, vector())
            result = recognition.snapshot()
        self.assertEqual(result["state"], "known")
        self.assertEqual(result["person_id"], person_id)
        self.assertEqual(result["name"], "Taylor")
        saved = json.loads(self.path.read_text())
        self.assertEqual(set(saved), {"schema_version", "model", "people"})
        self.assertEqual(set(saved["people"][0]), {"person_id", "name", "embeddings"})
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_a_single_strong_frame_is_not_enough_and_stale_identity_expires(self):
        self.store.enroll("Taylor", [vector()] * 8)
        recognition = PersonRecognition(self.store)
        recognition.observe(vector(), now=0)
        self.assertEqual(recognition.result["state"], "unknown")
        fill(recognition, vector(), start=.4, count=5)
        self.assertEqual(recognition.result["state"], "known")
        with patch("person_recognition.monotonic", return_value=4):
            result = recognition.snapshot()
        self.assertEqual(result["state"], "unknown")
        self.assertNotIn("name", result)

    def test_brief_detector_dropouts_keep_progress_but_never_return_a_stale_name(self):
        self.store.enroll("Taylor", [vector()] * 8)
        recognition = PersonRecognition(self.store)
        with patch("person_recognition.monotonic", return_value=0) as clock:
            for i in range(5):
                now = i * .5
                clock.return_value = now
                recognition.observe(vector(), now=now)
                if i < 4:
                    clock.return_value = now + .1
                    recognition.invalidate("no_face", transient=True)
                    self.assertEqual(recognition.snapshot()["state"], "unknown")
                    self.assertNotIn("name", recognition.snapshot())
            self.assertEqual(recognition.snapshot()["state"], "known")
            self.assertEqual(recognition.result["name"], "Taylor")
            clock.return_value = 4
            recognition.invalidate("no_face", transient=True)
            self.assertEqual(len(recognition.match_times), 0)
            self.assertEqual(recognition.snapshot()["state"], "unknown")

    def test_face_change_after_dropout_requires_new_evidence(self):
        self.store.enroll("Taylor", [vector()] * 8)
        self.store.enroll("Alex", [vector(1)] * 8)
        recognition = PersonRecognition(self.store)
        with patch("person_recognition.monotonic", return_value=1.3):
            fill(recognition, vector(), count=4)
            recognition.invalidate("no_face", transient=True)
            recognition.observe(vector(1), now=1.6)
            self.assertEqual(recognition.result["state"], "unknown")
            self.assertEqual(len(recognition.match_times), 1)
            fill(recognition, vector(1), start=2, count=4)
            self.assertEqual(recognition.result["state"], "known")
            self.assertEqual(recognition.result["name"], "Alex")

    def test_recognition_does_not_accumulate_sparse_matches_forever(self):
        self.store.enroll("Taylor", [vector()] * 8)
        recognition = PersonRecognition(self.store)
        for i in range(12):
            recognition.observe(vector(), now=i * 1.1)
            self.assertEqual(recognition.result["state"], "unknown")

    def test_enrollment_needs_a_current_clear_face_even_when_progress_is_preserved(self):
        recognition = PersonRecognition(self.store)
        with patch("person_recognition.monotonic", return_value=3):
            fill(recognition, vector())
            recognition.invalidate("Face image is blurry or poorly lit", transient=True)
            self.assertEqual(len(recognition.samples), 8)
            self.assertEqual(recognition.enrollment("Taylor", False, 1, "Taylor")["status"], "need_clear_face")
        self.assertFalse(self.path.exists())

    def test_weak_match_is_unknown_without_suggesting_a_name(self):
        self.store.enroll("Taylor", [vector()] * 8)
        weak = .59 * vector() + np.sqrt(1 - .59 ** 2) * vector(1)
        recognition = PersonRecognition(self.store)
        fill(recognition, weak)
        self.assertEqual(recognition.result["state"], "unknown")
        self.assertNotIn("name", recognition.result)
        self.assertNotIn("person_id", recognition.result)
        self.assertAlmostEqual(recognition.result["cosine_similarity"], .59)

    def test_close_runner_up_is_unknown_even_with_high_similarity(self):
        # Use a fixture store containing two similar enrolled people.
        other_id = str(uuid.uuid4())
        self.store.enroll("Taylor", [vector()] * 8)
        similar = .9 * vector() + np.sqrt(1 - .9 ** 2) * vector(1)
        self.store.people[other_id] = {"name": "Alex", "embeddings": [similar] * 8}
        recognition = PersonRecognition(self.store)
        fill(recognition, vector())
        self.assertEqual(recognition.result["state"], "unknown")
        self.assertNotIn("name", recognition.result)

    def test_person_change_and_multiple_faces_clear_known_identity(self):
        self.store.enroll("Taylor", [vector()] * 8)
        recognition = PersonRecognition(self.store)
        fill(recognition, vector())
        self.assertEqual(recognition.result["state"], "known")
        recognition.observe(vector(1), now=3.2)
        self.assertEqual(recognition.result["state"], "unknown")
        fill(recognition, vector(), start=4)
        recognition.invalidate("multiple_faces")
        self.assertEqual(recognition.result, {"state": "unknown", "reason": "multiple_faces"})
        self.assertEqual(len(recognition.samples), 0)

    def test_enrollment_requires_preparation_and_new_spoken_confirmation(self):
        recognition = PersonRecognition(self.store)
        with patch("person_recognition.monotonic", return_value=3):
            fill(recognition, vector())
            result = recognition.enrollment("Taylor", True, 1, "Yes")
            self.assertEqual(result["status"], "confirmation_required")
            self.assertFalse(self.path.exists())
            for seq, text in [(1, "yes"), (2, "no"), (2, "yes but that's not right"), (2, "maybe")]:
                result = recognition.enrollment("Taylor", True, seq, text)
                self.assertEqual(result["status"], "confirmation_required")
                self.assertFalse(self.path.exists())
            result = recognition.enrollment("Taylor", True, 3, "Yes, please.")
        self.assertEqual(result["status"], "enrolled")
        self.assertEqual(IdentityStore(self.path).people[result["person_id"]]["name"], "Taylor")

    def test_face_swap_during_confirmation_does_not_enroll(self):
        recognition = PersonRecognition(self.store)
        with patch("person_recognition.monotonic", return_value=3) as clock:
            fill(recognition, vector())
            recognition.enrollment("Taylor", False, 1, "My name is Taylor")
            fill(recognition, vector(1), start=4)
            clock.return_value = 7
            result = recognition.enrollment("Taylor", True, 2, "Yes")
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(self.path.exists())

    def test_short_or_stale_enrollment_does_not_save(self):
        recognition = PersonRecognition(self.store)
        with patch("person_recognition.monotonic", return_value=2):
            fill(recognition, vector(), count=5)
            self.assertEqual(recognition.enrollment("Taylor", False, 1, "Taylor")["status"], "need_clear_face")
        with patch("person_recognition.monotonic", return_value=20):
            fill(recognition, vector())
            self.assertEqual(recognition.enrollment("Taylor", False, 1, "Taylor")["status"], "need_clear_face")
        self.assertFalse(self.path.exists())

    def test_existing_face_keeps_its_id_and_name_is_not_silently_overwritten(self):
        person_id = self.store.enroll("Taylor", [vector()] * 8)
        self.assertEqual(self.store.enroll("Taylor", [vector()] * 8), person_id)
        with self.assertRaises(ValueError):
            self.store.enroll("Alex", [vector()] * 8)
        self.assertEqual(IdentityStore(self.path).people[person_id]["name"], "Taylor")

    def test_corrupt_store_is_not_replaced(self):
        self.path.parent.mkdir()
        for content in ["broken json", json.dumps({"schema_version": 1, "model": "other", "people": []})]:
            self.path.write_text(content)
            with self.assertRaises(ValueError):
                IdentityStore(self.path)
            self.assertEqual(self.path.read_text(), content)

    def test_logs_exclude_names_and_face_vectors(self):
        self.store.enroll("Taylor", [vector()] * 8)
        path = Path(self.directory.name) / "recognition.jsonl"
        recognition = PersonRecognition(self.store, path)
        fill(recognition, vector())
        text = path.read_text()
        self.assertNotIn("Taylor", text)
        self.assertNotIn("embeddings", text)
        self.assertIn('"state": "known"', text)

    def test_explicit_identity_correction_suppresses_name_for_session(self):
        self.store.enroll("Taylor", [vector()] * 8)
        recognition = PersonRecognition(self.store)
        fill(recognition, vector())
        self.assertEqual(recognition.result["state"], "known")
        self.assertEqual(recognition.reject_current()["state"], "unknown")
        fill(recognition, vector(), start=4)
        self.assertEqual(recognition.result["state"], "unknown")
        self.assertNotIn("name", recognition.result)
        self.assertEqual(len(IdentityStore(self.path).people), 1)

    def test_conversation_tool_uses_observed_transcript_for_confirmation(self):
        recognition = PersonRecognition(self.store)
        timing = ConversationTiming(Path(self.directory.name) / "timing.jsonl")
        media = TimedMedia(Mock(), timing, recognition)

        def transcript(text):
            timing.emit(logging.LogRecord("realtime", logging.DEBUG, "", 0,
                                         "User transcript: %s", (text,), None))

        with patch("person_recognition.monotonic", return_value=3):
            fill(recognition, vector())
            transcript("My name is Taylor")
            result = asyncio.run(media.enroll_person("Taylor", False))
            self.assertEqual(result["status"], "confirmation_required")
            self.assertEqual(asyncio.run(media.enroll_person("Taylor", True))["status"], "confirmation_required")
            transcript("Yes")
            self.assertEqual(asyncio.run(media.enroll_person("Taylor", True))["status"], "enrolled")


class EmbeddingTests(unittest.TestCase):
    def test_detector_preparation_preserves_detail_and_scales_both_axes(self):
        vision = FaceEmbeddings.__new__(FaceEmbeddings)
        frame = np.random.default_rng(2).integers(0, 256, (721, 1280, 3), dtype=np.uint8)
        small, scale = vision.prepare(frame)
        self.assertEqual(small.shape, (360, 640, 3))
        self.assertEqual(scale, (2, 721 / 360))
        # Averaging resampling must differ from the old every-fourth-pixel decimation.
        self.assertFalse(np.array_equal(small, frame[::2, ::2][:360]))

    def test_full_resolution_landmarks_are_scaled_for_alignment(self):
        vision = FaceEmbeddings.__new__(FaceEmbeddings)
        vision.encoder = Mock()
        aligned = np.random.default_rng(3).integers(0, 256, (112, 112, 3), dtype=np.uint8)
        vision.encoder.alignCrop.return_value = aligned
        vision.encoder.feature.return_value = vector()[None, :]
        row = np.array([10, 10, 30, 40, 17, 20, 33, 20, 25, 27, 19, 35, 31, 35, .95], dtype=np.float32)
        face = RecognitionFace(tuple(row[:4]), row)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        embedding = vision.embed(frame, face, scale=4)
        scaled = vision.encoder.alignCrop.call_args.args[1]
        np.testing.assert_allclose(scaled[:14], row[:14] * 4)
        self.assertEqual(scaled[14], row[14])
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1)
        np.testing.assert_array_equal(face.row, row)

    def test_small_faces_are_rejected_before_embedding(self):
        vision = FaceEmbeddings.__new__(FaceEmbeddings)
        vision.encoder = Mock()
        row = np.array([10, 10, 10, 10, 12, 12, 18, 12, 15, 15, 13, 18, 17, 18, .95], dtype=np.float32)
        with self.assertRaises(ValueError):
            vision.embed(np.zeros((480, 640, 3), dtype=np.uint8), RecognitionFace(tuple(row[:4]), row), 4)
        vision.encoder.feature.assert_not_called()


if __name__ == "__main__":
    unittest.main()
