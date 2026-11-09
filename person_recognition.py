"""Identity-only local enrollment and conservative temporal face matching."""

from collections import deque
import json
import os
from pathlib import Path
import re
import threading
from time import monotonic
import uuid

import numpy as np

MODEL_ID = "opencv-sface-2021dec-128"
MATCH_THRESHOLD = .60
MATCH_MARGIN = .12
MIN_STABLE_SAMPLES = 5


def unit_vector(value) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (128,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-6:
        raise ValueError("Invalid identity embedding")
    return vector / np.linalg.norm(vector)


class IdentityStore:
    """Persist only stable IDs, display names, and enrollment embeddings."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.people = {}
        if path.exists():
            content = json.loads(path.read_text())
            if content.get("schema_version") != 1 or content.get("model") != MODEL_ID:
                raise ValueError("Identity store schema/model mismatch; refusing to replace it")
            for person in content["people"]:
                person_id = str(uuid.UUID(person["person_id"]))
                name = self.validate_name(person["name"])
                embeddings = [unit_vector(sample) for sample in person["embeddings"]]
                if not 3 <= len(embeddings) <= 12 or person_id in self.people:
                    raise ValueError("Invalid identity enrollment record")
                self.people[person_id] = {"name": name, "embeddings": embeddings}

    @staticmethod
    def validate_name(name: str) -> str:
        if not isinstance(name, str) or any(ord(char) < 32 for char in name):
            raise ValueError("Name must be plain text")
        name = name.strip()
        if not 1 <= len(name) <= 80:
            raise ValueError("Name must contain 1–80 characters")
        return name

    def rank(self, embedding: np.ndarray) -> list[tuple[float, str]]:
        vector = unit_vector(embedding)
        scores = []
        for person_id, person in self.people.items():
            # Require agreement with several enrollment views, rather than one lucky match.
            similarities = sorted((float(vector @ sample) for sample in person["embeddings"]), reverse=True)
            scores.append((float(np.mean(similarities[:3])), person_id))
        return sorted(scores, reverse=True)

    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        ranked = self.rank(embedding)
        if not ranked:
            return None, 0.0
        best, person_id = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else -1.0
        if best < MATCH_THRESHOLD or best - runner_up < MATCH_MARGIN:
            return None, best
        return person_id, best

    def enroll(self, name: str, embeddings: list[np.ndarray]) -> str:
        name = self.validate_name(name)
        vectors = [unit_vector(sample) for sample in embeddings]
        if len(vectors) < 8:
            raise ValueError("At least eight clear face samples are required")
        center = unit_vector(np.mean(vectors, axis=0))
        person_id, _ = self.match(center)
        if person_id is not None:
            if self.people[person_id]["name"].casefold() != name.casefold():
                raise ValueError("This face already has a different enrolled name; it was not overwritten")
            return person_id
        ranked = self.rank(center)
        if ranked and ranked[0][0] >= MATCH_THRESHOLD:
            raise ValueError("Face matches are ambiguous; enrollment would risk mixing identities")
        person_id = str(uuid.uuid4())
        person = {"name": name, "embeddings": vectors[-12:]}
        updated = {**self.people, person_id: person}
        content = {"schema_version": 1, "model": MODEL_ID,
                   "people": [{"person_id": key, "name": value["name"],
                               "embeddings": [sample.tolist() for sample in value["embeddings"]]}
                              for key, value in updated.items()]}
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w") as output:
                json.dump(content, output)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self.people = updated
        return person_id


class PersonRecognition:
    """Return known identities only after sustained, unambiguous fresh matches."""

    def __init__(self, store: IdentityStore, log_path: Path | None = None) -> None:
        self.store = store
        self.log_path = log_path
        self.lock = threading.RLock()
        self.samples = deque(maxlen=20)
        self.candidate = None
        self.streak = 0
        self.streak_at = 0.0
        self.match_times = deque(maxlen=MIN_STABLE_SAMPLES)
        self.last_seen = float("-inf")
        self.face_usable = False
        self.result = {"state": "unknown", "reason": "no_face"}
        self.pending = None
        self.rejected_ids = set()
        self.last_diagnostic_at = float("-inf")
        self.scene_count = 0
        self.scene_at = float("-inf")

    def report_scene(self, face_count: int) -> None:
        """Expose current group visibility independently of embedding cadence."""
        with self.lock:
            self.scene_count, self.scene_at = face_count, monotonic()
            if face_count > 1:
                self.invalidate("multiple_faces")

    def set_result(self, result: dict) -> None:
        now = monotonic()
        changed = (self.result.get("state"), self.result.get("person_id"), self.result.get("reason")) != (
            result.get("state"), result.get("person_id"), result.get("reason"))
        if self.log_path is not None and (changed or now - self.last_diagnostic_at >= 1):
            # Do not put names or biometric vectors in recognition diagnostic events.
            with self.log_path.open("a") as output:
                output.write(json.dumps({"monotonic_s": monotonic(), **{key: value for key, value in result.items() if key != "name"}}) + "\n")
            self.last_diagnostic_at = now
        self.result = result

    def invalidate(self, reason: str, transient: bool = False) -> None:
        with self.lock:
            self.face_usable = False
            # A missed detector frame is unknown now, but need not erase fresh prior evidence.
            if not transient or monotonic() - self.last_seen > 1.5:
                self.samples.clear()
                self.match_times.clear()
                self.candidate, self.streak = None, 0
            self.set_result({"state": "unknown", "reason": reason})

    def observe(self, embedding: np.ndarray, now: float | None = None) -> None:
        now = monotonic() if now is None else now
        vector = unit_vector(embedding)
        with self.lock:
            if self.samples and (now - self.last_seen > 1.5 or float(vector @ self.samples[-1][1]) < .60):
                self.samples.clear()
                self.match_times.clear()
                self.candidate, self.streak = None, 0
            self.last_seen = now
            self.scene_count, self.scene_at = 1, now
            self.face_usable = True
            self.samples.append((now, vector))
            candidate, score = self.store.match(vector)
            ranked = self.store.rank(vector)
            diagnostics = {"cosine_similarity": round(score, 3),
                           "runner_up_margin": round(ranked[0][0] - ranked[1][0], 3) if len(ranked) > 1 else None}
            if candidate is None or candidate in self.rejected_ids:
                self.candidate, self.streak = None, 0
                self.match_times.clear()
                self.set_result({"state": "unknown", "reason": "identity_rejected" if candidate in self.rejected_ids else "no_confident_match",
                                 **diagnostics})
                return
            if candidate != self.candidate:
                self.candidate, self.streak, self.streak_at = candidate, 0, now
                self.match_times.clear()
            while self.match_times and now - self.match_times[0] > 4:
                self.match_times.popleft()
            self.match_times.append(now)
            self.streak = len(self.match_times)
            self.streak_at = self.match_times[0]
            if self.streak < MIN_STABLE_SAMPLES or now - self.streak_at < 1.5:
                self.set_result({"state": "unknown", "reason": "confirming_face_match",
                                 "matching_samples": self.streak, **diagnostics})
                return
            self.set_result({"state": "known", "person_id": candidate,
                             "name": self.store.people[candidate]["name"],
                             **diagnostics})

    def snapshot(self) -> dict:
        with self.lock:
            if monotonic() - self.last_seen > 1.5:
                self.invalidate("no_fresh_face")
            result = dict(self.result)
            result["face_count"] = self.scene_count if monotonic() - self.scene_at <= 1.5 else 0
            return result

    def reject_current(self) -> dict:
        """Honor an identity correction without rewriting stored biometric records."""
        with self.lock:
            if self.result.get("person_id") is not None:
                self.rejected_ids.add(self.result["person_id"])
            self.invalidate("identity_rejected")
            return dict(self.result)

    def enrollment(self, name: str, confirmed: bool, transcript_seq: int, transcript: str) -> dict:
        """Prepare a pairing, then require a later explicit spoken confirmation."""
        with self.lock:
            name = self.store.validate_name(name)
            now = monotonic()
            samples = [sample for timestamp, sample in self.samples if now - timestamp < 8]
            if not self.face_usable or now - self.last_seen > 1.5 or len(samples) < 8:
                return {"status": "need_clear_face", "instruction": "Ask the person to face the camera alone in good light for four seconds, then retry."}
            center = unit_vector(np.mean(samples, axis=0))
            if any(float(center @ sample) < .65 for sample in samples):
                return {"status": "need_clear_face", "instruction": "Face samples are inconsistent; ask the person to hold still and retry."}
            pending = self.pending
            if not confirmed or pending is None or pending["name"] != name or now - pending["at"] > 60:
                self.pending = {"name": name, "embedding": center, "seq": transcript_seq, "at": now}
                return {"status": "confirmation_required", "name": name,
                        "instruction": "Ask: Should I remember your face and name as " + name + "? Wait for a new explicit yes before confirming."}
            affirmative = re.match(r"^(yes|yeah|yep|correct|that['’]?s right|please do)\b", transcript.strip(), re.I)
            negative = re.search(r"\b(no|not|wrong|instead|don['’]?t)\b", transcript, re.I)
            if transcript_seq <= pending["seq"] or not affirmative or negative:
                return {"status": "confirmation_required", "instruction": "No later explicit affirmative answer was detected. Ask for confirmation; nothing was saved."}
            if float(center @ pending["embedding"]) < .65:
                self.pending = None
                return {"status": "unknown", "instruction": "The visible face changed. Restart enrollment; nothing was saved."}
            person_id = self.store.enroll(name, samples)
            self.pending = None
            self.candidate, self.streak, self.streak_at = None, 0, now
            self.match_times.clear()
            return {"status": "enrolled", "person_id": person_id, "name": name,
                    "instruction": "Confirm briefly that you will recognize their name next time. No other facts were saved."}
