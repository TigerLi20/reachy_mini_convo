"""Person-scoped SQLite dialogue, facts, summaries, and searchable episodes."""

import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def fact_key(text: str) -> str:
    key = "_".join(normalized(text).split())
    return {"favourite_food": "favorite_food", "favorite_meal": "favorite_food",
            "dogs_name": "dog_name", "dog_s_name": "dog_name"}.get(key, key)


class MemoryStore:
    """Keep authoritative memories local; filter by person before ranking anything."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = path
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        os.chmod(path, 0o600)
        self.db.executescript("""
            PRAGMA foreign_keys=ON;
            PRAGMA secure_delete=ON;
            CREATE TABLE IF NOT EXISTS people(person_id TEXT PRIMARY KEY, revision INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, person_id TEXT NOT NULL REFERENCES people,
                started REAL NOT NULL, ended REAL);
            CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY, person_id TEXT NOT NULL REFERENCES people,
                session_id TEXT NOT NULL REFERENCES sessions, role TEXT NOT NULL, text TEXT NOT NULL, created REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS turn_owner ON turns(person_id,session_id,id);
            CREATE TABLE IF NOT EXISTS facts(id TEXT PRIMARY KEY, person_id TEXT NOT NULL REFERENCES people,
                key TEXT NOT NULL, value TEXT NOT NULL, source_turn INTEGER NOT NULL,
                quote TEXT NOT NULL, updated REAL NOT NULL, embedding BLOB, UNIQUE(person_id,key));
            CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY, person_id TEXT NOT NULL REFERENCES people,
                session_id TEXT NOT NULL, text TEXT NOT NULL, sources TEXT NOT NULL, created REAL NOT NULL, embedding BLOB);
            CREATE TABLE IF NOT EXISTS summaries(person_id TEXT NOT NULL, session_id TEXT NOT NULL,
                text TEXT NOT NULL, through_turn INTEGER NOT NULL, revision INTEGER NOT NULL,
                PRIMARY KEY(person_id,session_id));
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_search USING fts5(person_id UNINDEXED,kind UNINDEXED,
                record_id UNINDEXED,text);
        """)
        self.db.commit()

    def close(self) -> None:
        with self.lock:
            self.db.close()

    def start(self, person_id: str) -> str:
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO people(person_id) VALUES(?)", (person_id,))
            session = str(uuid.uuid4())
            self.db.execute("INSERT INTO sessions VALUES(?,?,?,NULL)", (session, person_id, time.time()))
            return session

    def end(self, person_id: str, session_id: str) -> None:
        with self.lock, self.db:
            self.db.execute("UPDATE sessions SET ended=? WHERE id=? AND person_id=?", (time.time(), session_id, person_id))

    def append(self, person_id: str, session_id: str, role: str, text: str) -> int:
        if role not in ("user", "assistant") or not text.strip():
            raise ValueError("Invalid dialogue turn")
        with self.lock, self.db:
            owner = self.db.execute("SELECT person_id FROM sessions WHERE id=?", (session_id,)).fetchone()
            if owner is None or owner[0] != person_id:
                raise ValueError("Session owner mismatch")
            result = self.db.execute("INSERT INTO turns(person_id,session_id,role,text,created) VALUES(?,?,?,?,?)",
                                     (person_id, session_id, role, text[:12000], time.time()))
            return result.lastrowid

    def turns(self, person_id: str, session_id: str | None = None, limit: int = 16) -> list[dict]:
        with self.lock:
            sql = "SELECT id,role,text,session_id,created FROM turns WHERE person_id=?"
            params = [person_id]
            if session_id is not None:
                sql += " AND session_id=?"
                params.append(session_id)
            rows = self.db.execute(sql + " ORDER BY id DESC LIMIT ?", (*params, limit)).fetchall()
            return [dict(row) for row in reversed(rows)]

    def revision(self, person_id: str) -> int:
        with self.lock:
            row = self.db.execute("SELECT revision FROM people WHERE person_id=?", (person_id,)).fetchone()
            return row[0] if row else 0

    def _index(self, person_id, kind, record_id, text):
        self.db.execute("DELETE FROM memory_search WHERE person_id=? AND record_id=?", (person_id, record_id))
        self.db.execute("INSERT INTO memory_search VALUES(?,?,?,?)", (person_id, kind, record_id, text))

    def remember(self, person_id: str, session_id: str, source_turn: int, key: str, value: str,
                 quote: str, embedding=None, correction=False) -> dict:
        key, value = fact_key(key), value.strip()
        if not key or len(key) > 100 or not value or len(value) > 500:
            raise ValueError("Fact requires a short key and value")
        with self.lock, self.db:
            source = self.db.execute("SELECT text FROM turns WHERE id=? AND person_id=? AND session_id=? AND role='user'",
                                     (source_turn, person_id, session_id)).fetchone()
            if source is None or not normalized(quote) or normalized(quote) not in normalized(source[0]):
                raise ValueError("Quote must come from this person's current user turn")
            if normalized(value) not in normalized(quote):
                raise ValueError("The stored value must appear in the quoted statement; clarify inferred facts")
            if source[0].rstrip().endswith('?') and not re.search(r"\b(remember|save|note)\b", source[0], re.I):
                raise ValueError("A question is not a new personal fact; use recall or ask clarification")
            if re.match(r"^(if|suppose|hypothetically|pretend|imagine)\b", normalized(source[0])):
                raise ValueError("Hypothetical dialogue is not a personal fact")
            if not correction and not re.search(r"\b(i|my|me|mine|our|we)\b", normalized(quote)):
                raise ValueError("Ask for a clear first-person statement rather than saving another person's fact")
            if key == "dog_name" and not re.search(r"\b(my|our) dog\b", normalized(quote)):
                raise ValueError("Clarify that this is the current person's dog, not someone else's")
            previous = self.db.execute("SELECT * FROM facts WHERE person_id=? AND key=?", (person_id, key)).fetchone()
            if correction and previous is None:
                raise ValueError("No matching fact to correct; review your remembered facts first")
            record_id = previous["id"] if previous else str(uuid.uuid4())
            if previous and normalized(previous["value"]) == normalized(value):
                return {"status": "unchanged", "id": record_id, "key": key, "value": previous["value"]}
            if previous and normalized(previous["value"]) != normalized(value):
                self._redact(person_id, [previous["value"]])
                quote = re.sub(re.escape(previous["value"]), "[removed]", quote, flags=re.I)
            blob = np.asarray(embedding, dtype=np.float32).tobytes() if embedding is not None else None
            self.db.execute("""INSERT INTO facts VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(person_id,key) DO UPDATE SET value=excluded.value,source_turn=excluded.source_turn,
                quote=excluded.quote,updated=excluded.updated,embedding=excluded.embedding""",
                (record_id, person_id, key, value, source_turn, quote, time.time(), blob))
            self._index(person_id, "fact", record_id, f"{key}: {value}")
            return {"status": "corrected" if previous else "saved", "id": record_id, "key": key, "value": value}

    def review(self, person_id: str) -> list[dict]:
        with self.lock:
            return [dict(row) for row in self.db.execute(
                "SELECT id,key,value,source_turn,quote,updated FROM facts WHERE person_id=? ORDER BY updated DESC", (person_id,))]

    def _redact(self, person_id: str, values: list[str]) -> None:
        # Erase derived copies and bump the revision so in-flight jobs cannot resurrect them.
        self.db.execute("UPDATE people SET revision=revision+1 WHERE person_id=?", (person_id,))
        self.db.execute("DELETE FROM summaries WHERE person_id=?", (person_id,))
        self.db.execute("DELETE FROM episodes WHERE person_id=?", (person_id,))
        self.db.execute("DELETE FROM memory_search WHERE person_id=? AND kind='episode'", (person_id,))
        for row in self.db.execute("SELECT id,value FROM facts WHERE person_id=?", (person_id,)).fetchall():
            if any(re.search(re.escape(value), row["value"], re.I) for value in values):
                self.db.execute("DELETE FROM facts WHERE person_id=? AND id=?", (person_id, row["id"]))
                self.db.execute("DELETE FROM memory_search WHERE person_id=? AND record_id=?", (person_id, row["id"]))
        for table, column in [("turns", "text"), ("facts", "quote")]:
            for row in self.db.execute(f"SELECT id,{column} FROM {table} WHERE person_id=?", (person_id,)).fetchall():
                text = row[column]
                for value in values:
                    text = re.sub(re.escape(value), "[removed]", text, flags=re.I)
                self.db.execute(f"UPDATE {table} SET {column}=? WHERE id=? AND person_id=?", (text, row["id"], person_id))

    def forget(self, person_id: str, key: str) -> dict:
        with self.lock, self.db:
            row = self.db.execute("SELECT id,value FROM facts WHERE person_id=? AND key=?", (person_id, fact_key(key))).fetchone()
            if row is None:
                return {"status": "not_found", "instruction": "No fact was deleted. Review remembered keys to clarify the target."}
            self.db.execute("DELETE FROM facts WHERE person_id=? AND id=?", (person_id, row["id"]))
            self.db.execute("DELETE FROM memory_search WHERE person_id=? AND record_id=?", (person_id, row["id"]))
            self._redact(person_id, [row["value"]])
            return {"status": "forgotten", "key": fact_key(key)}

    def forget_all(self, person_id: str) -> dict:
        with self.lock, self.db:
            for table in ["facts", "episodes", "summaries", "memory_search", "turns", "sessions"]:
                self.db.execute(f"DELETE FROM {table} WHERE person_id=?", (person_id,))
            self.db.execute("UPDATE people SET revision=revision+1 WHERE person_id=?", (person_id,))
            return {"status": "forgotten_all", "face_enrollment_retained": True}

    def summary_input(self, person_id: str, session_id: str, final=False) -> dict:
        with self.lock:
            rows = self.turns(person_id, session_id, limit=10000)
            older = rows if final else rows[:-16]
            previous = self.db.execute("SELECT * FROM summaries WHERE person_id=? AND session_id=?", (person_id, session_id)).fetchone()
            through = previous["through_turn"] if previous else 0
            return {"person_id": person_id, "session_id": session_id, "revision": self.revision(person_id),
                    "previous": previous["text"] if previous else "", "turns": [r for r in older if r["id"] > through]}

    def save_summary(self, job: dict, text: str, embedding=None) -> bool:
        with self.lock, self.db:
            if not job["turns"] or self.revision(job["person_id"]) != job["revision"]:
                return False
            through = job["turns"][-1]["id"]
            previous = self.db.execute("SELECT through_turn FROM summaries WHERE person_id=? AND session_id=?",
                                       (job["person_id"], job["session_id"])).fetchone()
            if previous and previous[0] >= through:
                return False
            self.db.execute("INSERT OR REPLACE INTO summaries VALUES(?,?,?,?,?)",
                            (job["person_id"], job["session_id"], text[:4000], through, job["revision"]))
            episode_id = f'{job["session_id"]}:{through}'
            self.db.execute("DELETE FROM episodes WHERE id=? AND person_id=?", (episode_id, job["person_id"]))
            blob = np.asarray(embedding, dtype=np.float32).tobytes() if embedding is not None else None
            sources = [r[0] for r in self.db.execute("SELECT id FROM turns WHERE person_id=? AND session_id=? AND id<=? ORDER BY id",
                                                   (job["person_id"], job["session_id"], through))]
            self.db.execute("INSERT INTO episodes VALUES(?,?,?,?,?,?,?)",
                            (episode_id, job["person_id"], job["session_id"], text[:4000],
                             json.dumps(sources), time.time(), blob))
            self._index(job["person_id"], "episode", episode_id, text[:4000])
            return True

    def context(self, person_id: str, session_id: str) -> dict:
        with self.lock:
            summaries = [dict(row) for row in self.db.execute(
                "SELECT text,session_id FROM summaries WHERE person_id=? ORDER BY through_turn DESC LIMIT 2", (person_id,))]
            facts = [{key: value for key, value in fact.items() if key != "quote"} for fact in self.review(person_id)[:10]]
            recent = self.turns(person_id, session_id, 16) or self.turns(person_id, None, 16)
            context = {"facts": facts, "summaries": summaries, "recent_dialogue": recent}
            while len(json.dumps(context)) > 17000:
                if len(recent) > 1:
                    recent.pop(0)
                elif summaries:
                    summaries.pop()
                elif facts:
                    facts.pop()
                else:
                    break
            return context

    def recoverable_sessions(self) -> list[tuple[str, str]]:
        with self.lock:
            return [(row[0], row[1]) for row in self.db.execute("""SELECT t.person_id,t.session_id FROM turns t
                LEFT JOIN summaries s ON s.person_id=t.person_id AND s.session_id=t.session_id
                GROUP BY t.person_id,t.session_id HAVING MAX(t.id)>COALESCE(MAX(s.through_turn),0)""")]

    def episode_details(self, person_id: str, episode_id: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT sources FROM episodes WHERE person_id=? AND id=?", (person_id, episode_id)).fetchone()
            if row is None:
                return None
            sources = json.loads(row[0])
            count = sum(1 for fact in self.review(person_id) if fact["source_turn"] in sources)
            return {"source_messages": len(sources), "derived_facts": count, "sources": sources}

    def forget_episode(self, person_id: str, episode_id: str) -> dict:
        with self.lock, self.db:
            details = self.episode_details(person_id, episode_id)
            if details is None:
                return {"status": "not_found"}
            for source in details["sources"]:
                for row in self.db.execute("SELECT id FROM facts WHERE person_id=? AND source_turn=?", (person_id, source)).fetchall():
                    self.db.execute("DELETE FROM memory_search WHERE person_id=? AND record_id=?", (person_id, row[0]))
                self.db.execute("DELETE FROM facts WHERE person_id=? AND source_turn=?", (person_id, source))
                self.db.execute("DELETE FROM turns WHERE person_id=? AND id=?", (person_id, source))
            self._redact(person_id, [])
            return {"status": "forgotten_episode", "source_messages": details["source_messages"], "derived_facts": details["derived_facts"]}

    def recall(self, person_id: str, query: str, embedding=None, limit=6) -> list[dict]:
        with self.lock:
            entries = []
            for row in self.db.execute("SELECT * FROM facts WHERE person_id=?", (person_id,)):
                entries.append({"kind": "fact", "id": row["id"], "text": f'{row["key"]}: {row["value"]}',
                                "key": row["key"], "source_turn": row["source_turn"], "created": row["updated"],
                                "embedding": row["embedding"]})
            for row in self.db.execute("SELECT * FROM episodes WHERE person_id=?", (person_id,)):
                entries.append({"kind": "episode", "id": row["id"], "text": row["text"],
                                "source_turns": json.loads(row["sources"]), "created": row["created"], "embedding": row["embedding"]})
            words = normalized(query).split()
            keyword_ids = set()
            if words:
                expression = " OR ".join('"' + word + '"' for word in words[:20])
                keyword_ids = {row[0] for row in self.db.execute(
                    "SELECT record_id FROM memory_search WHERE memory_search MATCH ? AND person_id=? LIMIT 20",
                    (expression, person_id))}
            ranked = []
            for entry in entries:
                entry["date_utc"] = datetime.fromtimestamp(entry["created"], timezone.utc).isoformat()
                raw = entry.pop("embedding")
                similarity = 0.0
                if embedding is not None and raw is not None:
                    vector = np.frombuffer(raw, dtype=np.float32)
                    if vector.shape == np.asarray(embedding).shape:
                        similarity = float(vector @ embedding)
                score = similarity + (.4 if entry["id"] in keyword_ids else 0)
                if score >= .25 or not words:
                    ranked.append((score, entry["created"], entry))
            ranked.sort(key=lambda row: row[:2], reverse=True)
            return [entry for _, _, entry in ranked[:limit]]
