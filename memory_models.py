"""Small local models and background consolidation, separate from the audio loop."""

import logging
import queue
import re
import threading

import numpy as np

logger = logging.getLogger(__name__)


class LocalMemoryModels:
    """Use FLAN-T5 for summaries and MiniLM for text retrieval embeddings."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
        torch.set_num_threads(2)
        self.torch = torch
        self.lock = threading.Lock()
        print("Loading local memory models (first run downloads model weights)...", flush=True)
        summary_revision = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
        embedding_revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
        self.summary_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", revision=summary_revision)
        self.summary_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", revision=summary_revision).eval()
        self.text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", revision=embedding_revision)
        self.text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", revision=embedding_revision).eval()
        print("Local memory models ready.", flush=True)

    def embed(self, text: str) -> np.ndarray:
        with self.lock, self.torch.inference_mode():
            tokens = self.text_tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            output = self.text_model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(output.dtype)
            vector = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
            vector = self.torch.nn.functional.normalize(vector, p=2, dim=1)
            return vector[0].cpu().numpy().astype(np.float32)

    def summarize(self, previous: str, turns: list[dict]) -> str:
        # The model selects excerpts; only validated source text enters the summary.
        # This avoids a small local model inventing personal history while paraphrasing.
        excerpts = previous.splitlines() if previous else []
        statements = [row["text"][:700] for row in turns if row["role"] == "user" and row["text"].strip()]
        for offset in range(0, len(statements), 6):
            chunk = statements[offset:offset + 6]
            prompt = ("Select up to three important user statements about personal details or events. "
                      "Reply only with their index numbers.\n" +
                      "\n".join(f"{i + 1}. {text}" for i, text in enumerate(chunk)) + "\nSelected indexes:")
            with self.lock, self.torch.inference_mode():
                tokens = self.summary_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                output = self.summary_model.generate(**tokens, max_new_tokens=24, num_beams=2, do_sample=False)
                answer = self.summary_tokenizer.decode(output[0], skip_special_tokens=True)
            indices = list(dict.fromkeys(int(i) - 1 for i in re.findall(r"\b\d+\b", answer)
                                         if 1 <= int(i) <= len(chunk)))[:3]
            if not indices:
                indices = list(range(max(0, len(chunk) - 3), len(chunk)))
            if len(chunk) - 1 not in indices:
                indices = indices[:2] + [len(chunk) - 1]
            for i in indices:
                excerpt = "User said: " + chunk[i]
                if excerpt not in excerpts:
                    excerpts.append(excerpt)
            while sum(len(line) + 1 for line in excerpts) > 2400:
                excerpts.pop(0)
        return "\n".join(excerpts) or "The conversation continued without a new user statement."


class MemoryWorker:
    """Consolidate immutable person/session jobs; revisions invalidate deleted content."""

    def __init__(self, store, models) -> None:
        self.store, self.models = store, models
        self.jobs = queue.Queue()
        self.pending = set()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="memory-consolidation", daemon=True)
        self.thread.start()

    def submit(self, person_id: str, session_id: str, final=False) -> None:
        key = (person_id, session_id, final, self.store.revision(person_id))
        with self.lock:
            if key in self.pending:
                return
            self.pending.add(key)
            self.jobs.put(key)

    def busy(self, person_id: str, session_id: str) -> bool:
        with self.lock:
            return any(key[:2] == (person_id, session_id) for key in self.pending)

    def _run(self) -> None:
        while (key := self.jobs.get()) is not None:
            try:
                if self.store.revision(key[0]) != key[3]:
                    continue
                job = self.store.summary_input(*key[:2], final=key[2])
                if job["turns"]:
                    text = self.models.summarize(job["previous"], job["turns"])
                    vector = self.models.embed(text)
                    self.store.save_summary(job, text, vector)
            except Exception:
                # Do not include source text in error logs.
                logger.error("Memory consolidation failed; original scoped dialogue remains saved")
            finally:
                with self.lock:
                    self.pending.discard(key)
                self.jobs.task_done()

    def close(self) -> bool:
        self.jobs.put(None)
        self.thread.join(timeout=15)
        return not self.thread.is_alive()
