"""Immutable tool provenance inherited by asyncio tasks and worker threads."""

from contextvars import ContextVar
from dataclasses import dataclass

from relational_memory import MemoryScope


@dataclass(frozen=True)
class ToolOrigin:
    epoch: int
    scope: MemoryScope | None
    source_turn: int | None
    transcript_seq: int
    transcript: str
    runtime_id: str


tool_origin = ContextVar("relational_tool_origin", default=None)
