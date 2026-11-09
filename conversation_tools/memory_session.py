"""Bind one recognized, explicitly confirmed person to private memory."""

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


class MemorySession(Tool):
    name = "memory_session"
    description = ("Open personal memory only for one currently recognized person. First call confirmed=false, "
                   "say its returned question including Hi and the recognized name, then wait for a new explicit yes before "
                   "calling confirmed=true. The runtime selects the person ID. Group or unknown scenes cannot bind.")
    parameters_schema = {"type": "object", "properties": {"confirmed": {"type": "boolean"}}, "required": ["confirmed"]}

    async def __call__(self, deps: ToolDependencies, **kwargs) -> dict:
        return await deps.reachy_mini.media.memory_session(kwargs["confirmed"])
