"""Pair a stated name with a clear face after explicit confirmation."""

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


class EnrollPerson(Tool):
    name = "enroll_person"
    description = (
        "Enroll only the person's face and stated name for later sessions. First call with "
        "confirmed=false to prepare, then ask them to confirm the exact name and permission "
        "to remember their face. Call with confirmed=true only after a later explicit yes. "
        "A successful status=enrolled is the only evidence anything was saved. "
        "Never enroll guessed names, quoted names, or names of other people."
    )
    parameters_schema = {"type": "object", "properties": {
        "name": {"type": "string", "description": "The exact name the person gave for themselves"},
        "confirmed": {"type": "boolean", "description": "False to prepare; true only after the requested later spoken confirmation"}},
        "required": ["name", "confirmed"]}

    async def __call__(self, deps: ToolDependencies, **kwargs) -> dict:
        return await deps.reachy_mini.media.enroll_person(kwargs["name"], kwargs["confirmed"])
