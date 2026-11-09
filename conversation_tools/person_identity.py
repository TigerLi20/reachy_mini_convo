"""Expose only verified, current identity matches to the conversation."""

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


class PersonIdentity(Tool):
    name = "person_identity"
    description = (
        "Check the current visible person's enrolled identity. Call before the startup greeting "
        "and before using a remembered name. Only state=known permits a remembered name. "
        "An unknown result never licenses guessing a name; ask for clarification instead. "
        "If the user says the recognized name is wrong, pass reject_current=true to suppress "
        "that identity for the rest of this session."
    )
    parameters_schema = {"type": "object", "properties": {
        "reject_current": {"type": "boolean", "description": "True only when the person explicitly corrects the recognized identity"}},
        "required": []}

    async def __call__(self, deps: ToolDependencies, **kwargs) -> dict:
        return await deps.reachy_mini.media.person_identity(kwargs.get("reject_current", False))
