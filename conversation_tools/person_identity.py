"""Expose fresh visual matches or an explicitly confirmed active session."""

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


class PersonIdentity(Tool):
    name = "person_identity"
    description = (
        "Check identity before the startup greeting or when asked to check again. "
        "State=known is a fresh visual match; state=session_confirmed is continuity of an "
        "already confirmed private session, including its 20-second visual-loss grace period. "
        "Both permit the returned name. For session_confirmed, continue naturally and let face "
        "recognition retry silently; do not ask for camera adjustments or repeat memory consent. "
        "An unknown result never licenses guessing a name; ask for clarification instead. "
        "If the user says the recognized name is wrong, pass reject_current=true to suppress "
        "that identity for the rest of this session."
    )
    parameters_schema = {"type": "object", "properties": {
        "reject_current": {"type": "boolean", "description": "True only when the person explicitly corrects the recognized identity"}},
        "required": []}

    async def __call__(self, deps: ToolDependencies, **kwargs) -> dict:
        return await deps.reachy_mini.media.person_identity(kwargs.get("reject_current", False))
