"""Graceful end-of-conversation tool."""

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


class FinishConversation(Tool):
    name = "finish_conversation"
    description = (
        "End the conversation and put Reachy to sleep. Only use when the user clearly "
        "wants to end this conversation. Call once, then speak a short farewell in your next reply. "
        "Waits for that farewell's playback before sleeping. Never use for silence or ambiguous intent."
    )
    needs_response = True
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, deps: ToolDependencies, **kwargs) -> dict:
        return await deps.reachy_mini.media.finish_conversation(deps.go_to_sleep)
