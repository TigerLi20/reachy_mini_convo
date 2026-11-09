"""Person-scoped relational memory operations, without a model-selected owner."""

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


class RelationalMemoryTool(Tool):
    name = "relational_memory"
    description = (
        "Use only after a confirmed memory_session. Automatically remember clear user-stated personal facts; "
        "never save guesses, hypotheticals, assistant statements, or another person's facts. "
        "remember/correct require key, value, and an exact source quote from the current user turn; "
        "the value must occur in the quote. Use review to see stored keys and the actual transcript if needed. "
        "recall uses query to retrieve facts and dated session episodes. forget requires an exact remembered key; "
        "forget_episode takes an episode ID in query from recall, first confirmed=false to explain its "
        "cascade scope, then confirmed=true after a later explicit yes. "
        "forget_all only follows the user's explicit request to delete all conversational memories. "
        "A correction/delete resets live context automatically. Never claim a blocked/failed operation succeeded."
    )
    parameters_schema = {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["review", "recall", "remember", "correct", "forget", "forget_episode", "forget_all"]},
        "key": {"type": "string", "description": "Stable fact key, e.g. dog_name or favorite_food; reuse existing keys"},
        "value": {"type": "string"}, "quote": {"type": "string"}, "query": {"type": "string"},
        "confirmed": {"type": "boolean"}},
        "required": ["action"]}

    async def __call__(self, deps: ToolDependencies, **kwargs) -> dict:
        return await deps.reachy_mini.media.memory_operation(**kwargs)
