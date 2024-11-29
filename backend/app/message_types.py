from typing import Any, get_args
from langchain_core.messages import (
    FunctionMessage,
    MessageLikeRepresentation,
    ToolMessage,
)
from langgraph.graph.message import Messages, add_messages
from langchain_core.messages import _message_from_dict

class LiberalFunctionMessage(FunctionMessage):
    content: Any


class LiberalToolMessage(ToolMessage):
    content: Any


def _convert_pydantic_dict_to_message(
    data: MessageLikeRepresentation
) -> MessageLikeRepresentation:
    """Convert a dictionary to a message object if it matches message format."""
    if (
        isinstance(data, dict)
        and "content" in data
        and isinstance(data.get("type"), str)
    ):
        _type = data.pop("type")
        return _message_from_dict({"data": data, "type": _type})
    return data


def add_messages_liberal(left: Messages, right: Messages):
    # coerce to list
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return add_messages(
        [_convert_pydantic_dict_to_message(m) for m in left],
        [_convert_pydantic_dict_to_message(m) for m in right],
    )
