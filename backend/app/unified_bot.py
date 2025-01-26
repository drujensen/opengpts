from typing import Annotated, List, cast

from langchain.tools import BaseTool
from langchain_core.language_models.base import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END
from langgraph.graph.message import MessageGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from app.message_types import LiberalToolMessage, add_messages_liberal


def get_unified_bot_executor(
    llm: LanguageModelLike,
    system_message: str,
    tools: list[BaseTool],
    interrupt_before_action: bool,
    checkpoint: BaseCheckpointSaver,
):
    async def _get_messages(messages):
        msgs = []
        for m in messages:
            if isinstance(m, LiberalToolMessage):
                _dict = m.model_dump()
                _dict["content"] = str(_dict["content"])
                m_c = ToolMessage(**_dict)
                msgs.append(m_c)
            elif isinstance(m, FunctionMessage):
                msgs.append(HumanMessage(content=str(m.content)))
            else:
                msgs.append(m)

        return [SystemMessage(content=system_message)] + msgs

    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm
    agent = _get_messages | llm_with_tools
    tool_executor = ToolExecutor(tools)

    def should_continue(messages):
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"

    async def call_tool(messages):
        actions: list[ToolInvocation] = []
        last_message = cast(AIMessage, messages[-1])
        for tool_call in last_message.tool_calls:
            actions.append(
                ToolInvocation(
                    tool=tool_call["name"],
                    tool_input=tool_call["args"],
                )
            )
        responses = await tool_executor.abatch(actions)
        tool_messages = [
            LiberalToolMessage(
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
                content=response,
            )
            for tool_call, response in zip(last_message.tool_calls, responses)
        ]
        return tool_messages

    workflow = MessageGraph(Annotated[List[BaseMessage], add_messages_liberal])

    workflow.add_node("agent", agent)
    workflow.add_node("action", call_tool)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )

    workflow.add_edge("action", "agent")

    return workflow.compile(
        checkpointer=checkpoint,
        interrupt_before=["action"] if interrupt_before_action else None,
    )
