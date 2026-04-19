"""
Utilities for LangChain tool integration.

Provides decorators and helpers for creating tools that work with iChatBio's response system.
"""

import functools
import inspect
from typing import Callable, TypeVar, ParamSpec

from langchain_core.tools import StructuredTool

P = ParamSpec('P')
R = TypeVar('R')


def context_tool(func: Callable[P, R], return_direct: bool = False) -> StructuredTool:
    """
    Decorator to create a LangChain tool compatible with iChatBio context.
    
    This allows tools to be called by the LangChain agent while maintaining
    access to the iChatBio context through the current_context context variable.
    
    Example:
        @context_tool
        async def my_tool(query: str):
            context = current_context.get()
            async with context.begin_process("Processing...") as process:
                await process.log("Doing work...")
                
    Args:
        func: The async function to wrap as a tool
        return_direct: If True, the tool's return value ends the agent loop
        
    Returns:
        A StructuredTool that can be used with LangChain agents
    """
    return StructuredTool.from_function(
        coroutine=func,
        name=func.__name__,
        description=func.__doc__ or f"Tool: {func.__name__}",
        return_direct=return_direct,
    )