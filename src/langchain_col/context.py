"""
Context management for LangChain tools.

This module provides a way for LangChain tools to access the iChatBio
ResponseContext without passing it explicitly through every function call.
"""

import contextvars
from ichatbio.agent_response import ResponseContext

# Context variable to store the current ResponseContext
# This allows tools to access the context without explicit parameter passing.
# No default is set to ensure fail-fast behavior if context is missing.
current_context: contextvars.ContextVar[ResponseContext] = contextvars.ContextVar(
    "current_context"
)