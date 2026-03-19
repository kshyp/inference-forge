"""Profile Executor Agent (Agent 2).

Collects vLLM debug logs and nsys profiles.
"""

from .agent import ProfilerAgent

__all__ = [
    "ProfilerAgent",
]
