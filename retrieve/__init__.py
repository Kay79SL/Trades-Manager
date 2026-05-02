"""
DACARag Phase 5 retrieval orchestration package.

Six retrieval modules + one orchestrator function used by the chatbot UI.
"""

from .orchestrator import answer_query

__all__ = ["answer_query"]
