from __future__ import annotations

from .sort_node import SortNode


def create_node() -> SortNode:
    """Entrypoint for the Sort node. Fully self-contained: no external dependencies."""
    return SortNode()
