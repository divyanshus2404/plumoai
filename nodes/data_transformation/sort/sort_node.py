"""
Sort Node (node_id: sort, category: Data Transformation)

Deterministic action node that orders a list of items. For a list of objects
it sorts by a chosen field; for a list of primitives it sorts the values
directly. Ordering is type-aware so mixed types never raise, and items whose
sort value is missing/None are always placed last (in both directions).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from services.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


def _extract_list(previous_output: Any) -> List[Any]:
    """Extract the list to sort from previous_output. "output"/"items" are the
    workflow_executor_service convention (state.outputs[nid] ==
    {"output": ..., "items": [...]}); the rest cover ad-hoc dict shapes when
    this node is invoked outside that engine."""
    if previous_output is None:
        return []
    if isinstance(previous_output, list):
        return previous_output
    if isinstance(previous_output, dict):
        for key in ("output", "items", "result", "data", "rows", "records"):
            v = previous_output.get(key)
            if isinstance(v, list):
                return v
        return [previous_output]
    return [previous_output]


def _sort_value(item: Any, field: str) -> Any:
    """The value an item is ordered by: item[field] when a field is set and the
    item is a mapping, otherwise the item itself."""
    if field and isinstance(item, dict):
        return item.get(field)
    return item


def _type_key(value: Any) -> Tuple[int, Any]:
    """A comparison key that never raises on mixed types. The first element
    groups by type so values of different types are ordered by group rather
    than compared directly (which would raise in Python 3); the second is the
    natural value within that group."""
    if isinstance(value, bool):
        # bool is a subclass of int; treat it as a number (0/1).
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (0, value)
    if isinstance(value, str):
        return (1, value.lower())
    # Anything else (dicts, lists, etc.) — order by string form, after the rest.
    return (2, str(value))


class SortNode(BaseNode):
    NODE_ID = "sort"
    NODE_NAME = "Sort"
    CATEGORY = "Data Transformation"
    DESCRIPTION = "Sort the incoming list of items by a field, ascending or descending."

    def execute(
        self,
        node_outputs: Dict[str, Any],
        previous_output: Any,
        run_state: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        config = config or {}
        items = _extract_list(previous_output)

        field = config.get("field")
        field = str(field).strip() if field is not None else ""

        order = str(config.get("order") or "asc").strip().lower()
        if order not in ("asc", "desc"):
            order = "asc"

        # Partition out items whose sort value is None so they always land last,
        # regardless of direction.
        present = [it for it in items if _sort_value(it, field) is not None]
        missing = [it for it in items if _sort_value(it, field) is None]

        try:
            present.sort(
                key=lambda it: _type_key(_sort_value(it, field)),
                reverse=(order == "desc"),
            )
        except Exception as exc:  # defensive: _type_key is total, but never fail a run
            logger.warning("Sort node fell back to unsorted output: %s", exc)
            return {
                "success": False,
                "result": items,
                "error": f"Could not sort items: {exc}",
            }

        ordered = present + missing

        return {
            "success": True,
            "result": ordered,
            "count": len(ordered),
            "order": order,
            "field": field or None,
            "unsortable": len(missing),
        }


__all__ = ["SortNode"]
