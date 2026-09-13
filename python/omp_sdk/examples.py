"""Extension examples demonstrating correct host-owned state placement.

Durable session behavior must NOT use module-level counters, maps, or sets.
State is stored entirely within the authoritative session DOM via patches,
ensuring rewind, fork, resume, and replication remain consistent.
"""

import uuid
from typing import Any, Dict, Optional
from .extension import Extension, ExtensionContext
from .wrappers import Tool


class TodoCounterTool(Tool):
    """Tool that increments and reports a todo item counter in the session DOM."""

    name = "todo_counter"
    version = "1.0.0"
    description = "Increments and reports todo items stored in the host session DOM"
    parameter_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Title of todo item to add"}
        },
        "required": ["title"],
    }

    def __init__(self, context: ExtensionContext) -> None:
        self.context = context

    @staticmethod
    def _todo_children(snapshot: Dict[str, Any]) -> list:
        """Defensively extract the todo container's children (fresh sessions
        may have no `todo` node yet)."""
        nodes = snapshot.get("nodes", {}) if isinstance(snapshot, dict) else {}
        todo = nodes.get("todo", {}) if isinstance(nodes, dict) else {}
        children = todo.get("children", [])
        return children if isinstance(children, list) else []

    def execute(self, arguments: Dict[str, Any], context: Any) -> Dict[str, Any]:
        title = arguments.get("title", "Untitled")

        snapshot = self.context.session.get_snapshot()
        current_count = len(self._todo_children(snapshot))
        new_count = current_count + 1
        todo_element_id = f"todo_{uuid.uuid4().hex[:12]}"
        # Plain Python values: the wrapper encodes exact TypedValues
        # (String/Integer). Never pre-wrap as {"String": ...} here — that
        # produced the old {"Json": {"String": ...}} double-wrap bug.
        # NOTE: read-then-create is not atomic; two concurrent invocations may
        # compute the same `index`. The host journal orders the patches, so no
        # items are lost — only the human-readable index may collide. Tools
        # needing strict sequencing should use unique ids (as below) instead.
        self.context.dom.create_element(
            parent="todo",
            element_id=todo_element_id,
            kind="todo_item",
            attributes={"title": title, "index": new_count},
            text=title,
        )

        return {
            "status": "success",
            "todo_id": todo_element_id,
            "title": title,
            "total_todos": new_count,
        }


class StatefulTodoExtension(Extension):
    """Example extension demonstrating zero-local-state, DOM-backed durability."""

    def __init__(self, transport: Optional[Any] = None) -> None:
        super().__init__(extension_id="stateful_todo_example", transport=transport)

    def on_load(self, context: ExtensionContext) -> None:
        super().on_load(context)
        # Register tool that reads and writes authoritative DOM state
        tool = TodoCounterTool(context)
        self.register_tool(tool)
        self.sync_declarations()

    def get_current_todos(self) -> int:
        """Branch-aware query: Derives count from the active session snapshot."""
        if not self.context:
            return 0
        snapshot = self.context.session.get_snapshot()
        return len(TodoCounterTool._todo_children(snapshot))
