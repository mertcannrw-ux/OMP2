"""Extension lifecycle and context management for omp_sdk."""

from typing import Any, Dict, List, Optional
from .protocol import ComponentSpec, DirectorSpec, HostTransport, StreamTransport, ToolSpec
from .wrappers import (
    ArtifactClient,
    CommandClient,
    Component,
    ConVarWrapper,
    Director,
    DOMClient,
    JobClient,
    SessionClient,
    Tool,
)


class ExtensionContext:
    """Context passed to an extension upon loading."""

    def __init__(self, extension_id: str, transport: HostTransport) -> None:
        self.extension_id = extension_id
        self.transport = transport
        self.convars = ConVarWrapper(transport)
        self.commands = CommandClient(transport)
        self.session = SessionClient(transport)
        self.dom = DOMClient(transport)
        self.jobs = JobClient(transport)
        self.artifacts = ArtifactClient(transport)


class Extension:
    """Base class for omp Python extensions.

    Extensions implement declarative tools, directors, and components, and react
    to lifecycle events. Durable state must be stored in the session DOM via patches,
    NOT in module-level global variables.
    """

    def __init__(self, extension_id: str, transport: Optional[HostTransport] = None) -> None:
        self.extension_id = extension_id
        self.transport: HostTransport = transport or StreamTransport()
        self.context: Optional[ExtensionContext] = None
        self._tools: List[Tool] = []
        self._directors: List[Director] = []
        self._components: List[Component] = []

    def on_load(self, context: ExtensionContext) -> None:
        """Called when extension is loaded by the host."""
        self.context = context

    def on_unload(self) -> None:
        """Called when extension is unloaded by the host."""
        from .protocol import _tool_handles
        self.transport.register_declarations(self.extension_id, [], [], [])
        for name in list(_tool_handles):
            if name.startswith(f"{self.extension_id}/"):
                del _tool_handles[name]
        self.context = None

    def on_reload(self) -> None:
        """Called when reloaded by the host: clear live handles (mirroring the
        Rust registry reset) and re-sync declarations if a context exists."""
        from .protocol import _tool_handles
        for name in list(_tool_handles):
            if name.startswith(f"{self.extension_id}/"):
                del _tool_handles[name]
        if self.context:
            self.sync_declarations()

    def on_patch(self, patch: Dict[str, Any]) -> None:
        """Called when a journal patch is applied to the active session."""
        pass

    def register_tool(self, tool: Tool) -> None:
        self._tools.append(tool)

    def register_director(self, director: Director) -> None:
        self._directors.append(director)

    def register_component(self, component: Component) -> None:
        self._components.append(component)

    def sync_declarations(self) -> None:
        """Registers all declarative items with the host."""
        tool_specs = [t.to_spec() for t in self._tools]
        director_specs = [d.to_spec() for d in self._directors]
        component_specs = [c.to_spec() for c in self._components]
        for spec in tool_specs:
            if "/" in spec.name:
                raise ValueError(
                    f"tool name {spec.name!r} must not contain '/': "
                    "handles are namespaced as 'extension/tool'"
                )
        self.transport.register_declarations(
            extension_id=self.extension_id,
            tools=tool_specs,
            directors=director_specs,
            components=component_specs,
        )
        from .protocol import _tool_handles
        for tool in self._tools:
            _tool_handles[f"{self.extension_id}/{tool.name}"] = (tool, self.context)
