"""
conftest.py – stub out optional/heavy dependencies that are NOT installed in
the CI environment, so test modules can be collected and run without real API
keys or a running MCP server.

Stubs registered before any test module imports:
  • mcp / mcp.server / mcp.server.fastmcp  →  no-op FastMCP
  • groq                                    →  stub Groq client
"""
import sys
import types
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _make_stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# ---------------------------------------------------------------------------
# Stub: mcp.server.fastmcp  →  FastMCP no-op  (mcp is NOT in requirements)
# ---------------------------------------------------------------------------
class _FastMCP:
    def __init__(self, *args, **kwargs):
        pass

    def tool(self):
        """Decorator that leaves the wrapped function unchanged."""
        def decorator(fn):
            return fn
        return decorator

    def run(self):
        pass


if "mcp" not in sys.modules:
    sys.modules["mcp"] = _make_stub("mcp")
if "mcp.server" not in sys.modules:
    sys.modules["mcp.server"] = _make_stub("mcp.server")
if "mcp.server.fastmcp" not in sys.modules:
    sys.modules["mcp.server.fastmcp"] = _make_stub(
        "mcp.server.fastmcp", FastMCP=_FastMCP
    )


# ---------------------------------------------------------------------------
# Stub: groq  →  Groq stub with chainable chat.completions.create
# ---------------------------------------------------------------------------
if "groq" not in sys.modules:
    class _Groq:
        """Minimal Groq client stub."""
        def __init__(self, *args, **kwargs):
            self.chat = MagicMock()

    sys.modules["groq"] = _make_stub("groq", Groq=_Groq)
