import re
import ast
import json
import warnings
from pathlib import Path

from pylsp.workspace import Document
from pylsp.python_lsp import PythonLSPServer

from genlm_control.potential.base import Potential


class PythonLSP(Potential):
    """A potential that uses a language server to lint code incrementally."""

    def __init__(self):
        self.server = LSPDiagnosticServer("python")
        super().__init__(list(range(256)))

    def _is_syntactically_complete(self, code: str):
        try:
            ast.parse(code)
            return True
        except (SyntaxError, IndentationError):
            return False

    def _get_diagnostics(self, code: str, verbosity=0):
        diagnostics = self.server.get_diagnostics(code)
        if verbosity > 0:
            print(f"Diagnostics: {diagnostics}")
        errors = [d for d in diagnostics if d["severity"] == 1]
        if errors:
            return float("-inf")
        return 0

    def _preprocess(self, context):
        if isinstance(context, list):
            context = bytes(context)
        try:
            code = context.decode("utf-8")
        except UnicodeDecodeError:
            return
        return code

    def _backoff(self, context):
        linebreak_idx = context.rfind("\n")
        if linebreak_idx == -1:
            return
        return context[:linebreak_idx]

    async def prefix(self, context, verbosity=0):
        context = self._preprocess(context)
        if not context:
            return 0

        context = self._backoff(context)
        if not context:
            return 0

        if not self._is_syntactically_complete(context):
            return 0

        return self._get_diagnostics(context, verbosity)

    async def complete(self, context, verbosity=0):
        context = self._preprocess(context)
        if not context:
            return 0
        return self._get_diagnostics(context, verbosity)

    def __repr__(self):
        return "PythonLSP()"


class DiagnosticCapture:
    def __init__(self):
        self.diagnostics = []

    def write(self, data):
        try:
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            match = re.search(r"\r\n\r\n({.*})", data)
            if match:
                message = json.loads(match.group(1))
                if message.get("method") == "textDocument/publishDiagnostics":
                    self.diagnostics = message.get("params", {}).get("diagnostics", [])
        except Exception as e:
            warnings.warn(f"Failed to parse diagnostic data: {e}")

    def flush(self):
        pass

    def close(self):
        pass

    @property
    def closed(self):
        return False


class LSPDiagnosticServer:
    def __init__(self, language="python"):
        self.language = language
        self.diagnostic_capture = DiagnosticCapture()
        self.server = PythonLSPServer(None, self.diagnostic_capture)
        self.temp_file = Path("/tmp/genlm_control_temp.py")
        self.server.m_initialize(
            {
                "processId": None,
                "rootUri": None,
                "capabilities": {
                    "textDocument": {
                        "synchronization": {
                            "didSave": False,
                            "willSave": False,
                            "willSaveWaitUntil": False,
                        }
                    }
                },
            }
        )
        self.server.m_initialized()

    def get_diagnostics(self, code):
        """Get diagnostics for a piece of code."""
        self.diagnostic_capture.diagnostics = []
        doc = Document(self.temp_file.name, self.server.workspace, code)
        self.server.m_text_document__did_open(
            **{
                "textDocument": {
                    "uri": doc.uri,
                    "languageId": self.language,
                    "version": 1,
                    "text": code,
                }
            }
        )
        self.server._lint_text_document(doc.uri, self.server.workspace, True)
        return self.diagnostic_capture.diagnostics

    def __del__(self):
        if hasattr(self, "server"):
            self.server.m_shutdown()
            self.server.m_exit()
        if self.temp_file.exists():
            self.temp_file.unlink()
