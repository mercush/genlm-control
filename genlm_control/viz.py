import webbrowser
import http.server
import socketserver
import threading
import time
import os
import shutil
from pathlib import Path
import atexit


class SMCVisualizer:
    """Handles visualization of SMC inference process."""

    # Unclear whether this is the best way to do this...

    _server = None
    _server_thread = None
    _port = 8000
    _temp_files = set()  # Keep track of files to clean up

    @classmethod
    def set_port(cls, port):
        """
        Set the port for the visualization server.

        Args:
            port (int): The port to use for the visualization server.
        """
        if cls._server is not None:
            cls.shutdown_server()
        cls._port = port

    @classmethod
    def ensure_server_running(cls):
        """Ensures the HTTP server is running to serve visualization files."""
        if cls._server_thread is not None and cls._server_thread.is_alive():
            return cls._port

        # Get the html directory
        html_path = Path(__file__).parent / "html"
        if not html_path.exists():
            raise FileNotFoundError(
                f"Setting up visualization failed. Could not find HTML directory at {html_path}"
            )

        # Start HTTP server
        handler = http.server.SimpleHTTPRequestHandler
        try:
            cls._server = socketserver.TCPServer(
                ("", cls._port), handler, bind_and_activate=False
            )
            cls._server.allow_reuse_address = True
            cls._server.server_bind()
            cls._server.server_activate()
        except OSError as e:
            if e.errno == 98:  # Address already in use
                raise OSError(
                    f"Port {cls._port} is already in use. "
                    "Use SMCVisualizer.set_port() to specify a different port, "
                    "or pass viz_port to the InferenceEngine call."
                ) from None
            raise

        os.chdir(html_path)

        cls._server_thread = threading.Thread(target=cls._server.serve_forever)
        cls._server_thread.daemon = True
        cls._server_thread.start()

        time.sleep(0.5)
        return cls._port

    @classmethod
    def visualize_smc(cls, json_path, auto_open=False, cleanup=False):
        """Visualizes SMC data in a browser.

        Args:
            json_path (str): Path to the JSON file containing SMC data
            auto_open (bool): Whether to automatically open the visualization in a browser
            cleanup (bool): Whether to delete the JSON file after visualization (default is False)

        Returns:
            (str): URL to the visualization
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        port = cls.ensure_server_running()

        html_path = Path(__file__).parent / "html"
        dest_path = html_path / os.path.basename(json_path)
        if json_path != dest_path:
            shutil.copy2(json_path, dest_path)
            if cleanup:  # delete temp file on exit
                cls._temp_files.add(dest_path)
        elif cleanup:
            cls._temp_files.add(json_path)

        url = f"http://localhost:{port}/smc.html?path={json_path.name}"

        if auto_open:
            webbrowser.open(url)

        return url

    @classmethod
    def shutdown_server(cls):
        """Shuts down the visualization server and cleans up temporary files."""
        if cls._server is not None:
            cls._server.shutdown()
            cls._server.server_close()
            cls._server_thread.join()
            cls._server = None
            cls._server_thread = None

        # Clean up any temporary files
        for f in cls._temp_files:
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        cls._temp_files.clear()

        # Reset to default port
        cls._port = 8000


atexit.register(SMCVisualizer.shutdown_server)
