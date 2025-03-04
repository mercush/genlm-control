import webbrowser
import http.server
import socketserver
import threading
import time
import os
import shutil
from pathlib import Path


class InferenceVisualizer:
    """Visualization server for inference results.

    This class provides a web-based visualization tool for examining SMC inference runs.
    It is intended to be used in conjunction with the `InferenceEngine` class.

    Example:
        ```python
        from genlm_control.viz import InferenceVisualizer
        # create the visualizer
        viz = InferenceVisualizer()
        # run inference and save the record to a JSON file
        sequences = await engine(
            n_particles=10,
            max_tokens=20,
            json_path="smc_record.json" # save the record to a JSON file
        )
        # visualize the inference run
        viz.visualize("smc_record.json", auto_open=True)
        # clean up visualization server
        viz.shutdown_server()
        ```
    """

    def __init__(self, port=8000):
        """Initialize the visualization server.

        Args:
            port (int): The port to use for the visualization server.

        Raises:
            FileNotFoundError: If the HTML directory cannot be found
            OSError: If the port is already in use
        """
        self._server = None
        self._server_thread = None
        self._port = port
        self._temp_files = set()

        self._html_path = Path(__file__).parent / "html"
        if not self._html_path.exists():
            raise FileNotFoundError(
                f"Setting up visualization failed. Could not find HTML directory at {self._html_path}"
            )

        self._start_server()

    def _start_server(self):
        """Starts the HTTP server.

        Raises:
            OSError: If the port is already in use
        """
        handler = http.server.SimpleHTTPRequestHandler
        try:
            self._server = socketserver.TCPServer(
                ("", self._port), handler, bind_and_activate=False
            )
            self._server.allow_reuse_address = True
            self._server.server_bind()
            self._server.server_activate()
        except OSError as e:
            if e.errno == 98:  # Address already in use
                raise OSError(
                    f"Port {self._port} is already in use. "
                    "Use a different port when creating the visualizer."
                ) from None
            raise

        os.chdir(self._html_path)

        self._server_thread = threading.Thread(target=self._server.serve_forever)
        self._server_thread.daemon = True
        self._server_thread.start()

        time.sleep(0.5)  # Give the server a moment to start

    def set_port(self, port):
        """
        Set the port for the visualization server.

        Args:
            port (int): The port to use for the visualization server.
        """
        if port == self._port:
            return

        self.shutdown_server()
        self._port = port
        self._start_server()

    def visualize(self, json_path, auto_open=False):
        """Visualizes an inference run in a browser.

        Args:
            json_path (str): Path to the JSON file containing the inference run record
            auto_open (bool): Whether to automatically open the visualization in a browser

        Returns:
            str: The URL of the visualization

        Raises:
            FileNotFoundError: If the JSON file cannot be found
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        dest_path = self._html_path / os.path.basename(json_path)
        if json_path.resolve() != dest_path.resolve():
            shutil.copy2(json_path, dest_path)
            self._temp_files.add(dest_path)

        url = f"http://localhost:{self._port}/smc.html?path={json_path.name}"

        if auto_open:
            webbrowser.open(url)

        return url

    def shutdown_server(self):
        """Shuts down the visualization server and cleans up temporary files."""
        if self._server is not None:
            if self._server_thread is not None and self._server_thread.is_alive():
                self._server.shutdown()
                self._server_thread.join()
            self._server.server_close()
            self._server = None
            self._server_thread = None

        # Clean up any temporary files
        for f in self._temp_files:
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        self._temp_files.clear()

    def __del__(self):
        """Ensure server is shut down when object is deleted."""
        self.shutdown_server()
