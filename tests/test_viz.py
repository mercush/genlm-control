import pytest
import json
import socket
import tempfile
from pathlib import Path
import requests
import time
from genlm_control.viz import SMCVisualizer


def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@pytest.fixture
def cleanup_server():
    """Fixture to ensure server is shut down after each test."""
    yield
    SMCVisualizer.shutdown_server()


def test_server_starts_on_default_port(cleanup_server):
    """Test that server starts on the default port (8000)."""
    assert not is_port_in_use(8000)
    port = SMCVisualizer.ensure_server_running()
    assert port == 8000
    assert is_port_in_use(8000)


def test_server_changes_port(cleanup_server):
    """Test that server can change ports."""
    # Start on default port
    SMCVisualizer.ensure_server_running()
    assert is_port_in_use(8000)

    # Change to new port
    SMCVisualizer.set_port(8001)
    port = SMCVisualizer.ensure_server_running()
    assert port == 8001
    assert is_port_in_use(8001)
    assert not is_port_in_use(8000)


def test_visualization_with_json(cleanup_server):
    """Test visualization with a JSON file."""
    # Create a temporary JSON file
    test_data = {"step": 0, "particles": []}
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(test_data, f)
        json_path = Path(f.name)

    try:
        # Visualize the JSON
        url = SMCVisualizer.visualize_smc(json_path, auto_open=False, cleanup=True)

        # Check that server is running and URL is correct
        assert url.startswith("http://localhost:8000/smc.html")

        # Check that we can access the visualization
        response = requests.get(url)
        assert response.status_code == 200

        # Check that JSON file was copied to html directory
        html_dir = Path(__file__).parent.parent / "genlm_control" / "html"
        copied_json = html_dir / json_path.name
        assert copied_json.exists()

    finally:
        # Cleanup
        json_path.unlink()


def test_server_cleanup(cleanup_server):
    """Test that server cleanup works correctly."""
    # Start server
    SMCVisualizer.ensure_server_running()
    assert is_port_in_use(8000)

    # Shut down server
    SMCVisualizer.shutdown_server()

    # Give the server a moment to fully shut down
    time.sleep(0.5)

    # Verify server is shut down
    assert not is_port_in_use(8000)
    assert SMCVisualizer._server is None
    assert SMCVisualizer._server_thread is None


def test_multiple_visualizations(cleanup_server):
    """Test that multiple visualizations work correctly."""
    # Create two test JSON files
    test_data = {
        "step": 0,
        "model": "init",
        "particles": [{"contents": "a", "logweight": "0", "weight_incr": "0"}],
    }
    json_paths = []
    for i in range(2):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(test_data, f)
            json_paths.append(Path(f.name))

    try:
        # Visualize both files
        for json_path in json_paths:
            url = SMCVisualizer.visualize_smc(json_path, auto_open=False, cleanup=True)
            response = requests.get(url)
            assert response.status_code == 200

        # Check that both files were copied to html directory
        html_dir = Path(__file__).parent.parent / "genlm_control" / "html"
        for json_path in json_paths:
            copied_json = html_dir / json_path.name
            assert copied_json.exists()

    finally:
        # Cleanup
        for path in json_paths:
            path.unlink()
