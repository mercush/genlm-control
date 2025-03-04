import pytest
import json
import socket
import tempfile
from pathlib import Path
import requests
import time
from genlm_control.viz import InferenceVisualizer


def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@pytest.fixture
def viz():
    """Fixture that provides a visualizer and ensures cleanup."""
    visualizer = InferenceVisualizer(port=8000)
    yield visualizer
    visualizer.shutdown_server()


def test_server_starts_on_default_port():
    """Test that server starts on the default port (8000)."""
    assert not is_port_in_use(8000)
    viz = InferenceVisualizer()  # Should start immediately
    try:
        assert is_port_in_use(8000)
    finally:
        viz.shutdown_server()


def test_server_uses_specified_port():
    """Test that server uses the specified port."""
    assert not is_port_in_use(8001)
    viz = InferenceVisualizer(port=8001)
    try:
        assert is_port_in_use(8001)
        assert not is_port_in_use(8000)
    finally:
        viz.shutdown_server()


def test_port_change(viz):
    """Test that server can change ports."""
    assert is_port_in_use(8000)
    viz.set_port(8001)
    assert is_port_in_use(8001)
    assert not is_port_in_use(8000)


def test_visualization_with_json(viz):
    """Test visualization with a JSON file."""
    # Create a temporary JSON file
    test_data = {"step": 0, "particles": []}
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(test_data, f)
        json_path = Path(f.name)

    try:
        # Visualize the JSON
        url = viz.visualize(json_path, auto_open=False)

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


def test_server_cleanup():
    """Test that server cleanup works correctly."""
    viz = InferenceVisualizer()
    assert is_port_in_use(8000)

    viz.shutdown_server()
    time.sleep(0.5)  # Give the server a moment to fully shut down

    # Verify server is shut down
    assert not is_port_in_use(8000)


def test_multiple_visualizations(viz):
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
            url = viz.visualize(json_path, auto_open=False)
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


def test_port_in_use():
    """Test that appropriate error is raised when port is in use."""
    viz1 = InferenceVisualizer(port=8002)
    try:
        with pytest.raises(OSError) as exc_info:
            InferenceVisualizer(
                port=8002
            )  # Try to create second visualizer on same port
        assert "Port 8002 is already in use" in str(exc_info.value)
    finally:
        viz1.shutdown_server()
