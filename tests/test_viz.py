import pytest
import json
import socket
import tempfile
import time
import requests
from pathlib import Path
from genlm_control.viz import InferenceVisualizer


def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@pytest.fixture
def viz():
    """Fixture that provides a visualizer and ensures cleanup."""
    visualizer = InferenceVisualizer()
    yield visualizer
    visualizer.shutdown_server()


def test_server_starts_on_default_port():
    """Test that server starts on the default port (8000)."""
    assert not is_port_in_use(8000)
    viz = InferenceVisualizer()
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


def test_visualization_with_custom_dir():
    """Test visualization with a custom serve directory."""
    test_data = {
        "step": 0,
        "mode": "init",
        "particles": [{"contents": "a", "logweight": "0", "weight_incr": "0"}],
    }
    with tempfile.TemporaryDirectory() as serve_dir:
        viz = InferenceVisualizer(serve_dir=serve_dir)
        try:
            # Create a test JSON file in the serve directory
            json_path = Path(serve_dir) / "test.json"
            with open(json_path, "w") as f:
                json.dump(test_data, f)

            # Should be able to visualize immediately
            response = requests.get(f"http://localhost:8000/{json_path.name}")
            assert response.status_code == 200
            assert response.json() == test_data
        finally:
            viz.shutdown_server()


def test_visualization_with_external_file():
    """Test visualization with a file outside the serve directory."""
    viz = InferenceVisualizer()
    test_data = {
        "step": 0,
        "mode": "init",
        "particles": [{"contents": "a", "logweight": "0", "weight_incr": "0"}],
    }
    try:
        # Create a test JSON file in a different directory
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(test_data, f)
            json_path = Path(f.name)

        # Should copy file and make it available
        viz.visualize(json_path)
        response = requests.get(f"http://localhost:8000/{json_path.name}")
        assert response.status_code == 200
        assert response.json() == test_data

    finally:
        viz.shutdown_server()
        json_path.unlink()


def test_server_cleanup():
    """Test that server cleanup works correctly."""
    viz = InferenceVisualizer()
    temp_dir = viz._serve_dir
    assert temp_dir.exists()

    viz.shutdown_server()
    time.sleep(0.5)  # Give the server a moment to fully shut down

    # Verify server is shut down and temp directory is cleaned up
    assert not is_port_in_use(8000)
    assert not temp_dir.exists()


def test_port_in_use():
    """Test that appropriate error is raised when port is in use."""
    viz1 = InferenceVisualizer(port=8002)
    try:
        with pytest.raises(OSError) as exc_info:
            InferenceVisualizer(port=8002)
        assert "Port 8002 is already in use" in str(exc_info.value)
    finally:
        viz1.shutdown_server()
