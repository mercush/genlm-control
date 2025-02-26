import sys
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT / "examples"))


@pytest.mark.asyncio
async def test_getting_started_examples():
    """Test that all examples in getting_started.py run without errors."""
    try:
        from getting_started import main

        await main()
    except Exception as e:
        pytest.fail(f"getting_started.py examples failed with error: {str(e)}")
