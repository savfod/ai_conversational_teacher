"""Basic tests for the main module."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_conversational_teacher import __version__


def test_version():
    """Test that version is defined."""
    assert __version__ == "0.1.0"
