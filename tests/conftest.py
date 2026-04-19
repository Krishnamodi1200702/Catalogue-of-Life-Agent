# tests/conftest.py
"""
Pytest configuration - auto-starts agent server before tests.
"""

import pytest
import subprocess
import time
import sys
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def agent_server():
    """Start agent server before tests, stop after."""
    print("\n Starting COL agent server...")
    
    process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "src.langchain_col.agent:create_app",
            "--factory",
            "--host", "127.0.0.1",
            "--port", "9999",
            "--log-level", "error"  # ← Suppress startup logs
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).parent.parent
    )
    
    # Simple wait for server to start
    time.sleep(3)
    
    print(" Server ready\n")
    
    yield
    
    print("\n Stopping server...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()