import os

import pytest
from ai_client import AIClient


@pytest.fixture
def ai_client():
    mode = os.getenv("AI_TEST_MODE", "demo").strip().lower()
    return AIClient(mode=mode)
