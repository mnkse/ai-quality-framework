import os

import pytest


@pytest.fixture
def ai_client():
    from ai_client import AIClient

    mode = os.getenv("AI_TEST_MODE", "demo").strip().lower()
    return AIClient(mode=mode)
