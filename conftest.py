import os

import pytest


@pytest.fixture
def ai_client(request):
    from ai_quality.clients.llm_client import AIClient

    if request.node.get_closest_marker("live"):
        mode = "live"
    elif request.node.get_closest_marker("offline"):
        mode = "demo"
    else:
        mode = os.getenv("AI_TEST_MODE", "demo").strip().lower()

    return AIClient(mode=mode)
