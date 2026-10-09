from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ai_quality.clients.rag_client import RAGClient


pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "output_text",
    ["", "   ", "\n\t"],
    ids=["empty", "spaces", "whitespace"],
)
def test_rag_rejects_empty_answer(output_text):
    # Constructor is bypassed: no API key or .env is needed.
    client = RAGClient.__new__(RAGClient)
    client.model = "test-model"

    documents = [{
        "id": "delivery-policy",
        "content": "Standard delivery takes 3 business days.",
    }]

    client.retriever = SimpleNamespace(
        retrieve=Mock(return_value=documents),
    )

    create = Mock(
        return_value=SimpleNamespace(output_text=output_text),
    )
    client.client = SimpleNamespace(
        responses=SimpleNamespace(create=create),
    )

    question = "How long does delivery take?"

    with pytest.raises(ValueError, match="Model returned an empty answer"):
        client.get_answer(question)

    client.retriever.retrieve.assert_called_once_with(question)
    create.assert_called_once()


def test_rag_propagates_timeout():
    client = RAGClient.__new__(RAGClient)
    client.model = "test-model"

    client.retriever = SimpleNamespace(
        retrieve=Mock(return_value=[]),
    )

    timeout = TimeoutError("Simulated API timeout")
    create = Mock(side_effect=timeout)

    client.client = SimpleNamespace(
        responses=SimpleNamespace(create=create),
    )

    with pytest.raises(TimeoutError) as caught:
        client.get_answer("How long does delivery take?")

    assert caught.value is timeout
    create.assert_called_once()
