import pytest
from retriever import DocumentRetriever


@pytest.fixture
def retriever():
    return DocumentRetriever()


@pytest.mark.parametrize(
    "question, expected_ids",
    [
        pytest.param(
            "Can I return an item after 90 days?",
            ["return-policy"],
            id="return-document",
        ),
        pytest.param(
            "How long does delivery take?",
            ["delivery-policy"],
            id="delivery-document",
        ),
        pytest.param(
            "What is the warranty duration?",
            [],
            id="no-matching-document",
        ),
        pytest.param(
            "Explain return and shipping policies.",
            ["return-policy", "delivery-policy"],
            id="multiple-documents",
        ),
    ],
)
def test_retrieval(retriever, question, expected_ids):
    documents = retriever.retrieve(question)
    actual_ids = [document["id"] for document in documents]

    assert set(actual_ids) == set(expected_ids), (
        f"Expected documents: {expected_ids}\n"
        f"Retrieved documents: {actual_ids}"
    )
