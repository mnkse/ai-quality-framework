import pytest

from ai_quality.retrieval.document_retriever import DocumentRetriever


pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "transformed_question",
    [
        "CAN I RETURN AN UNUSED ITEM AFTER 90 DAYS?",
        "Can I return an unused item after 90 days?!",
        "Can I RETURN an unused item after 90 days?",
    ],
    ids=["uppercase", "punctuation", "mixed-case"],
)
def test_retrieval_preserves_document_selection(transformed_question):
    retriever = DocumentRetriever()

    original_question = "Can I return an unused item after 90 days?"

    original_ids = {
        doc["id"] for doc in retriever.retrieve(original_question)
    }
    transformed_ids = {
        doc["id"] for doc in retriever.retrieve(transformed_question)
    }

    assert original_ids == {"return-policy"}
    assert transformed_ids == original_ids, (
        f"Selection changed after transformation: {transformed_ids}"
    )
