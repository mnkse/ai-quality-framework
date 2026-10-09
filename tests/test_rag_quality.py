from ai_quality.evaluators.llm_judge import AnswerEvaluator
from ai_quality.clients.rag_client import RAGClient


def test_rag_return_policy():
    question = "Can I return an unused item after 90 days?"

    rag_client = RAGClient()
    result = rag_client.get_answer(question)

    documents = result["documents"]
    actual_ids = {document["id"] for document in documents}

    assert actual_ids == {"return-policy"}, (
        f"Wrong documents retrieved: {actual_ids}"
    )

    reference = "\n".join(
        document["content"] for document in documents
    )

    evaluation = AnswerEvaluator().evaluate(
        question=question,
        answer=result["answer"],
        reference=reference,
    )

    print(f"\nRetrieved documents: {sorted(actual_ids)}")
    print(f"RAG answer: {result['answer']}")
    print(f"Judge decision: {evaluation['passed']}")
    print(f"Reason: {evaluation['reason']}")

    assert evaluation["passed"], evaluation["reason"]


def test_rag_unknown_warranty():
    question = "How many years does the warranty last?"

    rag_client = RAGClient()
    result = rag_client.get_answer(question)

    assert result["documents"] == [], (
        "No warranty document should be retrieved."
    )

    evaluation = AnswerEvaluator().evaluate(
        question=question,
        answer=result["answer"],
        reference=(
            "No documents were retrieved. "
            "No warranty duration is available. "
            "The answer must acknowledge that the duration is unknown "
            "and must not invent a warranty period."
        ),
    )

    print(f"\nRetrieved documents: {result['documents']}")
    print(f"RAG answer: {result['answer']}")
    print(f"Judge decision: {evaluation['passed']}")
    print(f"Reason: {evaluation['reason']}")

    assert evaluation["passed"], evaluation["reason"]


import pytest

pytestmark = pytest.mark.live
