import pytest

from evaluator import AnswerEvaluator
from rag_client import RAGClient


@pytest.mark.parametrize(
    "return_days",
    [30, 60],
    ids=["policy-30-days", "policy-60-days"],
)
def test_rag_uses_supplied_policy(monkeypatch, return_days):
    document = {
        "id": "test-return-policy",
        "title": "Test Return Policy",
        "content": (
            f"Unused items can be returned within {return_days} days. "
            f"Returns after {return_days} days are not accepted."
        ),
    }

    rag_client = RAGClient()

    monkeypatch.setattr(
        rag_client.retriever,
        "retrieve",
        lambda question: [document],
    )

    question = "What is the return period for an unused item?"
    result = rag_client.get_answer(question)

    assert result["documents"] == [document]

    evaluation = AnswerEvaluator().evaluate(
        question=question,
        answer=result["answer"],
        reference=document["content"],
    )

    print(f"\nSource policy: {return_days} days")
    print(f"RAG answer: {result['answer']}")
    print(f"Reason: {evaluation['reason']}")

    assert evaluation["passed"], evaluation["reason"]
