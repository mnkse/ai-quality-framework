def test_return_after_deadline(ai_client):
    question = "Can I return an unused item after 90 days?"

    actual = ai_client.get_answer(question)
    normalized = actual.lower()

    print(f"\nQuestion: {question}")
    print(f"Answer: {actual}")

    assert "30 days" in normalized, (
        f"The answer must explain the 30-day limit: {actual}"
    )

    refusal_phrases = [
        "no",
        "cannot",
        "can't",
        "not eligible",
        "not possible",
        "outside",
        "exceeds",
    ]

    words = normalized.replace(",", " ").replace(".", " ").split()

    refuses_return = (
        "no" in words
        or any(phrase in normalized for phrase in refusal_phrases[1:])
    )

    assert refuses_return, (
        f"The answer must reject a return after 90 days: {actual}"
    )
