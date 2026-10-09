import json

import pytest

from ai_quality.config.settings import PROJECT_ROOT


pytestmark = pytest.mark.offline


@pytest.fixture(scope="module")
def documents():
    path = PROJECT_ROOT / "knowledge_base" / "documents.json"
    data = json.loads(path.read_text(encoding="utf-8-sig"))

    assert isinstance(data, list), "Documents must be a list."
    assert data, "Knowledge base must not be empty."
    assert all(isinstance(doc, dict) for doc in data), (
        "Every document must be an object."
    )

    return data


def test_required_fields(documents):
    required = {"id", "title", "keywords", "content"}

    for index, document in enumerate(documents):
        missing = required - document.keys()
        assert not missing, (
            f"Document {index} is missing fields: {sorted(missing)}"
        )


def test_non_empty_text_fields(documents):
    for index, document in enumerate(documents):
        for field in ("id", "title", "content"):
            value = document.get(field)
            assert isinstance(value, str) and value.strip(), (
                f"Document {index}: {field} must be non-empty text."
            )


def test_unique_document_ids(documents):
    ids = [document.get("id") for document in documents]

    assert all(isinstance(value, str) and value.strip() for value in ids), (
        "Every document must have a valid ID."
    )
    assert len(ids) == len(set(ids)), "Duplicate document IDs found."


def test_valid_keywords(documents):
    for index, document in enumerate(documents):
        keywords = document.get("keywords")

        assert isinstance(keywords, list) and keywords, (
            f"Document {index}: keywords must be a non-empty list."
        )
        assert all(
            isinstance(word, str) and word.strip()
            for word in keywords
        ), f"Document {index}: every keyword must be non-empty text."

        # Current retriever supports single word tokens.
        assert all(word.isalnum() for word in keywords), (
            f"Document {index}: keywords must be single alphanumeric tokens."
        )

        normalized = [word.casefold() for word in keywords]
        assert len(normalized) == len(set(normalized)), (
            f"Document {index}: duplicate keywords found."
        )
