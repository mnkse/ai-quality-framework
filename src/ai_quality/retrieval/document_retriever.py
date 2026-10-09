from ai_quality.config.settings import PROJECT_ROOT

import json
import re
from pathlib import Path


class DocumentRetriever:
    def __init__(self):
        data_file = (
            PROJECT_ROOT
            / "knowledge_base"
            / "documents.json"
        )

        with data_file.open(encoding="utf-8-sig") as file:
            self.documents = json.load(file)

    def retrieve(self, question):
        words = set(re.findall(r"\b\w+\b", question.casefold()))

        return [
            document
            for document in self.documents
            if words.intersection(
                keyword.casefold()
                for keyword in document["keywords"]
            )
        ]
