from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent

SUPPORT_RULES = (
    "You are a customer support assistant. "
    "Unused items can be returned within 30 days. "
    "Standard delivery takes 3 business days. "
    "Answer briefly in English using only these facts. "
    "If the answer is unavailable, say: I do not know."
)


class AIClient:
    def __init__(self, mode="demo"):
        if mode not in ("demo", "live"):
            raise ValueError("Mode must be demo or live.")

        self.mode = mode

        if mode == "live":
            settings = dotenv_values(
                PROJECT_ROOT / ".env",
                encoding="utf-8-sig",
            )

            api_key = settings.get("OPENAI_API_KEY")
            self.model = settings.get("OPENAI_MODEL")

            if not api_key or not self.model:
                raise ValueError("API key or model is missing in .env.")

            self.client = OpenAI(
                api_key=api_key,
                timeout=30,
                max_retries=0,
            )

    def get_answer(self, question):
        if self.mode == "live":
            response = self.client.responses.create(
                model=self.model,
                instructions=SUPPORT_RULES,
                input=question,
                max_output_tokens=150,
                store=False,
            )

            answer = response.output_text.strip()

            if not answer:
                raise ValueError("Model returned an empty answer.")

            return answer

        normalized_question = question.lower()

        if "return" in normalized_question:
            return "Unused items can be returned within 30 days."

        if "delivery" in normalized_question:
            return "Standard delivery takes 3 business days."

        return "I do not know based on the available information."


def get_answer(question):
    return AIClient(mode="demo").get_answer(question)
