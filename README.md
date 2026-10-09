# AI Quality Framework

Python test framework for evaluating customer-support LLM answers and a small Retrieval-Augmented Generation (RAG) application.

## Test coverage

| Area | What is checked |
| --- | --- |
| Retrieval | Return, delivery, unknown-topic and multiple-document selection |
| Answer quality | Semantic agreement with a reference using an LLM judge |
| Judge regression | Correct refusals, incorrect permissions and incorrect policy limits |
| Missing information | No invented warranty duration when no source is available |
| Context sensitivity | Answers follow a supplied 30-day or 60-day policy |
| Prompt injection | A retrieved document cannot override the support instructions in the tested attack |

The retriever uses keyword matching over JSON documents. It does not use embeddings or a vector database. Live answers and judge decisions can vary; passing examples are not a guarantee of general correctness or security.

## Technology

Python, pytest, OpenAI Responses API, python-dotenv, structured JSON judge output and pytest-html.

## Setup (Windows PowerShell)

Tested locally by the project author with Python 3.12.

```powershell
git clone https://github.com/mnkse/ai-quality-framework.git
cd ai-quality-framework
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit the local `.env` file:

```dotenv
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=your-supported-model
JUDGE_MODEL=your-supported-judge-model
```

Choose models available to your API account. The judge requires structured JSON-schema output. The clients read these settings from the project-root `.env` file. Keep the API key private; `.env` is excluded from Git.

## Offline retrieval tests

No API key or API calls are needed:

```powershell
python -m pytest tests/test_retriever.py -v
```

This file contains four retrieval cases. GitHub Actions runs these tests on pushes and pull requests and uploads an HTML report.

## Live RAG tests and report

```powershell
New-Item -ItemType Directory -Path .\reports -Force
python -m pytest tests/test_retriever.py tests/test_rag_quality.py tests/test_rag_context.py tests/test_rag_security.py -v --capture=tee-sys --tb=short --html=reports/rag-quality.html --self-contained-html
Start-Process .\reports\rag-quality.html
```

This selection contains nine cases. A normal complete run makes ten API calls: five RAG answers and five judge evaluations. RAG and judge tests call the API independently of `AI_TEST_MODE`. API usage is billed separately.

## Other live tests

```powershell
$env:AI_TEST_MODE = "live"
python -m pytest tests/test_llm_quality.py -v -s
python -m pytest tests/test_evaluator.py tests/test_judge_policy.py tests/test_judge_regression.py -v -s
```

`AI_TEST_MODE` selects demo/live behavior only for the `ai_client` fixture. Its demo mode uses fixed responses and does not establish live model quality. The business-rule refusal test is intended for live mode; the simple demo return response does not explicitly reject a 90-day request.

`test_evaluation_basics.py` is a teaching example: an incorrect answer can still pass a keyword check. It is not evidence that the answer is correct.

Running the entire test suite includes live judge/RAG calls; it is not an offline command.

## Project structure

- `ai_client.py`: demo and live support-answer client
- `retriever.py`: keyword-based document selection
- `rag_client.py`: answer generation from retrieved documents
- `evaluator.py`: semantic answer evaluation with a boolean decision and reason
- `conftest.py`: shared AI-client fixture
- `knowledge_base/documents.json`: return and delivery policies
- `test_data/`: parameterized input and evaluation datasets
- `tests/`: retrieval, LLM, judge, RAG and injection tests
- `.github/workflows/offline-tests.yml`: offline retrieval CI
- `reports/`: generated local HTML reports (not committed)

## Validation

The author reported all nine selected retrieval/RAG cases passing locally. The offline GitHub Actions report records each CI run. Live API tests are intentionally excluded from automatic CI.

Known limits include exact keyword retrieval, a small synthetic policy dataset, nondeterministic LLM evaluation and one document-injection example. The judge regression tests help detect known evaluation mistakes but do not eliminate judge errors.
