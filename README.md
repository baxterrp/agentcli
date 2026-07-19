# agentcli

A typed Python CLI where the model decides when to call functions instead of you calling it directly. Project 2 of 6 in a regulated-document AI platform — the same skeleton as repostat and askdocs, now the model can call functions you expose, use their results, and answer questions it couldn't answer from text alone.

## Usage

```
uv run agentcli "How much house can I afford with a $95,000 salary, $450 in monthly debts, at 43% DTI, 6.5% rate, over 30 years?"
```

```
Final Response: With an annual income of $95,000, $450 in monthly debts, a target debt-to-income
ratio of 43%, an interest rate of 6.5%, and a 30-year term, you can afford a maximum loan amount
of approximately $467,381. Your maximum monthly payment would be around $2,954.
```

The model reads the prompt, decides it needs the `affordability_in_reverse` tool, calls it with arguments it extracted from your sentence, and folds the result back into its answer — no `--tool` flag or manual routing involved.

## Auth

Authentication follows the same pattern as repostat and askdocs — secrets are read from environment variables, never hardcoded. This project is Azure-only, narrowed from askdocs' multi-provider setup to stay aligned with the AI-103 exam scope.

```
# .env
AZURE_ENDPOINT=https://your-foundry-project.services.ai.azure.com/openai/v1
AZURE_DEPLOYMENT=gpt-4o-mini
AZURE_TOKENS_MAX=1024                  # optional, defaults to 1024
```

Azure uses `DefaultAzureCredential` — run `az login` for local development. No raw API keys in Azure.

## Setup

```
uv sync
az login
uv run agentcli --help
```

## Development

```
uv run pytest          # run tests
uv run ruff check src  # lint
uv run pyright src     # type check
```

## Stack

- [openai](https://pypi.org/project/openai/) — OpenAI-compatible client pointed at Azure AI Foundry
- [azure-identity](https://pypi.org/project/azure-identity/) — `DefaultAzureCredential` for keyless auth
- [typer](https://typer.tiangolo.com/) — type-hint-driven CLI
- [python-dotenv](https://pypi.org/project/python-dotenv/) — `.env` file loading
- [pytest](https://pytest.org/) + [pytest-asyncio](https://pypi.org/project/pytest-asyncio/) — testing, sync and async

## What this project adds

- The raw tool-calling handshake: hand-written JSON schemas describing each tool, a request that offers them to the model, and a response that may ask for one instead of answering in text
- Five financial tools the model can call: monthly mortgage payment, debt-to-income ratio, amortization balance at a point in time, extra-payment payoff analysis, and reverse affordability (income → max loan)
- Parsing a tool-call response, executing the matching Python function, and sending the result back to the model for a final, grounded answer
- Azure-only provider — single-provider by choice, unlike askdocs' multi-provider abstraction

## Part of a series

| Project | What it adds |
|---------|-------------|
| repostat | Python language fundamentals: CLI, REST, typed models, error handling, secrets, tests |
| askdocs | LLM SDK, streaming, naive RAG, multi-provider |
| **agentcli** | Tool-calling agents, memory, asyncio, MCP |
| ragservice | FastAPI, embeddings, vector + hybrid search, citations, PII handling |
| extractor | Document intelligence, vision, batch processing, structured validation |
| evalkit | Evals, observability, cost tracking, tracing, Docker |
