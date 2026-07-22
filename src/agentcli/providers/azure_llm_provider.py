import os
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
)


class AzureLLMProvider:
    def __init__(self):
        self._endpoint = os.getenv("AZURE_ENDPOINT") or ""
        self._model = os.getenv("AZURE_DEPLOYMENT") or ""
        self._tokens_max = int(os.getenv("AZURE_TOKENS_MAX") or 1024)
        self._credential = DefaultAzureCredential()
        self._client = AsyncOpenAI(
            base_url=self._endpoint,
            api_key=get_bearer_token_provider(
                self._credential, "https://ai.azure.com/.default"
            ),
        )

    async def __aenter__(self) -> AzureLLMProvider:
        return self

    async def __aexit__(self, *args) -> None:
        await self._credential.close()

    async def ask(
        self,
        messages: list[Any],
        tools: list[Any],
    ) -> ChatCompletion:
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=False,
            tools=tools,
            max_tokens=self._tokens_max,
        )
