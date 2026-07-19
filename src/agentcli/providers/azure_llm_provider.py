import os
from typing import Any, Iterator

from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai.types.chat import (
    ChatCompletionToolParam,
    ChatCompletionMessageParam,
    ChatCompletion,
)


class AzureLLMProvider:
    def __init__(self):
        self._endpoint = os.getenv("AZURE_ENDPOINT") or ""
        self._model = os.getenv("AZURE_DEPLOYMENT") or ""
        self._tokens_max = int(os.getenv("AZURE_TOKENS_MAX") or 1024)
        self._client = OpenAI(
            base_url=self._endpoint,
            api_key=get_bearer_token_provider(
                DefaultAzureCredential(), "https://ai.azure.com/.default"
            ),
        )

    def ask_for_tools(
        self,
        messages: list[Any],
        tools: list[Any],
    ) -> ChatCompletion:
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=False,
            tools=tools,
            max_tokens=self._tokens_max,
        )

    def ask(self, prompt: str) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=self._tokens_max,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
