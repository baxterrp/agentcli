from dotenv import load_dotenv
import typer

from agentcli.providers.azure_llm_provider import AzureLLMProvider


def main():
    provider = AzureLLMProvider()
    for chunk in provider.ask("Write a poem about a cat."):
        print(chunk, end="", flush=True)


def cli():
    load_dotenv()
    typer.run(main)
