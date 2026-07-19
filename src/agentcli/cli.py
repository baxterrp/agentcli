from dotenv import load_dotenv
import typer

from agentcli.providers.azure_llm_provider import AzureLLMProvider
import agentcli.tools.financial_calculator as tools


def main():
    provider = AzureLLMProvider()
    for chunk in provider.ask("Write a poem about a cat."):
        print(chunk, end="", flush=True)

    print(tools.monthly_mortgage_payment(320000, 6.5, 30))
    print(tools.debt_to_income_ratio(2500, 7916.67))
    print(tools.amortization_point_in_time(320000, 6.5, 30, 60))
    print(tools.extra_payment_calculator(320000, 6.5, 30, 200))
    print(tools.affordability_in_reverse(95000, 450, 43, 6.5, 30))


def cli():
    load_dotenv()
    typer.run(main)
