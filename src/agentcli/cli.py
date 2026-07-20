import dataclasses
import inspect
import json
import os
from typing import Any

from dotenv import load_dotenv
import typer
from agentcli.providers.azure_llm_provider import AzureLLMProvider
import agentcli.tools.financial_calculator as tools

LOOP_CAP: int = int(os.getenv("AGENTCLI_LOOP_CAP", 5))


def main():
    provider = AzureLLMProvider()
    schemas = tools.get_calculator_tools()
    messages: Any = []
    WELCOME_MESSAGE = """
        Financial Calculator CLI
        Ask about mortgages, debt-to-income, and affordability. Examples:

        - "What's the monthly payment on a $400,000 loan at 6.5% for 30 years?"
        - "What's my DTI if I make $95k a year with $450 in monthly debts?"
        - "What's my remaining balance after 5 years on a $350,000 loan at 6% for 30 years?"
        - "How much would I save paying an extra $200/month on a $300,000 loan at 6.5% for 30 years?"
        - "How much house can I afford on $95,000/year with $450 in monthly debts at 36% DTI, 6.5% rate, 30-year term?"

        Type a question and press Enter. Ctrl+C to exit.
        """
    print(WELCOME_MESSAGE)

    while True:
        try:
            prompt = input(">>> ")
            messages.append({"role": "user", "content": prompt})
            call_tool(provider, messages, schemas)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def call_tool(provider: AzureLLMProvider, messages: Any, schemas: Any):
    iteration = 0

    while True:
        # check if loop cap is reached
        if iteration >= LOOP_CAP:
            print("Loop cap reached. Exiting.")
            break

        # get initial response from model with tools
        response = provider.ask_for_tools(messages, schemas)
        response_message = response.choices[0].message

        # append the model's response to the messages list
        messages.append(response_message)
        tool_calls = response_message.tool_calls

        # if no tool calls, print and exit
        if not tool_calls:
            print(response_message.content)
            break

        # find and execute tool
        else:
            for tool_call in tool_calls or []:
                if tool_call.type == "function":
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments
                    extracted_tool_args = json.loads(tool_args) if tool_args else {}
                    result = dispatch(tool_name, extracted_tool_args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": json.dumps(result),
                        }
                    )
            iteration += 1


def dispatch(tool_name: str, args: Any) -> Any:
    try:
        match tool_name:
            case "monthly_mortgage_payment":
                inspect.signature(tools.monthly_mortgage_payment).bind(**args)
                return tools.monthly_mortgage_payment(**args)
            case "debt_to_income_ratio":
                inspect.signature(tools.debt_to_income_ratio).bind(**args)
                return tools.debt_to_income_ratio(**args)
            case "amortization_point_in_time":
                inspect.signature(tools.amortization_point_in_time).bind(**args)
                return tools.amortization_point_in_time(**args)
            case "extra_payment_calculator":
                inspect.signature(tools.extra_payment_calculator).bind(**args)
                return dataclasses.asdict(tools.extra_payment_calculator(**args))
            case "affordability_in_reverse":
                inspect.signature(tools.affordability_in_reverse).bind(**args)
                return dataclasses.asdict(tools.affordability_in_reverse(**args))
        return f"Unknown tool: {tool_name}"
    except TypeError as e:
        return f"Argument mismatch for tool '{tool_name}': {e}"


def cli():
    load_dotenv()
    typer.run(main)
