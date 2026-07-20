import dataclasses
import json
from typing import Any

from dotenv import load_dotenv
import typer
from agentcli.providers.azure_llm_provider import AzureLLMProvider
import agentcli.tools.financial_calculator as tools


def main():
    provider = AzureLLMProvider()
    schemas = tools.get_calculator_tools()
    messages: Any = []

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
    response = provider.ask_for_tools(messages, schemas)
    response_message = response.choices[0].message
    messages.append(response_message)
    tool_calls = response_message.tool_calls

    if tool_calls:
        for tool_call in tool_calls:
            if tool_call.type == "function":
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                extracted_tool_args = json.loads(tool_args) if tool_args else {}

                if tool_name == "monthly_mortgage_payment":
                    result = tools.monthly_mortgage_payment(**extracted_tool_args)
                elif tool_name == "debt_to_income_ratio":
                    result = tools.debt_to_income_ratio(**extracted_tool_args)
                elif tool_name == "amortization_point_in_time":
                    result = tools.amortization_point_in_time(**extracted_tool_args)

                # the below two tools return dataclasses, so we convert them to dicts for JSON serialization
                elif tool_name == "extra_payment_calculator":
                    result = dataclasses.asdict(
                        tools.extra_payment_calculator(**extracted_tool_args)
                    )
                elif tool_name == "affordability_in_reverse":
                    result = dataclasses.asdict(
                        tools.affordability_in_reverse(**extracted_tool_args)
                    )
                else:
                    result = f"Unknown tool: {tool_name}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(result),
                    }
                )

    final_response = provider.ask_for_tools(messages, schemas)
    print(final_response.choices[0].message.content)


def cli():
    load_dotenv()
    typer.run(main)
