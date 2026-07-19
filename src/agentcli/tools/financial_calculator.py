from typing import Iterator
from math import log, ceil
from dataclasses import dataclass
from openai.types.chat import ChatCompletionToolParam


@dataclass
class PayoffResult:
    payoff_months: int
    months_saved: int
    interest_saved: float
    new_total_interest: float


@dataclass
class AffordabilityResult:
    max_loan_amount: float
    max_payment: float


def get_calculator_tools() -> list[ChatCompletionToolParam]:
    get_mortgage_payment_schema: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": "monthly_mortgage_payment",
            "description": "Calculate the fixed monthly payment for a fully amortizing mortgage, given principal, annual interest rate, and loan term.",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {
                        "type": "number",
                        "description": "Loan principal amount in dollars.",
                    },
                    "annual_rate_pct": {
                        "type": "number",
                        "description": "Annual interest rate as a percentage, e.g. 6.5 for 6.5%, not a decimal fraction.",
                    },
                    "years": {
                        "type": "integer",
                        "description": "Loan term length in years.",
                    },
                },
                "required": ["principal", "annual_rate_pct", "years"],
            },
        },
    }

    get_debt_to_income_schema: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": "debt_to_income_ratio",
            "description": "Calculate debt-to-income ratio as a percentage, given total monthly debt payments and gross monthly income.",
            "parameters": {
                "type": "object",
                "properties": {
                    "monthly_debt_payments": {
                        "type": "number",
                        "description": "Total monthly debt payments in dollars, including the mortgage payment if applicable.",
                    },
                    "monthly_income": {
                        "type": "number",
                        "description": "Gross monthly income in dollars.",
                    },
                },
                "required": ["monthly_debt_payments", "monthly_income"],
            },
        },
    }

    get_amortization_schema: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": "amortization_point_in_time",
            "description": "Calculate the remaining loan balance at a specific point in the amortization schedule.",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {
                        "type": "number",
                        "description": "Loan principal amount in dollars.",
                    },
                    "annual_rate_pct": {
                        "type": "number",
                        "description": "Annual interest rate as a percentage, e.g. 6.5 for 6.5%, not a decimal fraction.",
                    },
                    "years": {
                        "type": "integer",
                        "description": "Loan term length in years.",
                    },
                    "months": {
                        "type": "integer",
                        "description": "Number of months elapsed since the loan began (0 = at origination). Must not exceed the total number of payments (years * 12).",
                    },
                },
                "required": ["principal", "annual_rate_pct", "years", "months"],
            },
        },
    }

    get_extra_payment_schema: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": "extra_payment_calculator",
            "description": "Calculate the payoff time and interest savings from paying extra toward principal each month.",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {
                        "type": "number",
                        "description": "Loan principal amount in dollars.",
                    },
                    "annual_rate_pct": {
                        "type": "number",
                        "description": "Annual interest rate as a percentage, e.g. 6.5 for 6.5%, not a decimal fraction.",
                    },
                    "years": {
                        "type": "integer",
                        "description": "Loan term length in years.",
                    },
                    "extra_payment": {
                        "type": "number",
                        "description": "Additional amount paid toward principal each month, in dollars, on top of the regular payment.",
                    },
                },
                "required": [
                    "principal",
                    "annual_rate_pct",
                    "years",
                    "extra_payment",
                ],
            },
        },
    }

    get_affordability_schema: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": "affordability_in_reverse",
            "description": "Estimate the maximum loan amount and monthly payment affordable given income, existing debts, and a target debt-to-income ratio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "annual_income": {
                        "type": "number",
                        "description": "Gross annual income in dollars.",
                    },
                    "target_dti": {
                        "type": "number",
                        "description": "Target debt-to-income ratio as a percentage, e.g. 43 for 43%, not a decimal fraction.",
                    },
                    "annual_rate_pct": {
                        "type": "number",
                        "description": "Annual interest rate as a percentage, e.g. 6.5 for 6.5%, not a decimal fraction.",
                    },
                    "years": {
                        "type": "integer",
                        "description": "Loan term length in years.",
                    },
                    "monthly_debts": {
                        "type": "number",
                        "description": "Existing monthly debt payments in dollars, not including the new mortgage payment being solved for.",
                    },
                },
                "required": [
                    "annual_income",
                    "target_dti",
                    "annual_rate_pct",
                    "years",
                    "monthly_debts",
                ],
            },
        },
    }

    return [
        get_mortgage_payment_schema,
        get_debt_to_income_schema,
        get_amortization_schema,
        get_extra_payment_schema,
        get_affordability_schema,
    ]


def monthly_mortgage_payment(
    principal: float, annual_rate_pct: float, years: int
) -> float:
    interest_rate = annual_rate_pct / 100 / 12
    number_of_payments = years * 12

    if principal <= 0:
        raise ValueError("Principal must be a positive number.")

    if number_of_payments <= 0:
        raise ValueError("Number of payments must be a positive integer.")

    if interest_rate == 0:
        return principal / number_of_payments

    return (
        principal
        * (interest_rate * (1 + interest_rate) ** number_of_payments)
        / ((1 + interest_rate) ** number_of_payments - 1)
    )


def debt_to_income_ratio(monthly_debt_payments: float, monthly_income: float) -> float:
    if monthly_income <= 0:
        raise ValueError("Monthly income must be a positive number.")

    return (monthly_debt_payments / monthly_income) * 100


def amortization_point_in_time(
    principal: float, annual_rate_pct: float, years: int, months: int
) -> float:
    number_of_payments = years * 12

    if months < 0 or months > number_of_payments:
        raise ValueError(
            "Months must be a non-negative integer not exceeding the total number of payments."
        )

    monthly_rate = annual_rate_pct / 100 / 12
    monthly_payment = monthly_mortgage_payment(principal, annual_rate_pct, years)

    if monthly_rate == 0:
        return principal - (monthly_payment * months)

    month = monthly_rate + 1
    growth = month**months

    return principal * growth - (monthly_payment * (growth - 1) / monthly_rate)


def extra_payment_calculator(
    principal: float, annual_rate_pct: float, years: int, extra_payment: float
) -> PayoffResult:
    monthly_rate = annual_rate_pct / 100 / 12
    number_of_payments = years * 12

    if extra_payment < 0:
        raise ValueError("Extra payment must be non-negative.")

    if monthly_rate <= 0:
        raise ValueError("Annual interest rate must be positive.")

    monthly_payment = monthly_mortgage_payment(principal, annual_rate_pct, years)
    new_payment = monthly_payment + extra_payment

    if new_payment <= principal * monthly_rate:
        raise ValueError("Extra payment is too low to reduce the principal.")

    months_to_payoff = ceil(
        -log(1 - (principal * monthly_rate) / new_payment) / log(1 + monthly_rate)
    )

    months_saved = number_of_payments - months_to_payoff
    new_total_interest = months_to_payoff * new_payment - principal
    original_total_interest = number_of_payments * monthly_payment - principal
    interest_saved = original_total_interest - new_total_interest

    return PayoffResult(
        payoff_months=months_to_payoff,
        months_saved=months_saved,
        interest_saved=interest_saved,
        new_total_interest=new_total_interest,
    )


def affordability_in_reverse(
    annual_income: float,
    monthly_debts: float,
    target_dti: float,
    annual_rate_pct: float,
    years: int,
) -> AffordabilityResult:
    if annual_income <= 0:
        raise ValueError("Annual income must be a positive number.")

    if years <= 0:
        raise ValueError("Years must be a positive integer.")

    if annual_rate_pct <= 0:
        raise ValueError("Annual interest rate must be a positive number.")

    max_payment = (annual_income / 12 * target_dti / 100) - monthly_debts

    if max_payment <= 0:
        raise ValueError(
            "Target DTI and monthly debts result in non-positive max payment."
        )

    monthly_rate = annual_rate_pct / 100 / 12
    number_of_payments = years * 12

    max_loan_amount = (
        max_payment
        * ((1 + monthly_rate) ** number_of_payments - 1)
        / (monthly_rate * (1 + monthly_rate) ** number_of_payments)
    )

    return AffordabilityResult(max_loan_amount=max_loan_amount, max_payment=max_payment)
