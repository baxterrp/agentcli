from math import log, ceil
from dataclasses import dataclass


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
