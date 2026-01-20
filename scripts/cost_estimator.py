"""Estimate rough training costs based on hardware and runtime assumptions."""

from dataclasses import dataclass


@dataclass
class CostInputs:
    hourly_rate: float
    num_nodes: int
    hours: float


def estimate_cost(inputs: CostInputs) -> float:
    return inputs.hourly_rate * inputs.num_nodes * inputs.hours


def main() -> None:
    inputs = CostInputs(hourly_rate=12.5, num_nodes=8, hours=24)
    total = estimate_cost(inputs)
    print(f"Estimated cost: ${total:,.2f}")


if __name__ == "__main__":
    main()
