"""Run the interpretability evaluation and store the results as a CSV file.

Usage::

    uv run python scripts/run_experiment.py
"""

from __future__ import annotations

from pathlib import Path

from iml_interpretability import (
    StructuralInterpretability,
    evaluate,
    get_models,
    load_diabetes,
    load_endgames,
    load_titanic,
    load_wdbc,
)

OUTPUT = Path("results/scores.csv")
SEED = 0


def main() -> None:
    results = evaluate(
        datasets=[load_titanic(), load_diabetes(), load_wdbc(), load_endgames()],
        models=get_models(seed=SEED),
        metrics=[StructuralInterpretability()],  # RelativeNumberOfLeaves doesnt work for ensembels
        seed=SEED,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)
    print(results.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
