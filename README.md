# iML 26/27 – Exercise 1: Quantifying Interpretability

Template repository for the bonus task. **Fork this repository**, replace the example with your own
work, and add the PDF of your presentation. **You are expected to rewrite this README as well**: by
the deadline it should describe your approach, how to reproduce your results, and contain your own
[AI statement](#ai-statement).

> **Deadline: Sunday, November 15th, 2026, 23:59.** Your code and presentation must be submitted by then.

## Tasks

### Task 1 (1  point): An interpretability metric

Develop a metric that quantifies model interpretability. It has to work within one model class,
e.g. for decision trees, random forests and gradient-boosted trees. Then evaluate a few
inherently interpretable models and/or datasets and investigate the *threshold of
interpretability*: at what point does a model stop being interpretable?

- A trivial idea would be the *number of features used*. Your metric should go beyond that.
- We will check that you did not simply copy someone else's interpretability metric.
- You may use an LLM-based metric. In that case you have to provide evidence that its scores
  agree with human judgments.

#### We only did task 1 this week. I wasn't sure what to write in the README, as everything is pretty much explained in the presentation.pdf 😃

### Task 2 (2 bonus points): Meta-features and interpretability

Find connections between a dataset's meta-features (e.g. number of samples, number of features,
class balance, …) and the interpretability, measured with your metric from Task 1, of several
inherently interpretable models. This repository has no code for Task 2. Structure it as you see fit.

### Presentation

Prepare a **5-minute presentation** of your metric and results across models and datasets (and, if
applicable, your Task 2 findings). Commit it as [`presentation/presentation.pdf`](presentation/).

## Setup

The repository is an installable Python package. We recommend [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/run_experiment.py
```

I updated the weights in my metric to the weights of the Logistic-Regression model.
If you want to rerun the fitting do:
```bash
uv run python scripts/run_fit.py
```

This fits a tree for all labeled trees in [`labels.csv`](labeling/labels.csv).
In case you want to label additional trees add them to [`items.csv`](labeling/items.csv)
or write a custom tree-sampler. 
I used the AI generated tree-sampler ([`make_labeling_set.py`](labeling/make_labeling_set.py)).

To run my version of the tree-sampler (that generated the file  [`items.csv`](labeling/items.csv)), run:
```bash
uv run python scripts/make_labeling_set.py
```

To use the html interface for labeling run: 
```bash
open labeling/label_trees.html
```
and export the results to [`labels.csv`](labeling/labels.csv).



## Repository structure

```
iml_interpretability/
├── data.py        # Dataset container + load_titanic(), load_diabetes(), load_wdbc(), load_endgames()
├── models.py      # get_models(): the models you compare (decision tree, random forest, gradient boost)
├── metric.py      # InterpretabilityMetric base class + StructuralInterpretability
├── tree.py        # Node + tree helpers: trees_of, simplify, essential_split_share, effective_n_trees
└── evaluate.py    # evaluate(): fits all models on all datasets and computes all metrics
scripts/
├── run_experiment.py   # runs evaluate() and writes results/scores.csv
└── run_fit.py          # fits a logistic regression on labeling/labels.csv, writes labeling/fitted_weights.json
labeling/
├── make_labeling_set.py      # samples models, renders the trees, writes items.csv + label_trees.html
├── label_page_template.html  # HTML/JS template for the labeling page
├── label_trees.html          # generated labeling page (open in a browser, export labels.csv)
├── items.csv                 # the 100 sampled models with their features
├── labels.csv                # my hand labels (interpretable yes/no)
└── fitted_weights.json       # threshold + weights from run_fit.py

tests/                  # pytest tests
presentation/           # presentation.pdf here
```


## Code style

I used [ruff](https://docs.astral.sh/ruff/) as linter and formatter.


## AI statement
I used Claude Desktop for:
- Brainstroming on what features to use and discussed that. 
- Writing tests (I clarified the scope and let it automatically generate tests (95% of them were super useful)).
- Building the web-interface for the labeling process. I feel like this is what LLMs are very good at (building simple interfaces very fast).
  - I didnt verify the code at all, but did "Stichproben" on the output file (that is used as the basis for the training of the LM).
  - So everything under "labeling" is ai generated.
  - The idea of AI generating the candidate set as well was, that I am the only grader, so if I would generate the candidates AND grade them, that might introduce a higher bias then splitting the tasks. 
- Creating and "beatifying" graphs 
  - for the graphs i didnt double check what the ai did, but i didnt use them for anything. It was just to get an intuition of what the metric does (probably)
