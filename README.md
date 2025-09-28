# ECE30861 SWE Project
# Team Members: Eren Ulke, Salaheldin , Hussein Hanafy, Felix Wu
LLM Scoring Pipeline

A lightweight, CLI-first pipeline to evaluate LLM behavior at scale. The system flows left-to-right—fetch → analyze → score → format—and emits both human-readable logs and machine-readable NDJSON for downstream analysis. A top-level run helper script simplifies installation, batch evaluation from an input file, and testing (with coverage).

Project flow:

1.Fetch evaluation artifacts from configured sources.

2.Analyze raw inputs/outputs to extract features/signals.

3.Score each item into quantitative and categorical metrics.

4.Format results as NDJSON plus readable summaries and logs.

5.Test the end-to-end flow with coverage reporting.

Project Structure
.
├── run                 # user-facing entrypoint (install/run/test)
├── CLI.py              # orchestrates end-to-end flow
├── URL_Fetcher.py      # data acquisition (generalized providers)
├── LLM_Analyzer.py     # feature/Signal extraction
├── Scorer.py           # numeric + categorical metrics
├── Output_Formatter.py # NDJSON + human-readable outputs
├── tests/              # unit/integration tests
└── out.ndjson          # example result artifact (after a run)

Install
./run install
# Installs Python deps, sets up local venv if applicable.

Quick Start
# Provide an input list (one item per line), e.g. urls.txt
./run urls.txt

# Artifacts:
# - out.ndjson
# - logs: acme.log, err.log


