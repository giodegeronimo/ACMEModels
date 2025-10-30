# ACME Models CLI

Command line tool for scoring pre-trained ML assets against ACME's quality metrics.

## Requirements

- Python 3.11+
- `pip3`
- Valid credentials for the services you plan to use (GitHub and Purdue GenAI Studio).

## Installation

Install Python dependencies with the project wrapper:

```bash
./run install
```

## Environment Configuration

Runtime settings can be supplied via standard environment variables or a local `.env` file (loaded automatically on first use). The following values are required:

- `GITHUB_TOKEN` – personal access token used for GitHub API requests. The CLI validates that the token resembles a real PAT (e.g., starts with `ghp_`).
- `LOG_FILE` – absolute or relative path to a writable log file. The parent directory will be created if it does not already exist.

Optional settings:

- `LOG_LEVEL` – `0` (silent, default), `1` (info), or `2` (debug).
- `GEN_AI_STUDIO_API_KEY` – required for metrics that call Purdue's GenAI Studio.
- `ACME_ENABLE_README_FALLBACK` – set to `0` to disable README scraping fallbacks when locating external resources.
- `ACME_IGNORE_FAIL` – set to `1` to force placeholder metrics to behave as implemented during testing.

Store secrets in your shell session or an untracked `.env` file (never commit real tokens).

## Usage

Run the CLI against a newline-delimited URL manifest:

```bash
GITHUB_TOKEN=ghp_your_token LOG_FILE=/tmp/acme.log ./run /absolute/path/to/URL_FILE
```

The `URL_FILE` can contain model, dataset, and code repository URLs. The CLI emits NDJSON on stdout that includes per-metric scores and latencies for each model encountered.

### Manifest Format

Each non-empty line in the manifest represents a model and follows the order:

```
<git repository URL>,<dataset URL>,<huggingface model URL>
```

Fields may be left blank, but trailing commas must remain so parsing stays consistent. Example:

```
https://github.com/google-research/bert,https://huggingface.co/datasets/squad,https://huggingface.co/google-bert/bert-base-uncased
,,https://huggingface.co/openai/whisper-tiny
```

### Output Format

The CLI writes one NDJSON line per model. A representative record looks like:

```json
{"name":"bert-base-uncased","category":"MODEL","net_score":0.79,"net_score_latency":312,"ramp_up_time":0.82,"ramp_up_time_latency":104,"bus_factor":0.55,"bus_factor_latency":87,"performance_claims":1.0,"performance_claims_latency":65,"license":1.0,"license_latency":40,"size_score":{"raspberry_pi":0.15,"jetson_nano":0.32,"desktop_pc":0.88,"aws_server":1.0},"size_score_latency":53,"dataset_and_code_score":0.74,"dataset_and_code_score_latency":41,"dataset_quality":0.68,"dataset_quality_latency":58,"code_quality":0.71,"code_quality_latency":92}
```

Fields absent from a manifest remain unspecified in the output. Numerical values fall within `[0, 1]`, and latencies are reported in milliseconds.

## Web UI (Prototype)

A Flask-powered prototype user interface is available for browsing the registry and exercising the REST endpoints.

```bash
LOG_FILE=/tmp/acme-web.log ./run web
```

By default the server listens on `http://127.0.0.1:5000`. Override with the environment variables `ACME_WEB_HOST` and `ACME_WEB_PORT`. The UI surfaces the following REST resources:

- `/api/models` – paginated directory with optional regex filtering (`q` parameter).
- `/api/models/<model_id>` – detailed model metadata.
- `/api/models/<model_id>/lineage` – simplified lineage graph derived from config metadata.
- `/api/models/<model_id>/size-cost` – download footprint heuristics.
- `/api/models/ingest` – submit a HuggingFace model for ingestion (requires all non-latency metrics ≥ 0.5).
- `/api/license-check` – evaluate GitHub and model license compatibility for fine-tuning plus inference.
- `/api/reset` – restore the registry to its default, empty state.

The HTML templates follow WCAG 2.1 AA guidance, include skip links, and are keyboard navigable. Front-end scripts populate dynamic content and degrade gracefully if JavaScript is disabled.

## Running Tests

Execute the curated test suite with coverage summary:

```bash
GITHUB_TOKEN=ghp_your_token LOG_FILE=/tmp/acme-test.log ./run test
```

For full pytest output, use:

```bash
GITHUB_TOKEN=ghp_your_token LOG_FILE=/tmp/acme-test.log ./run pytest -k "pattern"
```

### Web UI Tests

Unit tests for the Flask views and API live under `tests/web`. A Selenium smoke test is included; enable it by setting `SELENIUM_BROWSER` (e.g., `chrome` or `firefox`) and ensuring the corresponding WebDriver binary is on `PATH`.

## Logging

Logs are written to `LOG_FILE` when `LOG_LEVEL` is greater than zero. Debug logging (`LOG_LEVEL=2`) provides detailed metric computations suitable for troubleshooting.

## LLM Configuration

Model identifiers, temperature, and other LLM-related defaults are centralized in `src/config.py`. Update those constants if you need to target a different Purdue GenAI Studio model or adjust sampling parameters.
