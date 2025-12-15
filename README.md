# ACME Models CLI

ACME Models is a small CLI + backend utilities for scoring model repositories and serving artifact operations. The project is structured around:

- `src/`: core scoring/metrics, clients, storage, CLI entrypoint
- `backend/`: serverless-style handlers (Lambda-compatible) for artifact APIs
- `tests/`: unit + integration-style tests

## Quick Start

### Prerequisites

- Python 3.12+
- `pip` (bundled with Python)

### Install

```bash
./run install
```

### Run the CLI

```bash
python3 -m src.CLIApp URL_FILE
```

Or via the launcher (also validates required env vars):

```bash
./run URL_FILE
```

## Development

### Lint / Typecheck

Run the repo checks (isort + mypy + flake8):

```bash
./check
```

### Tests

Run the test suite and print the short required summary:

```bash
./run test
```

Run pytest directly (shows full output and per-file missing lines):

```bash
./run pytest
```

### Coverage (High-Assurance)

Integration/E2E-style coverage (>= 95%) and HTML report:

```bash
bash scripts/coverage_integration.sh
open htmlcov-integration/index.html
```

Component-style coverage (>= 90%) and HTML report:

```bash
bash scripts/coverage_component.sh
open htmlcov-component/index.html
```

“Error messages produced” proxy (percent of executed `raise ...` statements), using the integration coverage data:

```bash
python3 scripts/error_raise_coverage.py --data-file .coverage.integration --fail-under 80
```

## Environment Variables

Create a `.env` file (see `.env_example`) to configure runtime behavior.

### Required for normal CLI runs

- `GITHUB_TOKEN`: token used by GitHub API clients.
- `LOG_FILE`: path to a `.log` file where logs should be written.

### Optional

- `LOG_LEVEL`: `0` disables logging; `1` enables INFO; `2` enables DEBUG.
- `GEN_AI_STUDIO_API_KEY`: API token for Purdue GenAI Studio (required only if invoking the Purdue client / LLM workflows).
- `ACME_IGNORE_FAIL`: set to `1` to ignore FAIL flags in metrics during tests.
- `ACME_ENABLE_README_FALLBACK`: set to `0/1` to control README-based fallback parsing in some metrics.

## Troubleshooting

- If “error coverage” looks wrong, regenerate coverage first (the helpers ensure `.coverage.integration` and `.coverage.component` contain real data):
  - `bash scripts/coverage_integration.sh`
  - `bash scripts/coverage_component.sh`

## Metrics

The CLI computes a set of metrics for each input record and returns a list of `MetricResult` objects (`metric`, `key`, `value`, `latency_ms`, optional `error`). The standard metric set is defined in `src/metrics/registry.py`.

- Net Score (`net_score`): overall score computed as the average of numeric metric values in `[0, 1]` (rounded to 2 decimals).
- Ramp-up Score (`ramp_up_time`): estimates how quickly a user can adopt the model by looking for README structure, usage examples, and helpful links (with optional LLM assistance).
- Bus Factor (`bus_factor`): estimates contributor diversity/activity using GitHub metadata inferred from the model card/manifest.
- License Score (`license`): evaluates the model’s license signal against the project’s license policy.
- Size Score (`size_score`): per-device deployability scores computed from model weight artifact sizes; returns a mapping like `{raspberry_pi, jetson_nano, desktop_pc, aws_server}`.
- Dataset & Code Availability (`dataset_and_code_score`): checks for referenced datasets and code repositories (manifest/metadata/README).
- Dataset Quality (`dataset_quality`): evaluates dataset documentation, popularity/adoption, licensing, and freshness when a dataset is identified.
- Code Quality (`code_quality`): estimates repository hygiene signals (CI, lint/type configs, docs, and recent activity).
- Reproducibility (`reproducibility`): checks whether README demo code blocks can execute without manual fixes (with an optional LLM repair loop for minor issues).
- Reviewedness (`reviewedness`): estimates the fraction of code lines authored via reviewed PRs using sampling; returns `-1.0` for non-GitHub repos and `0.0` for analysis failures.
- Performance Claims (`performance_claims`): returns `1.0` if explicit benchmark/evaluation evidence is detected, else `0.0` (with optional LLM fallback).
- Tree Score (`tree_score`): uses Hugging Face “base model / fine-tuned from” lineage to average parent model scores; falls back to the target model’s own net score when no parents are found.

## Backend API

The serverless backend is defined in `backend/template.yaml` and implemented in `backend/src/handlers/`.

### Authentication

- Call `PUT /authenticate` with `{ "user": { "name": "..." }, "secret": { "password": "..." } }` to receive a token string (usually `bearer ...`).
- For authenticated endpoints, pass the token via `X-Authorization` (preferred) or `Authorization`.

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `PUT` | `/authenticate` | Issue an auth token for the default admin user. |
| `GET` | `/health` | Health check heartbeat (no auth). |
| `GET` | `/tracks` | Returns the planned tracks list (no auth). |
| `POST` | `/artifact/{artifact_type}` | Create a new artifact (`artifact_type` in `{model,dataset,code}`) from a request body like `{ "url": "...", "name": "optional" }`. |
| `GET` | `/artifacts/{artifact_type}/{id}` | Fetch artifact metadata + a `download_url`. |
| `PUT` | `/artifacts/{artifact_type}/{id}` | Update an artifact’s mutable fields (currently, the source `data.url`) while enforcing immutable metadata. |
| `DELETE` | `/artifacts/{artifact_type}/{id}` | Delete an artifact and associated stored objects (best effort across blob/metadata/index/rating). |
| `POST` | `/artifacts` | Enumerate artifacts by name/type queries; supports pagination via an `offset` query param and `offset` response header. |
| `POST` | `/artifact/byRegEx` | Search artifacts by a request body like `{ "regex": "..." }`. |
| `GET` | `/download/{id}` | Return a signed download URL (JSON with `?format=json`, otherwise redirects). |
| `GET` | `/artifact/model/{id}/rate` | Return a cached model rating payload; computes and stores a rating on cache miss. |
| `POST` | `/artifact/model/{id}/license-check` | Evaluate license compatibility for a model given `{ "github_url": "..." }`. |
| `GET` | `/artifact/model/{id}/lineage` | Return the complete lineage family graph for a model (nodes + edges). |
| `GET` | `/artifact/{artifact_type}/{id}/cost` | Calculate artifact cost; supports `?dependency=true` to include dependency lineage. |
| `DELETE` | `/reset` | Reset backing stores (intended for admin/testing). |
