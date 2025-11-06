/* global document, window */

async function fetchJson(url, options = {}) {
    const response = await fetch(url, {
        headers: {
            "Content-Type": "application/json",
            Accept: "application/json"
        },
        ...options
    });

    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || response.statusText);
    }

    return response.json();
}

function formatTimestamp(epochSeconds) {
    const date = new Date(epochSeconds * 1000);
    return date.toLocaleString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit"
    });
}

function formatScore(value) {
    if (value === null || value === undefined || value === "") {
        return "—";
    }
    const numeric = Number.parseFloat(value);
    if (Number.isNaN(numeric)) {
        return String(value);
    }
    return numeric.toFixed(2);
}

function formatLatency(value) {
    if (value === null || value === undefined || value === "") {
        return "—";
    }
    const numeric = Number.parseFloat(value);
    if (Number.isNaN(numeric)) {
        return String(value);
    }
    return `${Math.round(numeric)} ms`;
}

function normalizeMetricValue(value) {
    if (value === null || value === undefined) {
        return value;
    }
    if (Array.isArray(value)) {
        return value.map(normalizeMetricValue);
    }
    if (typeof value === "object") {
        const normalized = {};
        for (const [key, nestedValue] of Object.entries(value)) {
            normalized[key] = normalizeMetricValue(nestedValue);
        }
        return normalized;
    }
    if (typeof value === "boolean") {
        return value;
    }
    const numeric = Number(value);
    if (!Number.isNaN(numeric) && Number.isFinite(numeric)) {
        return numeric;
    }
    return value;
}

function parseCliResults(rawText, preferredName) {
    if (!rawText) {
        throw new Error("CLI output is required.");
    }
    const lines = rawText
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean);
    if (!lines.length) {
        throw new Error("CLI output is required.");
    }

    const records = [];
    for (const line of lines) {
        try {
            const parsed = JSON.parse(line);
            if (parsed && typeof parsed === "object") {
                records.push(parsed);
            }
        } catch (error) {
            throw new Error("CLI output must be valid JSON (NDJSON).");
        }
    }
    if (!records.length) {
        throw new Error("No valid records found in CLI output.");
    }

    let selected = null;
    const normalizedPreference = preferredName ? preferredName.trim().toLowerCase() : "";
    if (normalizedPreference) {
        selected = records.find(
            (record) => typeof record.name === "string"
                && record.name.trim().toLowerCase() === normalizedPreference
        );
    }
    if (!selected) {
        [selected] = records;
    }

    if (!selected || typeof selected !== "object") {
        throw new Error("Unable to determine which CLI record to use.");
    }

    const metrics = {};
    for (const [key, value] of Object.entries(selected)) {
        if (key === "name" || key === "category") continue;
        metrics[key] = normalizeMetricValue(value);
    }

    return {
        record: selected,
        metrics,
        totalRecords: records.length,
    };
}

function initModelDirectory() {
    const form = document.getElementById("model-search-form");
    const tableBody = document.getElementById("model-table-body");
    const statusEl = document.getElementById("models-status");
    const loadMoreBtn = document.getElementById("load-more");

    if (!form || !tableBody || !statusEl || !loadMoreBtn) {
        return;
    }

    let nextOffset = null;
    let activeQuery = "";
    let loading = false;

    const renderRows = (items) => {
        if (!items.length && tableBody.children.length === 0) {
            statusEl.textContent = "No models matched your search.";
            return;
        }

        for (const item of items) {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td><a href="/models/${encodeURIComponent(item.id)}">${item.name}</a></td>
                <td>${item.owner ?? "—"}</td>
                <td>${formatScore(item.net_score)}</td>
                <td>${item.license?.toUpperCase?.() ?? "—"}</td>
                <td>${item.vetted ? "Yes" : "No"}</td>
                <td>${item.size_mb != null ? Number.parseFloat(item.size_mb).toFixed(1) : "—"}</td>
                <td>${item.updated_at ? formatTimestamp(item.updated_at) : "—"}</td>
            `;
            tableBody.appendChild(row);
        }
    };

    const fetchArtifactPage = async ({ append = false } = {}) => {
        const params = new URLSearchParams();
        params.set("limit", "20");
        if (append && nextOffset !== null) {
            params.set("offset", String(nextOffset));
        }

        const response = await fetch(`/api/artifacts?${params.toString()}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Accept: "application/json"
            },
            body: JSON.stringify([{ name: "*" }])
        });

        if (!response.ok) {
            const message = await response.text();
            throw new Error(message || response.statusText);
        }

        const items = await response.json();
        if (!append) {
            tableBody.innerHTML = "";
        }
        renderRows(items);

        const offsetHeader = response.headers.get("offset");
        if (offsetHeader !== null) {
            const parsed = Number.parseInt(offsetHeader, 10);
            nextOffset = Number.isNaN(parsed) ? null : parsed;
        } else {
            nextOffset = null;
        }

        loadMoreBtn.disabled = Boolean(activeQuery) || nextOffset === null;
        statusEl.textContent = tableBody.children.length ? "" : "No models matched your search.";
    };

    const fetchRegexResults = async (pattern) => {
        const payload = await fetchJson("/api/artifact/byRegEx", {
            method: "POST",
            body: JSON.stringify({ regex: pattern })
        });
        tableBody.innerHTML = "";
        renderRows(payload);
        nextOffset = null;
        loadMoreBtn.disabled = true;
        statusEl.textContent = tableBody.children.length ? "" : "No models matched your search.";
    };

    const loadPage = async ({ append = false } = {}) => {
        if (loading) return;
        loading = true;
        statusEl.textContent = "Loading models…";
        loadMoreBtn.disabled = true;

        try {
            if (activeQuery) {
                if (!append) {
                    await fetchRegexResults(activeQuery);
                }
            } else {
                await fetchArtifactPage({ append });
            }
        } catch (error) {
            statusEl.textContent = `Unable to load models. ${error.message}`;
            nextOffset = null;
            loadMoreBtn.disabled = true;
        } finally {
            loading = false;
            if (!statusEl.textContent.startsWith("Unable")) {
                statusEl.textContent = tableBody.children.length ? "" : "No models matched your search.";
                loadMoreBtn.disabled = Boolean(activeQuery) || nextOffset === null;
            }
        }
    };

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        activeQuery = form.q.value.trim();
        nextOffset = null;
        loadPage({ append: false });
    });

    loadMoreBtn.addEventListener("click", () => {
        if (activeQuery) return;
        loadPage({ append: true });
    });

    loadPage();
}

function initModelDetail() {
    const panel = document.querySelector("[data-model-id]");
    if (!panel) return;

    const modelId = panel.getAttribute("data-model-id");
    const lineageContainer = document.getElementById("lineage-container");
    const sizeContainer = document.getElementById("size-cost-container");
    const metricsTableBody = document.getElementById("metrics-table-body");
    const metricsStatusCell = document.getElementById("metrics-status");
    const netScoreHighlight = document.getElementById("headline-net-score");

    const METRIC_ORDER = [
        ["net_score", "Net Score"],
        ["ramp_up_time", "Ramp Up Time"],
        ["bus_factor", "Bus Factor"],
        ["performance_claims", "Performance Claims"],
        ["license", "License Compliance"],
        ["dataset_and_code_score", "Dataset & Code Score"],
        ["dataset_quality", "Dataset Quality"],
        ["code_quality", "Code Quality"],
        ["reproducibility", "Reproducibility"],
        ["reviewedness", "Reviewedness"],
        ["tree_score", "Tree Score"],
        ["size_score", "Size Suitability"],
    ];

    const renderMetrics = (payload) => {
        if (!metricsTableBody) return;
        metricsTableBody.innerHTML = "";

        if (!payload || typeof payload !== "object") {
            const row = document.createElement("tr");
            const cell = document.createElement("td");
            cell.colSpan = 3;
            cell.textContent = "No metric details available.";
            row.appendChild(cell);
            metricsTableBody.appendChild(row);
            return;
        }

        if (netScoreHighlight) {
            netScoreHighlight.textContent = formatScore(payload.net_score);
        }

        for (const [key, label] of METRIC_ORDER) {
            const value = payload[key];
            const latency = payload[`${key}_latency`];

            const row = document.createElement("tr");
            const nameCell = document.createElement("td");
            const valueCell = document.createElement("td");
            const latencyCell = document.createElement("td");

            nameCell.textContent = label;

            if (value && typeof value === "object" && !Array.isArray(value)) {
                const list = document.createElement("ul");
                list.className = "metric-list";
                for (const [subKey, subValue] of Object.entries(value)) {
                    const item = document.createElement("li");
                    const prettyKey = subKey.replace(/_/g, " ");
                    const strong = document.createElement("strong");
                    strong.textContent = `${prettyKey}:`;
                    item.appendChild(strong);
                    item.append(` ${formatScore(subValue)}`);
                    list.appendChild(item);
                }
                valueCell.appendChild(list);
            } else {
                valueCell.textContent = formatScore(value);
            }

            latencyCell.textContent = formatLatency(latency);

            row.appendChild(nameCell);
            row.appendChild(valueCell);
            row.appendChild(latencyCell);
            metricsTableBody.appendChild(row);
        }
    };

    if (metricsStatusCell) {
        metricsStatusCell.textContent = "Loading metrics…";
    }

    const renderLineage = (payload) => {
        if (!lineageContainer) return;
        if (!payload || !Array.isArray(payload.nodes) || !payload.nodes.length) {
            lineageContainer.innerHTML = "<p>No lineage information available.</p>";
            return;
        }

        const list = document.createElement("ul");
        list.className = "lineage-list";
        for (const node of payload.nodes) {
            const item = document.createElement("li");
            const metadata = node.metadata || {};
            const vetted = (node.vetted ?? metadata.vetted) ? "vetted" : "unvetted";
            const license = (node.license ?? metadata.license ?? "unknown")
                .toString()
                .toUpperCase();
            item.textContent = `${node.name} • ${license} • ${vetted}`;
            list.appendChild(item);
        }

        lineageContainer.innerHTML = "";
        lineageContainer.appendChild(list);
    };

    const renderSize = (payload) => {
        if (!sizeContainer) return;
        if (!payload) {
            sizeContainer.innerHTML = "<p>No size information available.</p>";
            return;
        }

        // New artifact cost response: entries keyed by artifact id.
        if (!payload.model_id && !payload.capacity_score) {
            const entries = Object.entries(payload);
            if (!entries.length) {
                sizeContainer.innerHTML = "<p>No size information available.</p>";
                return;
            }

            const parseCost = (value, fallback = 0) => {
                const parsed = Number.parseFloat(value);
                return Number.isNaN(parsed) ? fallback : parsed;
            };

            const root = payload[modelId] || entries[0][1];
            const standalone = parseCost(root.standalone_cost ?? root.total_cost, 0);
            const total = parseCost(root.total_cost, standalone);
            const dependencies = entries.filter(([id]) => id !== modelId);

            let innerHtml = `
                <p>Total download cost: ${total.toFixed(1)} MB</p>
            `;
            if (!Number.isNaN(standalone) && standalone !== total) {
                innerHtml += `<p>Standalone cost: ${standalone.toFixed(1)} MB</p>`;
            }
            sizeContainer.innerHTML = innerHtml;

            if (dependencies.length) {
                const list = document.createElement("ul");
                list.className = "size-grid";
                for (const [id, cost] of dependencies) {
                    const card = document.createElement("li");
                    const dependencyCost = parseCost(cost.total_cost, 0);
                    card.innerHTML = `<strong>${id}</strong>: ${dependencyCost.toFixed(1)} MB`;
                    list.appendChild(card);
                }
                sizeContainer.appendChild(list);
            }
            return;
        }

        // Legacy payload fallback for older endpoints.
        const grid = document.createElement("div");
        grid.className = "size-grid";
        for (const [tier, score] of Object.entries(payload.capacity_score || {})) {
            const card = document.createElement("article");
            card.className = "size-card";
            card.innerHTML = `
                <h3>${tier.replace(/_/g, " ")}</h3>
                <p>Suitability score: ${(score * 100).toFixed(0)}%</p>
            `;
            grid.appendChild(card);
        }

        sizeContainer.innerHTML = `
            <p>Download size: ${(payload.size_mb ?? 0).toFixed(1)} MB</p>
            <p>Estimated download time: ${payload.estimated_download_minutes} minutes</p>
        `;
        sizeContainer.appendChild(grid);
    };

    fetchJson(`/api/artifact/model/${encodeURIComponent(modelId)}/rate`)
        .then(renderMetrics)
        .catch(() => {
            if (metricsTableBody) {
                metricsTableBody.innerHTML = "";
                const row = document.createElement("tr");
                const cell = document.createElement("td");
                cell.colSpan = 3;
                cell.textContent = "Failed to load metrics.";
                row.appendChild(cell);
                metricsTableBody.appendChild(row);
            }
            if (netScoreHighlight) {
                netScoreHighlight.textContent = "—";
            }
        });

    fetchJson(`/api/artifact/model/${encodeURIComponent(modelId)}/lineage`)
        .then(renderLineage)
        .catch(() => {
            if (lineageContainer) {
                lineageContainer.innerHTML = "<p>Failed to load lineage.</p>";
            }
        });

    fetchJson(`/api/artifact/model/${encodeURIComponent(modelId)}/cost?dependency=true`)
        .then(renderSize)
        .catch(() => {
            if (sizeContainer) {
                sizeContainer.innerHTML = "<p>Failed to load size cost data.</p>";
            }
        });
}

function initIngestForm() {
    const form = document.getElementById("ingest-form");
    const statusEl = document.getElementById("ingest-status");
    if (!form || !statusEl) return;

    const cliField = document.getElementById("cli-results");
    const previewEl = document.getElementById("cli-preview");
    let cachedCli = null;

    const renderCliPreview = () => {
        if (!previewEl) return;
        if (!cliField || !cliField.value.trim()) {
            previewEl.textContent = "Paste CLI output to populate metric scores automatically.";
            previewEl.classList.remove("form-help--error");
            cachedCli = null;
            return;
        }

        try {
            const parsed = parseCliResults(cliField.value, form.name.value);
            cachedCli = parsed;

            const summaryParts = [];
            if (parsed.record.name) {
                summaryParts.push(`Loaded metrics for ${parsed.record.name}.`);
            }
            if (typeof parsed.metrics.net_score !== "undefined") {
                summaryParts.push(`Net score ${formatScore(parsed.metrics.net_score)}.`);
            }
            if (parsed.totalRecords > 1) {
                summaryParts.push(`(${parsed.totalRecords} records detected; using the best match.)`);
            }
            previewEl.textContent = summaryParts.join(" ");
            previewEl.classList.remove("form-help--error");
        } catch (error) {
            cachedCli = null;
            previewEl.textContent = error.message;
            previewEl.classList.add("form-help--error");
        }
    };

    if (cliField) {
        cliField.addEventListener("input", renderCliPreview);
    }
    const nameField = form.querySelector("input[name='name']");
    if (nameField) {
        nameField.addEventListener("input", () => {
            if (cliField && cliField.value.trim()) {
                renderCliPreview();
            }
        });
    }

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        statusEl.hidden = false;
        statusEl.textContent = "Submitting request…";

        const cliText = cliField ? cliField.value : "";
        let parsedCli = cachedCli;
        if (!parsedCli) {
            try {
                parsedCli = parseCliResults(cliText, form.name.value);
            } catch (error) {
                statusEl.textContent = `Unable to read CLI output. ${error.message}`;
                return;
            }
        }

        const submittedBy = form.submitted_by.value.trim();
        let modelName = form.name.value.trim();
        if (!modelName && parsedCli.record.name) {
            modelName = parsedCli.record.name;
            form.name.value = modelName;
        }
        if (!modelName) {
            statusEl.textContent = "Model name is required.";
            return;
        }

        const sourceUrl = form.source_url.value.trim();
        if (!sourceUrl) {
            statusEl.textContent = "HuggingFace URL is required.";
            return;
        }

        const payload = {
            name: modelName,
            url: sourceUrl,
            download_url: sourceUrl,
            owner: submittedBy || "external",
            submitted_by: submittedBy || undefined,
            metrics: parsedCli.metrics || {},
        };

        if (parsedCli.record.category) {
            payload.tags = [parsedCli.record.category];
        }

        try {
            const response = await fetchJson("/api/artifact/model", {
                method: "POST",
                body: JSON.stringify(payload)
            });
            const artifactId = response?.metadata?.id ?? "unknown";
            statusEl.textContent = `Artifact registered. ID: ${artifactId}`;
            form.reset();
            cachedCli = null;
            if (previewEl) {
                previewEl.textContent = "Paste CLI output to populate metric scores automatically.";
                previewEl.classList.remove("form-help--error");
            }
        } catch (error) {
            statusEl.textContent = `Unable to submit request. ${error.message}`;
        }
    });

    renderCliPreview();
}

function initLicenseForm() {
    const form = document.getElementById("license-form");
    const statusEl = document.getElementById("license-status");
    if (!form || !statusEl) return;

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        statusEl.hidden = false;
        statusEl.textContent = "Checking license compatibility…";

        const artifactId = form.artifact_id.value.trim();
        const githubUrl = form.github_url.value.trim();
        if (!artifactId || !githubUrl) {
            statusEl.textContent = "Artifact ID and GitHub URL are required.";
            return;
        }

        try {
            const result = await fetchJson(
                `/api/artifact/model/${encodeURIComponent(artifactId)}/license-check`,
                {
                    method: "POST",
                    body: JSON.stringify({ github_url: githubUrl })
                }
            );
            const verdict = result ? "Compatible" : "Not compatible";
            statusEl.textContent = `Result: ${verdict}.`;
        } catch (error) {
            statusEl.textContent = `Unable to check compatibility. ${error.message}`;
        }
    });
}

function initResetButton() {
    const button = document.getElementById("reset-button");
    const feedback = document.getElementById("reset-feedback");
    if (!button || !feedback) return;

    button.addEventListener("click", async () => {
        feedback.hidden = false;
        feedback.textContent = "Resetting registry…";
        try {
            await fetchJson("/api/reset", { method: "DELETE" });
            feedback.textContent = "Registry restored to defaults.";
        } catch (error) {
            feedback.textContent = `Reset failed. ${error.message}`;
        }
    });
}

document.addEventListener("DOMContentLoaded", () => {
    initModelDirectory();
    initModelDetail();
    initIngestForm();
    initLicenseForm();
    initResetButton();
});
