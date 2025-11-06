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

    const metricInputs = form.querySelectorAll("input[name^='metrics']");

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        statusEl.hidden = false;
        statusEl.textContent = "Submitting request…";

        const metrics = {};
        metricInputs.forEach((input) => {
            const key = input.name.replace("metrics[", "").replace("]", "");
            metrics[key] = Number.parseFloat(input.value);
        });

        const submittedBy = form.submitted_by.value.trim();
        const payload = {
            name: form.name.value.trim(),
            url: form.source_url.value.trim(),
            download_url: form.source_url.value.trim(),
            owner: submittedBy || "external",
            submitted_by: submittedBy || undefined,
            metrics
        };

        try {
            const response = await fetchJson("/api/artifact/model", {
                method: "POST",
                body: JSON.stringify(payload)
            });
            const artifactId = response?.metadata?.id ?? "unknown";
            statusEl.textContent = `Artifact registered. ID: ${artifactId}`;
            form.reset();
        } catch (error) {
            statusEl.textContent = `Unable to submit request. ${error.message}`;
        }
    });
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
