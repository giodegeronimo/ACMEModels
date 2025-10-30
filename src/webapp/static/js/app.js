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

    let nextCursor = null;
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
                <td>${item.owner}</td>
                <td>${item.license?.toUpperCase?.() ?? "—"}</td>
                <td>${item.vetted ? "Yes" : "No"}</td>
                <td>${item.size_mb?.toFixed?.(1) ?? "—"}</td>
                <td>${formatTimestamp(item.updated_at)}</td>
            `;
            tableBody.appendChild(row);
        }
    };

    const setLoading = (state) => {
        loading = state;
        loadMoreBtn.disabled = state || !nextCursor;
        statusEl.textContent = state ? "Loading models…" : "";
    };

    const loadPage = async ({ append = false } = {}) => {
        if (loading) return;
        setLoading(true);

        try {
            const params = new URLSearchParams();
            params.set("limit", "20");
            if (activeQuery) params.set("q", activeQuery);
            if (append && nextCursor) params.set("cursor", nextCursor);

            const payload = await fetchJson(`/api/models?${params.toString()}`);
            if (!append) {
                tableBody.innerHTML = "";
            }
            renderRows(payload.items || []);
            nextCursor = payload.next_cursor ?? null;
            loadMoreBtn.disabled = !nextCursor;
            statusEl.textContent = payload.items?.length ? "" : "No models matched your search.";
        } catch (error) {
            statusEl.textContent = `Unable to load models. ${error.message}`;
            nextCursor = null;
        } finally {
            setLoading(false);
        }
    };

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        activeQuery = form.q.value.trim();
        nextCursor = null;
        loadPage({ append: false });
    });

    loadMoreBtn.addEventListener("click", () => loadPage({ append: true }));
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
        if (!payload || !payload.nodes?.length) {
            lineageContainer.innerHTML = "<p>No lineage information available.</p>";
            return;
        }

        const list = document.createElement("ul");
        list.className = "lineage-list";
        for (const node of payload.nodes) {
            const item = document.createElement("li");
            const vetted = node.vetted ? "vetted" : "unvetted";
            item.textContent = `${node.name} • ${node.license.toUpperCase()} • ${vetted}`;
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

    fetchJson(`/api/models/${encodeURIComponent(modelId)}/lineage`)
        .then(renderLineage)
        .catch(() => {
            if (lineageContainer) {
                lineageContainer.innerHTML = "<p>Failed to load lineage.</p>";
            }
        });

    fetchJson(`/api/models/${encodeURIComponent(modelId)}/size-cost`)
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

        const payload = {
            name: form.name.value.trim(),
            source_url: form.source_url.value.trim(),
            submitted_by: form.submitted_by.value.trim(),
            metrics
        };

        try {
            const response = await fetchJson("/api/models/ingest", {
                method: "POST",
                body: JSON.stringify(payload)
            });
            statusEl.textContent = `Request queued. ID: ${response.request_id}`;
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

        const payload = {
            repo_license: form.repo_license.value.trim(),
            model_id: form.model_id.value.trim() || undefined,
            model_license: form.model_license.value.trim() || undefined
        };

        try {
            const response = await fetchJson("/api/license-check", {
                method: "POST",
                body: JSON.stringify(payload)
            });
            statusEl.textContent = `Result: ${response.compatibility}. ${response.rationale}`;
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
            await fetchJson("/api/reset", { method: "POST" });
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
