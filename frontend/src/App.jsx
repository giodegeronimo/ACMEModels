import { useEffect, useMemo, useState } from "react";
import {
  createArtifact,
  getArtifact,
  getArtifactCost,
  getArtifactLineage,
  getArtifactRate,
  getIngestRequests,
  getStats,
  licenseCheck,
  listArtifacts,
  resetRegistry,
  searchArtifactsByRegex,
} from "./api";
import { parseCliResults } from "./utils/cli";
import "./App.css";

const ROUTES = [
  { pattern: "/", component: DashboardPage, label: "Dashboard" },
  { pattern: "/models", component: ModelDirectoryPage, label: "Models" },
  { pattern: "/models/:id", component: ModelDetailPage },
  { pattern: "/ingest", component: IngestPage, label: "Ingest" },
  { pattern: "/license", component: LicensePage, label: "License" },
  { pattern: "/admin/reset", component: ResetPage, label: "Admin" },
];

function getPathFromHash() {
  return window.location.hash.replace(/^#/, "") || "/";
}

function matchRoute(pathname) {
  for (const route of ROUTES) {
    const params = matchPattern(route.pattern, pathname);
    if (params) {
      return { ...route, params };
    }
  }
  return { ...ROUTES[0], params: {} };
}

function matchPattern(pattern, pathname) {
  const normalizedPattern = pattern.replace(/\/+$/, "") || "/";
  const normalizedPath = pathname.replace(/\/+$/, "") || "/";

  const patternSegments = normalizedPattern.split("/").filter(Boolean);
  const pathSegments = normalizedPath.split("/").filter(Boolean);

  if (normalizedPattern === "/") {
    return normalizedPath === "/" ? {} : null;
  }

  if (patternSegments.length !== pathSegments.length) {
    return null;
  }

  const params = {};
  for (let i = 0; i < patternSegments.length; i += 1) {
    const patternSegment = patternSegments[i];
    const pathSegment = pathSegments[i];
    if (patternSegment.startsWith(":")) {
      params[patternSegment.slice(1)] = decodeURIComponent(pathSegment);
    } else if (patternSegment !== pathSegment) {
      return null;
    }
  }

  return params;
}

function useHashRoute() {
  const [path, setPath] = useState(getPathFromHash());

  useEffect(() => {
    function handler() {
      setPath(getPathFromHash());
    }

    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, []);

  return matchRoute(path);
}

function Layout({ currentPath, children }) {
  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="header-content">
          <a className="site-title" href="#/">
            ACME Models Registry
          </a>
          <nav>
            <ul className="nav-list">
              {ROUTES.filter((route) => route.label).map((route) => (
                <li key={route.pattern}>
                  <a
                    href={`#${route.pattern}`}
                    aria-current={
                      currentPath === route.pattern ? "page" : undefined
                    }
                  >
                    {route.label}
                  </a>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      </header>
      <main>{children}</main>
    </div>
  );
}

function DashboardPage() {
  const [stats, setStats] = useState(null);
  const [requests, setRequests] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let isMounted = true;

    async function load() {
      try {
        const [statsPayload, requestPayload] = await Promise.all([
          getStats(),
          getIngestRequests(),
        ]);
        if (isMounted) {
          setStats(statsPayload);
          setRequests(requestPayload);
          setError("");
        }
      } catch (err) {
        if (isMounted) {
          setError(err.message || "Unable to load dashboard data.");
        }
      }
    }

    load();
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Registry Overview</h2>
        <p className="lead-text">
          Monitor vetted assets, ingestion pipelines, and license guardrails.
        </p>
      </div>
      {error && <p className="alert alert--warning">{error}</p>}
      <div className="grid grid--stats">
        {[
          ["Total Models", stats?.total_models],
          ["Vetted", stats?.vetted],
          ["Unvetted", stats?.unvetted],
          ["Proprietary", stats?.proprietary],
          ["Public", stats?.public],
        ].map(([label, value]) => (
          <article className="stat-card" key={label}>
            <h3>{label}</h3>
            <p className="stat-value">{value ?? "—"}</p>
          </article>
        ))}
      </div>
      <div className="panel-header">
        <h2>Recent Ingestion Requests</h2>
        <p>Requests must meet minimum metric thresholds before packaging.</p>
      </div>
      {requests.length ? (
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Submitted By</th>
                <th>Status</th>
                <th>Submitted</th>
              </tr>
            </thead>
            <tbody>
              {requests.map((item) => (
                <tr key={item.request_id}>
                  <td>{item.name}</td>
                  <td>{item.submitted_by || "anonymous"}</td>
                  <td>{item.status}</td>
                  <td>
                    {item.submitted_at
                      ? new Date(item.submitted_at * 1000).toLocaleString()
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p role="status">No ingestion requests submitted yet.</p>
      )}
    </section>
  );
}

function ModelDirectoryPage() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [nextOffset, setNextOffset] = useState(null);
  const [searchMode, setSearchMode] = useState(false);

  const loadPage = async ({ append = false } = {}) => {
    setLoading(true);
    setError("");
    try {
      const { items: fetchedItems, nextOffset: headerOffset } =
        await listArtifacts({
          limit: 20,
          offset: append ? nextOffset : null,
        });
      setItems((prev) =>
        append ? [...prev, ...fetchedItems] : fetchedItems || []
      );
      setNextOffset(headerOffset);
      setSearchMode(false);
    } catch (err) {
      setError(err.message || "Unable to load models.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPage();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const runSearch = async (event) => {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) {
      loadPage({ append: false });
      return;
    }
    setLoading(true);
    setError("");
    try {
      const results = await searchArtifactsByRegex(trimmed);
      setItems(results || []);
      setNextOffset(null);
      setSearchMode(true);
    } catch (err) {
      setError(err.message || "Search failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Model Directory</h2>
        <p>Browse ACME-approved and third-party models side-by-side.</p>
      </div>
      <form className="form-inline" onSubmit={runSearch}>
        <label htmlFor="regex-search" className="visually-hidden">
          Regex search
        </label>
        <input
          id="regex-search"
          type="search"
          placeholder="Regex search (e.g. .*safeguard)"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <button type="submit">Search</button>
        <p className="form-help">
          Supports case-insensitive regular expressions across model names and
          cards.
        </p>
      </form>
      {error && <p className="alert alert--warning">{error}</p>}
      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Owner</th>
              <th>Net Score</th>
              <th>License</th>
              <th>Vetted</th>
              <th>Size (MB)</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.id}>
                <td>
                  <a href={`#/models/${encodeURIComponent(item.id)}`}>
                    {item.name}
                  </a>
                </td>
                <td>{item.owner || "—"}</td>
                <td>{formatScore(item.net_score)}</td>
                <td>{item.license?.toUpperCase?.() ?? "—"}</td>
                <td>{item.vetted ? "Yes" : "No"}</td>
                <td>
                  {typeof item.size_mb === "number"
                    ? item.size_mb.toFixed(1)
                    : "—"}
                </td>
                <td>
                  {item.updated_at
                    ? new Date(item.updated_at * 1000).toLocaleString()
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!items.length && !loading && (
          <p role="status">No models matched your search.</p>
        )}
        <div className="pagination-controls">
          <button
            type="button"
            onClick={() => loadPage({ append: true })}
            disabled={loading || searchMode || !nextOffset}
          >
            {loading ? "Loading…" : "Load more"}
          </button>
        </div>
      </div>
    </section>
  );
}

function ModelDetailPage({ params }) {
  const { id } = params;
  const [detail, setDetail] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [lineage, setLineage] = useState(null);
  const [cost, setCost] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let isMounted = true;

    async function load() {
      try {
        const artifactEnvelope = await getArtifact("model", id);
        const flattened = {
          ...artifactEnvelope.metadata,
          ...artifactEnvelope.data,
        };
        if (isMounted) {
          setDetail(flattened);
          setError("");
        }
      } catch (err) {
        if (isMounted) {
          setError(err.message || "Model not found.");
        }
      }
    }

    async function loadExtras() {
      try {
        const [rating, lineagePayload, costPayload] = await Promise.all([
          getArtifactRate(id).catch(() => null),
          getArtifactLineage(id).catch(() => null),
          getArtifactCost(id, { includeDependencies: true }).catch(() => null),
        ]);
        if (isMounted) {
          setMetrics(rating);
          setLineage(lineagePayload);
          setCost(costPayload);
        }
      } catch {
        // Ignore, individual requests already guarded.
      }
    }

    load();
    loadExtras();
    return () => {
      isMounted = false;
    };
  }, [id]);

  if (error) {
    return (
      <section className="panel">
        <h2>Model Details</h2>
        <p className="alert alert--warning">{error}</p>
      </section>
    );
  }

  if (!detail) {
    return (
      <section className="panel">
        <h2>Model Details</h2>
        <p>Loading model details…</p>
      </section>
    );
  }

  const costSummary = useMemo(() => {
    if (!cost || typeof cost !== "object") return null;
    const root = cost[id] || Object.values(cost)[0];
    if (!root) return null;
    const entries = Object.entries(cost).filter(
      ([key]) => key !== id && cost[key]
    );
    return { root, dependencies: entries };
  }, [cost, id]);

  return (
    <>
      <section className="panel">
        <div className="panel-header">
          <h2>{detail.name}</h2>
          <p>{detail.description || detail.card_excerpt}</p>
        </div>
        <dl className="detail-grid">
          <div>
            <dt>Model ID</dt>
            <dd>
              <code>{detail.id}</code>
            </dd>
          </div>
          <div>
            <dt>Owner</dt>
            <dd>{detail.owner || "—"}</dd>
          </div>
          <div>
            <dt>Vetted Status</dt>
            <dd>
              {detail.vetted ? (
                <span className="status status--success">Vetted</span>
              ) : (
                <span className="status status--warning">Pending review</span>
              )}
            </dd>
          </div>
          <div>
            <dt>License</dt>
            <dd>{detail.license?.toUpperCase?.() ?? "—"}</dd>
          </div>
          <div>
            <dt>Net Score</dt>
            <dd>
              <span className="metric-highlight">
                {formatScore(metrics?.net_score)}
              </span>
            </dd>
          </div>
          <div>
            <dt>Model Card</dt>
            <dd>
              {detail.card_url ? (
                <a href={detail.card_url} target="_blank" rel="noreferrer">
                  Open card
                </a>
              ) : (
                "—"
              )}
            </dd>
          </div>
          <div>
            <dt>Artifact Size</dt>
            <dd>
              {typeof detail.size_mb === "number"
                ? `${detail.size_mb.toFixed(1)} MB`
                : "—"}
            </dd>
          </div>
          <div>
            <dt>Tags</dt>
            <dd>
              {(detail.tags || []).map((tag) => (
                <span className="chip" key={tag}>
                  {tag}
                </span>
              ))}
            </dd>
          </div>
          <div>
            <dt>Updated</dt>
            <dd>
              {detail.updated_at
                ? new Date(detail.updated_at * 1000).toLocaleString()
                : "—"}
            </dd>
          </div>
        </dl>
      </section>
      <section className="panel">
        <div className="panel-header">
          <h2>Quality Metrics</h2>
        </div>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Score</th>
                <th>Latency</th>
              </tr>
            </thead>
            <tbody>
              {metrics ? (
                metricRows(metrics).map(([label, value, latency]) => (
                  <tr key={label}>
                    <td>{label}</td>
                    <td>{value}</td>
                    <td>{latency}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={3}>Metrics are not available.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
      <section className="panel">
        <div className="panel-header">
          <h2>Lineage Graph</h2>
          <p>Relationships are derived from config metadata and ingestion history.</p>
        </div>
        <div className="lineage" aria-live="polite">
          {lineage && lineage.nodes?.length ? (
            <ul className="lineage-list">
              {lineage.nodes.map((node) => (
                <li key={node.artifact_id || node.id}>
                  <strong>{node.name}</strong> —{" "}
                  {(node.metadata?.license || node.license || "unknown").toUpperCase()}{" "}
                  • {node.metadata?.vetted || node.vetted ? "vetted" : "unvetted"}
                </li>
              ))}
            </ul>
          ) : (
            <p>No lineage information available.</p>
          )}
        </div>
      </section>
      <section className="panel">
        <div className="panel-header">
          <h2>Size Cost Estimate</h2>
        </div>
        <div className="size-cost" aria-live="polite">
          {costSummary ? (
            <>
              <p>
                Total download cost:{" "}
                {formatScore(costSummary.root.total_cost)} MB
              </p>
              {typeof costSummary.root.standalone_cost === "number" && (
                <p>
                  Standalone cost:{" "}
                  {formatScore(costSummary.root.standalone_cost)} MB
                </p>
              )}
              {costSummary.dependencies.length > 0 && (
                <div>
                  <h3>Dependencies</h3>
                  <ul className="lineage-list">
                    {costSummary.dependencies.map(([depId, dep]) => (
                      <li key={depId}>
                        {depId}: {formatScore(dep.total_cost)} MB
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          ) : (
            <p>No size information available.</p>
          )}
        </div>
      </section>
    </>
  );
}

function IngestPage() {
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [submittedBy, setSubmittedBy] = useState("");
  const [cliText, setCliText] = useState("");
  const [cliSummary, setCliSummary] = useState(
    "Paste CLI output to populate metrics automatically."
  );
  const [cliError, setCliError] = useState("");
  const [parsedCli, setParsedCli] = useState(null);
  const [status, setStatus] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!cliText.trim()) {
      setCliSummary("Paste CLI output to populate metrics automatically.");
      setCliError("");
      setParsedCli(null);
      return;
    }
    try {
      const parsed = parseCliResults(cliText, name);
      setParsedCli(parsed);
      const parts = [];
      if (parsed.record.name) {
        parts.push(`Loaded metrics for ${parsed.record.name}.`);
      }
      if (parsed.metrics.net_score !== undefined) {
        parts.push(`Net score ${formatScore(parsed.metrics.net_score)}.`);
      }
      if (parsed.totalRecords > 1) {
        parts.push(`(${parsed.totalRecords} records detected.)`);
      }
      setCliSummary(parts.join(" "));
      setCliError("");
    } catch (err) {
      setCliError(err.message);
      setParsedCli(null);
    }
  }, [cliText, name]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!parsedCli) {
      setCliError("Valid CLI output is required.");
      return;
    }
    if (!url.trim()) {
      setStatus("A HuggingFace URL is required.");
      return;
    }

    const payload = {
      name: name.trim() || parsedCli.record.name || "unnamed-model",
      url: url.trim(),
      download_url: url.trim(),
      owner: submittedBy.trim() || "external",
      submitted_by: submittedBy.trim() || undefined,
      metrics: parsedCli.metrics,
    };

    if (parsedCli.record.category) {
      payload.tags = [parsedCli.record.category];
    }

    setSubmitting(true);
    setStatus("Submitting request…");
    try {
      const response = await createArtifact(payload);
      const artifactId = response?.metadata?.id ?? "unknown";
      setStatus(`Artifact registered. ID: ${artifactId}`);
      setName("");
      setUrl("");
      setSubmittedBy("");
      setCliText("");
      setParsedCli(null);
    } catch (err) {
      setStatus(err.message || "Unable to submit request.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Request Model Ingestion</h2>
        <p>Submit a public HuggingFace model for review.</p>
      </div>
      <form onSubmit={handleSubmit}>
        <label htmlFor="ingest-name">Model name</label>
        <input
          id="ingest-name"
          type="text"
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="e.g. google-bert/bert-base-uncased"
        />
        <label htmlFor="ingest-url">HuggingFace URL</label>
        <input
          id="ingest-url"
          type="url"
          value={url}
          onChange={(event) => setUrl(event.target.value)}
          placeholder="https://huggingface.co/..."
          required
        />
        <label htmlFor="ingest-submitted">Submitted by</label>
        <input
          id="ingest-submitted"
          type="text"
          value={submittedBy}
          onChange={(event) => setSubmittedBy(event.target.value)}
          placeholder="Your ACME username"
        />
        <label htmlFor="cli-output">CLI output</label>
        <textarea
          id="cli-output"
          rows={6}
          value={cliText}
          onChange={(event) => setCliText(event.target.value)}
          placeholder='{"name":"my-model","net_score":0.92,...}'
          required
        />
        <p className={`form-help ${cliError ? "form-help--error" : ""}`}>
          {cliError || cliSummary}
        </p>
        <button type="submit" disabled={submitting}>
          {submitting ? "Submitting…" : "Submit request"}
        </button>
      </form>
      {status && <p className="status-message">{status}</p>}
    </section>
  );
}

function LicensePage() {
  const [artifactId, setArtifactId] = useState("");
  const [githubUrl, setGithubUrl] = useState("");
  const [status, setStatus] = useState("");
  const [checking, setChecking] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!artifactId.trim() || !githubUrl.trim()) {
      setStatus("Artifact ID and GitHub URL are required.");
      return;
    }
    setChecking(true);
    setStatus("Checking license compatibility…");
    try {
      const result = await licenseCheck(artifactId.trim(), githubUrl.trim());
      const verdict = result ? "Compatible" : "Not compatible";
      setStatus(`Result: ${verdict}`);
    } catch (err) {
      setStatus(err.message || "Unable to check compatibility.");
    } finally {
      setChecking(false);
    }
  };

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>License Compatibility</h2>
        <p>
          Evaluate whether fine-tuning is permissible for a model artifact
          against a GitHub repository.
        </p>
      </div>
      <form onSubmit={handleSubmit}>
        <label htmlFor="license-artifact">Artifact ID</label>
        <input
          id="license-artifact"
          type="text"
          value={artifactId}
          onChange={(event) => setArtifactId(event.target.value)}
          placeholder="acme/solar-safeguard"
          required
        />
        <label htmlFor="license-github">GitHub repository URL</label>
        <input
          id="license-github"
          type="url"
          value={githubUrl}
          onChange={(event) => setGithubUrl(event.target.value)}
          placeholder="https://github.com/org/project"
          required
        />
        <button type="submit" disabled={checking}>
          {checking ? "Checking…" : "Check compatibility"}
        </button>
      </form>
      {status && <p className="status-message">{status}</p>}
    </section>
  );
}

function ResetPage() {
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  const handleReset = async () => {
    setLoading(true);
    setStatus("Resetting registry…");
    try {
      await resetRegistry();
      setStatus("Registry restored to defaults.");
    } catch (err) {
      setStatus(err.message || "Reset failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Admin: Reset Registry</h2>
        <p>Restores the prototype datastore to the default records.</p>
      </div>
      <button type="button" onClick={handleReset} disabled={loading}>
        {loading ? "Resetting…" : "Reset registry"}
      </button>
      {status && <p className="status-message">{status}</p>}
    </section>
  );
}

function metricRows(metrics) {
  if (!metrics || typeof metrics !== "object") return [];
  const entries = [];
  Object.keys(metrics)
    .filter((key) => !key.endsWith("_latency"))
    .forEach((key) => {
      const label = key.replace(/_/g, " ");
      const value = formatScore(metrics[key]);
      const latency = formatLatency(metrics[`${key}_latency`]);
      if (metrics[key] && typeof metrics[key] === "object" && !Array.isArray(metrics[key])) {
        entries.push([label, "", latency]);
        Object.entries(metrics[key]).forEach(([innerKey, innerValue]) => {
          entries.push([
            `— ${innerKey.replace(/_/g, " ")}`,
            formatScore(innerValue),
            "",
          ]);
        });
      } else {
        entries.push([label, value, latency]);
      }
    });
  return entries;
}

function formatScore(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return String(value);
  }
  return numeric.toFixed(2);
}

function formatLatency(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return String(value);
  }
  return `${Math.round(numeric)} ms`;
}

function App() {
  const route = useHashRoute();
  const PageComponent = route.component || DashboardPage;

  return (
    <Layout currentPath={route.pattern}>
      <PageComponent params={route.params || {}} />
    </Layout>
  );
}

export default App;
