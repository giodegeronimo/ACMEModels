import { useEffect, useMemo, useState } from "react";
import {
  createArtifact,
  getArtifact,
  getArtifactCost,
  getArtifactLineage,
  getArtifactRate,
  fetchAllArtifacts,
  getTracks,
  licenseCheck,
  listArtifacts,
  resetRegistry,
  searchArtifactsByRegex,
  authenticate,
  getAuthToken,
  resetAuthTokenToDefault,
} from "./api";
import "./App.css";

const ROUTES = [
  { pattern: "/", component: DashboardPage, label: "Dashboard" },
  { pattern: "/models", component: ModelDirectoryPage, label: "Artifacts" },
  { pattern: "/models/:type/:id", component: ModelDetailPage },
  { pattern: "/models/:id", component: ModelDetailPage },
  { pattern: "/ingest", component: IngestPage, label: "Ingest" },
  { pattern: "/auth", component: AuthPage, label: "Auth" },
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
        <a className="skip-link" href="#main-content">
          Skip to main content
        </a>
        <div className="header-content">
          <a className="site-title" href="#/" aria-label="ACME Models Registry home">
            ACME Models Registry
          </a>
          <nav aria-label="Primary">
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
      <main id="main-content">{children}</main>
    </div>
  );
}

function DashboardPage() {
  const [stats, setStats] = useState(null);
  const [recentArtifacts, setRecentArtifacts] = useState([]);
  const [tracks, setTracks] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let isMounted = true;

    async function load() {
      try {
        const [allArtifacts, plannedTracks] = await Promise.all([
          fetchAllArtifacts({ pageSize: 50 }),
          getTracks().catch(() => []),
        ]);
        if (isMounted) {
          setStats(summarizeArtifacts(allArtifacts));
          setRecentArtifacts(allArtifacts.slice(0, 10));
          setTracks(plannedTracks || []);
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
      <h1 className="visually-hidden">Dashboard</h1>
      <div className="panel-header">
        <h2>Registry Overview</h2>
        <p className="lead-text">
          Snapshot of artifacts available from the deployed AWS backend.
        </p>
      </div>
      {error && (
        <p className="alert alert--warning" role="alert">
          {error}
        </p>
      )}
      <div className="grid grid--stats">
        {[
          ["Total Artifacts", stats?.total],
          ["Models", stats?.models],
          ["Datasets", stats?.datasets],
          ["Code", stats?.code],
        ].map(([label, value]) => (
          <article className="stat-card" key={label}>
            <h3>{label}</h3>
            <p className="stat-value">{value ?? "—"}</p>
          </article>
        ))}
      </div>
      <div className="panel-header">
        <h2>Recent Artifacts</h2>
        <p>Latest entries returned by the registry API.</p>
      </div>
      {recentArtifacts.length ? (
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Artifact ID</th>
              </tr>
            </thead>
            <tbody>
              {recentArtifacts.map((item) => (
                <tr key={item.id}>
                  <td>
                    <a
                      className="text-clip"
                      href={`#/models/${encodeURIComponent(
                        item.type || "model"
                      )}/${encodeURIComponent(item.id)}`}
                      title={item.name}
                    >
                      {item.name}
                    </a>
                  </td>
                  <td>{item.type}</td>
                  <td>
                    <code>{item.id}</code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p role="status">No artifacts available yet.</p>
      )}
      {tracks.length > 0 && (
        <>
          <div className="panel-header">
            <h2>Planned Tracks</h2>
            <p>Returned directly from the backend /tracks endpoint.</p>
          </div>
          <ul className="lineage-list">
            {tracks.map((track) => (
              <li key={track}>{track}</li>
            ))}
          </ul>
        </>
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
      <h1 className="visually-hidden">Artifact Directory</h1>
      <div className="panel-header">
        <h2>Artifact Directory</h2>
        <p>Browse artifacts (models, datasets and codebases) currently available in the registry.</p>
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
      {error && (
        <p className="alert alert--warning" role="alert">
          {error}
        </p>
      )}
      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Artifact ID</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.id}>
                <td>
                  <a
                    className="text-clip"
                    href={`#/models/${encodeURIComponent(
                      item.type || "model"
                    )}/${encodeURIComponent(item.id)}`}
                    title={item.name}
                  >
                    {item.name}
                  </a>
                </td>
                <td>{item.type}</td>
                <td>
                  <code>{item.id}</code>
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
  const { id, type } = params;
  const artifactType = type || "model";
  const [detail, setDetail] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [lineage, setLineage] = useState(null);
  const [cost, setCost] = useState(null);
  const [error, setError] = useState("");

  const costSummary = useMemo(() => {
    if (!cost || typeof cost !== "object") return null;
    const root = cost[id] || Object.values(cost)[0];
    if (!root) return null;
    const entries = Object.entries(cost).filter(
      ([key]) => key !== id && cost[key]
    );
    return { root, dependencies: entries };
  }, [cost, id]);

  const detailSummary = useMemo(
    () =>
      detail?.description ||
      detail?.card_excerpt ||
      (detail?.url
        ? `Source: ${detail.url}`
        : "Artifact metadata provided by the registry."),
    [detail]
  );

  useEffect(() => {
    let isMounted = true;

    async function load() {
      try {
        const artifactEnvelope = await getArtifact(artifactType, id);
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
          setError(err.message || "Artifact not found.");
        }
      }
    }

    async function loadExtras() {
      if (artifactType !== "model") {
        setMetrics(null);
        setLineage(null);
        setCost(null);
        return;
      }
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
  }, [id, artifactType]);

  if (error) {
    return (
      <section className="panel">
        <h1>Artifact Details</h1>
        <p className="alert alert--warning" role="alert">
          {error}
        </p>
      </section>
    );
  }

  if (!detail) {
    return (
      <section className="panel">
        <h1>Artifact Details</h1>
        <p role="status" aria-live="polite">
          Loading artifact details…
        </p>
      </section>
    );
  }

  return (
    <>
      <section className="panel">
        <div className="panel-header">
          <h1>{detail.name}</h1>
          <p>{detailSummary}</p>
        </div>
        <dl className="detail-grid">
          <div>
            <dt>Artifact ID</dt>
            <dd>
              <code>{detail.id}</code>
            </dd>
          </div>
          <div>
            <dt>Type</dt>
            <dd>{detail.type || artifactType}</dd>
          </div>
          <div>
            <dt>Source URL</dt>
            <dd>
              {detail.url ? (
                <a href={detail.url} target="_blank" rel="noreferrer">
                  {detail.url}
                </a>
              ) : (
                "—"
              )}
            </dd>
          </div>
          <div>
            <dt>Download</dt>
            <dd>
              {detail.download_url ? (
                <a
                  href={detail.download_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  Download from registry
                </a>
              ) : (
                "—"
              )}
            </dd>
          </div>
        </dl>
      </section>
      {artifactType === "model" && (
        <>
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
      )}
    </>
  );
}

function IngestPage() {
  const [artifactType, setArtifactType] = useState("model");
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [status, setStatus] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!url.trim()) {
      setStatus("A source URL is required.");
      return;
    }

    const payload = {
      url: url.trim(),
    };
    if (name.trim()) {
      payload.name = name.trim();
    }

    setSubmitting(true);
    setStatus("Submitting artifact…");
    try {
      const response = await createArtifact(artifactType, payload);
      const artifactId = response?.metadata?.id ?? "unknown";
      setStatus(
        `Artifact registered. Type=${artifactType} ID=${artifactId}`
      );
      setName("");
      setUrl("");
    } catch (err) {
      setStatus(err.message || "Unable to submit request.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="panel">
      <h1>Artifact Ingest</h1>
      <div className="panel-header">
        <h2>Submit new artifact</h2>
        <p>
          Use this page to add either a model, dataset, or codebase to the registry.
        </p>
      </div>
      <form onSubmit={handleSubmit}>
        <label htmlFor="ingest-type">Artifact type</label>
        <select
          id="ingest-type"
          value={artifactType}
          onChange={(event) => setArtifactType(event.target.value)}
        >
          <option value="model">model</option>
          <option value="dataset">dataset</option>
          <option value="code">code</option>
        </select>

        <label htmlFor="ingest-name">Artifact name (optional)</label>
        <input
          id="ingest-name"
          type="text"
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="e.g. google-bert/bert-base-uncased"
        />

        <label htmlFor="ingest-url">Source URL</label>
        <input
          id="ingest-url"
          type="url"
          value={url}
          onChange={(event) => setUrl(event.target.value)}
          placeholder="https://huggingface.co/... or https://github.com/..."
          required
        />

        <p className="form-help">
          We will assign an ID and ingest the artifact from the
          provided URL.
        </p>

        <button type="submit" disabled={submitting}>
          {submitting ? "Submitting…" : "Submit artifact"}
        </button>
      </form>
      {status && (
        <p className="status-message" role="status" aria-live="polite">
          {status}
        </p>
      )}
    </section>
  );
}

function AuthPage() {
  const [username, setUsername] = useState("ece30861defaultadminuser");
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState("");
  const [token, setToken] = useState(getAuthToken() || "");

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!username.trim() || !password.trim()) {
      setStatus("Username and password are required.");
      return;
    }
    setStatus("Authenticating…");
    try {
      const newToken = await authenticate(username.trim(), password.trim());
      setToken(newToken || "");
      setStatus("Authenticated. Token applied to future requests.");
      setPassword("");
    } catch (err) {
      setStatus(err.message || "Authentication failed.");
    }
  };

  const handleReset = () => {
    resetAuthTokenToDefault();
    const defaultToken = getAuthToken() || "";
    setToken(defaultToken);
    setStatus("Reverted to default token.");
  };

  return (
    <section className="panel">
      <h1>Authenticate</h1>
      <div className="panel-header">
        <h2>Request token</h2>
        <p>
          Obtain an authorization token via <code>PUT /authenticate</code> and
          apply it to all API calls.
        </p>
      </div>
      <form onSubmit={handleSubmit}>
        <label htmlFor="auth-username">Username</label>
        <input
          id="auth-username"
          type="text"
          value={username}
          onChange={(event) => setUsername(event.target.value)}
          placeholder="ece30861defaultadminuser"
        />
        <label htmlFor="auth-password">Password</label>
        <input
          id="auth-password"
          type="password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          placeholder="Enter admin password"
        />
        <button type="submit">Authenticate</button>
      </form>
      <div className="status-message" aria-live="polite" role="status">
        <p>
          <strong>Current token:</strong>{" "}
          {token ? (
            <code className="text-clip" title={token}>
              {token}
            </code>
          ) : (
            "not set"
          )}
        </p>
        <button type="button" onClick={handleReset}>
          Use default token
        </button>
      </div>
      {status && (
        <p className="status-message" role="status" aria-live="polite">
          {status}
        </p>
      )}
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
      <h1>License Compatibility</h1>
      <div className="panel-header">
        <h2>Check compatibility</h2>
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
      {status && (
        <p className="status-message" role="status" aria-live="polite">
          {status}
        </p>
      )}
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
      <h1>Reset Registry</h1>
      <div className="panel-header">
        <h2>Admin: Reset Registry</h2>
        <p>Restores the prototype datastore to the default records.</p>
      </div>
      <button type="button" onClick={handleReset} disabled={loading}>
        {loading ? "Resetting…" : "Reset registry"}
      </button>
      {status && (
        <p className="status-message" role="status" aria-live="polite">
          {status}
        </p>
      )}
    </section>
  );
}

function summarizeArtifacts(artifacts) {
  const summary = {
    total: 0,
    models: 0,
    datasets: 0,
    code: 0,
  };
  if (!Array.isArray(artifacts)) {
    return summary;
  }
  summary.total = artifacts.length;
  artifacts.forEach((artifact) => {
    if (artifact.type === "model") summary.models += 1;
    else if (artifact.type === "dataset") summary.datasets += 1;
    else if (artifact.type === "code") summary.code += 1;
  });
  return summary;
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
