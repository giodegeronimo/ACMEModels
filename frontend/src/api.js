const DEFAULT_AUTH_TOKEN =
  import.meta.env.VITE_API_TOKEN ||
  import.meta.env.VITE_AUTH_TOKEN ||
  "dev-token";

const AUTH_STORAGE_KEY = "acme_auth_token";

let authToken =
  (typeof localStorage !== "undefined" &&
    localStorage.getItem(AUTH_STORAGE_KEY)) ||
  DEFAULT_AUTH_TOKEN;

const BASE_HEADERS = {
  Accept: "application/json",
  "Content-Type": "application/json",
};

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "/api"
);

export function setAuthToken(token) {
  authToken = token || "";
  if (typeof localStorage !== "undefined") {
    if (token) {
      localStorage.setItem(AUTH_STORAGE_KEY, token);
    } else {
      localStorage.removeItem(AUTH_STORAGE_KEY);
    }
  }
}

export function getAuthToken() {
  return authToken || null;
}

export function resetAuthTokenToDefault() {
  setAuthToken(DEFAULT_AUTH_TOKEN);
}

function resolveUrl(path) {
  if (API_BASE_URL) {
    return `${API_BASE_URL}${path}`;
  }
  return path;
}

async function request(path, options = {}) {
  const headers = {
    ...BASE_HEADERS,
    ...(authToken
      ? { "X-Authorization": authToken, Authorization: authToken }
      : {}),
    ...(options.headers || {}),
  };

  const response = await fetch(resolveUrl(path), {
    headers,
    ...options,
  });

  let payload;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    payload = await response.json();
  } else {
    payload = await response.text();
  }

  if (!response.ok) {
    const message =
      (payload && payload.error) ||
      (typeof payload === "string" && payload) ||
      response.statusText;
    throw new Error(message || "Request failed");
  }

  return { data: payload, response };
}

export async function listArtifacts({ limit = 20, offset = null } = {}) {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  if (offset !== null && offset !== undefined) {
    params.set("offset", String(offset));
  }

  const { data, response } = await request(
    `/artifacts?${params.toString()}`,
    {
      method: "POST",
      body: JSON.stringify([{ name: "*" }]),
    }
  );

  const nextHeader = response.headers.get("offset");
  const nextOffset = nextHeader || null;

  return { items: data || [], nextOffset };
}

export async function searchArtifactsByRegex(regex) {
  const { data } = await request("/artifact/byRegEx", {
    method: "POST",
    body: JSON.stringify({ regex }),
  });
  return data || [];
}

export async function getArtifact(artifactType, artifactId) {
  const safeId = encodeURIComponent(artifactId);
  const { data } = await request(`/artifacts/${artifactType}/${safeId}`);
  return data;
}

export async function createArtifact(artifactType, payload) {
  const safeType = encodeURIComponent(artifactType);
  const { data } = await request(`/artifact/${safeType}`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return data;
}

export async function getArtifactRate(artifactId) {
  const safeId = encodeURIComponent(artifactId);
  const { data } = await request(`/artifact/model/${safeId}/rate`);
  return data;
}

export async function getArtifactLineage(artifactId) {
  const safeId = encodeURIComponent(artifactId);
  const { data } = await request(`/artifact/model/${safeId}/lineage`);
  return data;
}

export async function getArtifactCost(artifactId, { includeDependencies = true } = {}) {
  const safeId = encodeURIComponent(artifactId);
  const params = new URLSearchParams();
  if (includeDependencies) params.set("dependency", "true");
  const { data } = await request(
    `/artifact/model/${safeId}/cost?${params.toString()}`
  );
  return data;
}

export async function licenseCheck(artifactId, githubUrl) {
  const safeId = encodeURIComponent(artifactId);
  const { data } = await request(`/artifact/model/${safeId}/license-check`, {
    method: "POST",
    body: JSON.stringify({ github_url: githubUrl }),
  });
  return data;
}

export async function resetRegistry() {
  const { data } = await request("/reset", { method: "DELETE" });
  return data;
}

export async function getStats() {
  const artifacts = await fetchAllArtifacts();
  const summary = {
    total: artifacts.length,
    models: 0,
    datasets: 0,
    code: 0,
  };
  artifacts.forEach((artifact) => {
    if (artifact.type === "model") summary.models += 1;
    else if (artifact.type === "dataset") summary.datasets += 1;
    else if (artifact.type === "code") summary.code += 1;
  });
  return summary;
}

export async function getRecentArtifacts(limit = 10) {
  const { items } = await listArtifacts({ limit });
  return items || [];
}

export async function fetchAllArtifacts({
  pageSize = 100,
  maxPages = 25,
} = {}) {
  const collected = [];
  let offset = null;
  let page = 0;

  while (page < maxPages) {
    const { items, nextOffset } = await listArtifacts({
      limit: pageSize,
      offset,
    });
    collected.push(...(items || []));
    if (!nextOffset) {
      break;
    }
    offset = nextOffset;
    page += 1;
  }

  return collected;
}

export async function getTracks() {
  const { data } = await request("/tracks");
  return data?.plannedTracks || [];
}

export async function authenticate(username, password) {
  const payload = {
    user: { name: username },
    secret: { password },
  };
  const { data } = await request("/authenticate", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
  if (typeof data === "string") {
    setAuthToken(data);
  }
  return data;
}
