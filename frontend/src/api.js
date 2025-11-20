const JSON_HEADERS = {
  Accept: "application/json",
  "Content-Type": "application/json",
  "X-Authorization": "dev-token",
};

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "/api"
);

function resolveUrl(path) {
  if (API_BASE_URL) {
    return `${API_BASE_URL}${path}`;
  }
  return path;
}

async function request(path, options = {}) {
  const response = await fetch(resolveUrl(path), {
    headers: JSON_HEADERS,
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
  const nextOffset =
    nextHeader !== null && nextHeader !== undefined
      ? Number.parseInt(nextHeader, 10)
      : null;

  return { items: data || [], nextOffset: Number.isNaN(nextOffset) ? null : nextOffset };
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

export async function createArtifact(payload) {
  const { data } = await request("/artifact/model", {
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
  const { data } = await request("/stats");
  return data;
}

export async function getIngestRequests() {
  const { data } = await request("/ingest/requests");
  return data.items || [];
}
