function normalizeValue(value) {
  if (value === null || value === undefined) {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((entry) => normalizeValue(entry));
  }
  if (typeof value === "object") {
    const normalized = {};
    for (const [key, nestedValue] of Object.entries(value)) {
      normalized[key] = normalizeValue(nestedValue);
    }
    return normalized;
  }
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed === "") return value;
    const numeric = Number(trimmed);
    if (!Number.isNaN(numeric) && Number.isFinite(numeric)) {
      return numeric;
    }
  }
  return value;
}

export function parseCliResults(rawText, preferredName = "") {
  const lines = (rawText || "")
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
    } catch {
      throw new Error("CLI output must be valid JSON (NDJSON).");
    }
  }

  if (!records.length) {
    throw new Error("No valid records found in CLI output.");
  }

  const preferred = preferredName ? preferredName.trim().toLowerCase() : "";
  let selected = records[0];
  if (preferred) {
    const match = records.find(
      (record) =>
        typeof record.name === "string" &&
        record.name.trim().toLowerCase() === preferred
    );
    if (match) {
      selected = match;
    }
  }

  const metrics = {};
  for (const [key, value] of Object.entries(selected)) {
    if (key === "name" || key === "category") continue;
    metrics[key] = normalizeValue(value);
  }

  return {
    record: selected,
    metrics,
    totalRecords: records.length,
  };
}
