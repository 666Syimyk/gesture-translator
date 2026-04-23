const DEFAULT_API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";
const API_BASE_URL_STORAGE_KEY = "gesture_translator_api_base_url";

function normalizeApiBaseUrl(value) {
  const raw = String(value ?? "").trim();
  if (!raw) {
    return "";
  }

  return raw.endsWith("/") ? raw.slice(0, -1) : raw;
}

export function getApiBaseUrl() {
  if (typeof window === "undefined") {
    return normalizeApiBaseUrl(DEFAULT_API_BASE_URL);
  }

  const fromWindow =
    window.__APP_CONFIG?.API_BASE_URL ??
    window.__APP_CONFIG?.apiBaseUrl ??
    "";
  const fromStorage = window.localStorage?.getItem(API_BASE_URL_STORAGE_KEY) ?? "";

  return normalizeApiBaseUrl(fromStorage || fromWindow || DEFAULT_API_BASE_URL);
}

export function setApiBaseUrl(value) {
  if (typeof window === "undefined") {
    return;
  }

  const normalized = normalizeApiBaseUrl(value);
  if (!normalized) {
    window.localStorage?.removeItem(API_BASE_URL_STORAGE_KEY);
    return;
  }

  window.localStorage?.setItem(API_BASE_URL_STORAGE_KEY, normalized);
}

export function buildApiUrl(path) {
  const API_BASE_URL = getApiBaseUrl();

  if (!path) {
    return API_BASE_URL;
  }

  if (/^https?:\/\//i.test(path)) {
    return path;
  }

  return `${API_BASE_URL}${path}`;
}

export async function request(path, options = {}) {
  const isFormData =
    typeof FormData !== "undefined" && options.body instanceof FormData;
  const isSerializableJsonBody =
    options.body !== undefined &&
    options.body !== null &&
    !isFormData &&
    typeof options.body !== "string";
  const body = isSerializableJsonBody
    ? JSON.stringify(options.body)
    : options.body;

  const response = await fetch(buildApiUrl(path), {
    ...options,
    body,
    headers: {
      ...(isFormData ? {} : { "Content-Type": "application/json" }),
      ...(options.headers ?? {}),
    },
  });

  if (!response.ok) {
    let message = "Request failed";

    try {
      const errorBody = await response.json();
      message = errorBody.error || message;
    } catch {
      // Keep the default message if the response body is not JSON.
    }

    throw new Error(message);
  }

  if (response.status === 204) {
    return null;
  }

  return response.json();
}
