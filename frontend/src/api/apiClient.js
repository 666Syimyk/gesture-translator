const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export function buildApiUrl(path) {
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
