import { request } from "./apiClient";

export async function createRecognitionRun(payload) {
  return request("/api/recognitions", {
    method: "POST",
    body: payload,
  });
}

export async function fetchRecognitionRuns(query = {}) {
  const searchParams = new URLSearchParams();

  if (query.userEmail) {
    searchParams.set("userEmail", query.userEmail);
  }

  if (query.sourceMode) {
    searchParams.set("sourceMode", query.sourceMode);
  }

  if (query.signLanguage) {
    searchParams.set("signLanguage", query.signLanguage);
  }

  if (query.recognitionLevel) {
    searchParams.set("recognitionLevel", query.recognitionLevel);
  }

  const suffix = searchParams.size ? `?${searchParams.toString()}` : "";
  return request(`/api/recognitions${suffix}`);
}
