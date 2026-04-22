import { request } from "./apiClient";

export async function prepareDatasetExport({
  signLanguage,
  recognitionLevel = "phrase",
  trainRatio = 0.7,
  valRatio = 0.15,
  testRatio = 0.15,
} = {}) {
  return request("/api/dataset/export", {
    method: "POST",
    body: {
      signLanguage,
      recognitionLevel,
      trainRatio,
      valRatio,
      testRatio,
    },
  });
}

export async function fetchLatestDatasetExport(recognitionLevel = "phrase") {
  const query = `?recognitionLevel=${encodeURIComponent(recognitionLevel)}`;
  return request(`/api/dataset/export/latest${query}`);
}

export async function fetchDatasetAnalytics(signLanguage, recognitionLevel = "phrase") {
  const params = new URLSearchParams();

  if (signLanguage) {
    params.set("signLanguage", signLanguage);
  }

  if (recognitionLevel) {
    params.set("recognitionLevel", recognitionLevel);
  }

  const query = params.size ? `?${params.toString()}` : "";
  return request(`/api/dataset/analytics${query}`);
}

export async function fetchDatasetExports(
  signLanguage,
  limit = 10,
  recognitionLevel = "phrase",
) {
  const params = new URLSearchParams();

  if (signLanguage) {
    params.set("signLanguage", signLanguage);
  }

  if (recognitionLevel) {
    params.set("recognitionLevel", recognitionLevel);
  }

  params.set("limit", String(limit));
  const query = `?${params.toString()}`;
  return request(`/api/dataset/exports${query}`);
}
