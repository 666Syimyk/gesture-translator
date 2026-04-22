import { request } from "./apiClient";
import { DEFAULT_MODEL_SCOPE } from "../recognitionLevels";

export async function fetchLatestModel(
  recognitionLevel = DEFAULT_MODEL_SCOPE,
  options = {},
) {
  const params = new URLSearchParams();

  if (recognitionLevel) {
    params.set("recognitionLevel", recognitionLevel);
  }

  if (Array.isArray(options.allowedRecognitionLevels)) {
    const normalizedLevels = options.allowedRecognitionLevels.filter(Boolean);

    if (normalizedLevels.length) {
      params.set("allowedRecognitionLevels", normalizedLevels.join(","));
    }
  }

  if (options.profile) {
    params.set("profile", options.profile);
  }

  const query = params.size ? `?${params.toString()}` : "";
  return request(`/api/ml/model/latest${query}`);
}

export async function trainLatestModel({
  recognitionLevel = "phrase",
  modelType = "baseline",
  epochs,
  maxSequenceLength,
  hiddenSize,
  classBalance,
  classWeightPower,
  focusLabelKeys,
  focusWeightMultiplier,
  confidenceThreshold = 0.85,
} = {}) {
  return request("/api/ml/train", {
    method: "POST",
    body: {
      recognitionLevel,
      modelType,
      epochs,
      maxSequenceLength,
      hiddenSize,
      classBalance,
      classWeightPower,
      focusLabelKeys,
      focusWeightMultiplier,
      confidenceThreshold,
    },
  });
}

export async function fetchLatestModelEvaluation(recognitionLevel = "phrase") {
  const query = `?recognitionLevel=${encodeURIComponent(recognitionLevel)}`;
  return request(`/api/ml/evaluation/latest${query}`);
}

export async function evaluateLatestModel({
  recognitionLevel = "phrase",
  confidenceThreshold = 0.85,
} = {}) {
  return request("/api/ml/evaluate", {
    method: "POST",
    body: {
      recognitionLevel,
      confidenceThreshold,
    },
  });
}

export async function fetchLatestBenchmark(recognitionLevel = "phrase") {
  const query = `?recognitionLevel=${encodeURIComponent(recognitionLevel)}`;
  return request(`/api/ml/benchmark/latest${query}`);
}

export async function runModelBenchmark({
  recognitionLevel = "phrase",
  modelTypes = ["baseline", "gru", "lstm", "tcn"],
  confidenceThreshold = 0.85,
} = {}) {
  return request("/api/ml/benchmark", {
    method: "POST",
    body: {
      recognitionLevel,
      modelTypes,
      confidenceThreshold,
    },
  });
}

export async function fetchModelRuns(
  signLanguage,
  limit = 15,
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
  return request(`/api/model-runs${query}`);
}

export async function predictLatestModel({
  recognitionLevel = DEFAULT_MODEL_SCOPE,
  profile,
  trainingVideoId,
  landmarkPath,
  sequence,
  allowedRecognitionLevels,
  allowedLabelKeys,
} = {}) {
  return request("/api/ml/predict", {
    method: "POST",
    body: {
      recognitionLevel,
      profile,
      trainingVideoId,
      landmarkPath,
      sequence,
      allowedRecognitionLevels,
      allowedLabelKeys,
    },
  });
}
