import { buildApiUrl, request } from "./apiClient";

function formatTrainingVideo(item) {
  const createdAt = item.created_at ?? item.createdAt ?? new Date().toISOString();
  const videoUrlPath = item.video_url ?? item.videoUrl ?? item.video_path ?? "";

  return {
    id: item.id,
    labelId: item.label_id ?? item.labelId ?? item.phrase_id ?? item.phraseId ?? null,
    userId: item.user_id ?? item.userId ?? null,
    userEmail: item.user_email ?? item.userEmail ?? "",
    phraseId: item.phrase_id ?? item.phraseId ?? null,
    phraseText: item.phrase_text ?? item.phraseText ?? "",
    entryType:
      item.entry_type ?? item.entryType ?? item.label_type ?? item.labelType ?? item.recognition_level ?? item.recognitionLevel ?? "phrase",
    labelType:
      item.label_type ?? item.labelType ?? item.entry_type ?? item.entryType ?? item.recognition_level ?? item.recognitionLevel ?? "phrase",
    recognitionLevel: item.recognition_level ?? item.recognitionLevel ?? "phrase",
    unitCode: item.unit_code ?? item.unitCode ?? "",
    categoryName: item.category_name ?? item.categoryName ?? "",
    signLanguage: item.sign_language ?? item.signLanguage ?? "rsl",
    signerLabel: item.signer_label ?? item.signerLabel ?? "",
    durationMs: Number(item.duration_ms ?? item.durationMs ?? 0),
    datasetSplit: item.dataset_split ?? item.datasetSplit ?? "unassigned",
    status: item.status ?? "draft",
    qualityScore:
      item.quality_score ?? item.qualityScore ?? null,
    reviewStatus: item.review_status ?? item.reviewStatus ?? "pending",
    reviewNotes: item.review_notes ?? item.reviewNotes ?? "",
    reviewedBy: item.reviewed_by ?? item.reviewedBy ?? null,
    reviewerEmail: item.reviewer_email ?? item.reviewerEmail ?? "",
    reviewedAt: item.reviewed_at ?? item.reviewedAt ?? null,
    createdAt,
    videoUrl: videoUrlPath ? buildApiUrl(videoUrlPath) : "",
    landmarkSequenceId:
      item.landmark_sequence_id ?? item.landmarkSequenceId ?? null,
    landmarkFilePath:
      item.landmark_file_path ?? item.landmarkFilePath ?? "",
    landmarkUrl: item.landmark_url
      ? buildApiUrl(item.landmark_url)
      : item.landmarkUrl
        ? buildApiUrl(item.landmarkUrl)
        : "",
    landmarkFrameCount: Number(
      item.landmark_frame_count ?? item.landmarkFrameCount ?? 0,
    ),
    landmarkStatus: item.landmark_status ?? item.landmarkStatus ?? "pending",
    landmarkValidFrameRatio:
      item.landmark_valid_frame_ratio === null ||
      item.landmark_valid_frame_ratio === undefined
        ? item.landmarkValidFrameRatio ?? null
        : Number(item.landmark_valid_frame_ratio),
    landmarkMissingHandRatio:
      item.landmark_missing_hand_ratio === null ||
      item.landmark_missing_hand_ratio === undefined
        ? item.landmarkMissingHandRatio ?? null
        : Number(item.landmark_missing_hand_ratio),
    landmarkMissingFaceRatio:
      item.landmark_missing_face_ratio === null ||
      item.landmark_missing_face_ratio === undefined
        ? item.landmarkMissingFaceRatio ?? null
        : Number(item.landmark_missing_face_ratio),
    landmarkMissingPoseRatio:
      item.landmark_missing_pose_ratio === null ||
      item.landmark_missing_pose_ratio === undefined
        ? item.landmarkMissingPoseRatio ?? null
        : Number(item.landmark_missing_pose_ratio),
    landmarkNormalizationVersion:
      item.landmark_normalization_version ??
      item.landmarkNormalizationVersion ??
      "none",
    landmarkErrorMessage:
      item.landmark_error_message ?? item.landmarkErrorMessage ?? "",
    landmarkUpdatedAt:
      item.landmark_updated_at ?? item.landmarkUpdatedAt ?? null,
    time: new Date(createdAt).toLocaleTimeString("ru-RU", {
      hour: "2-digit",
      minute: "2-digit",
    }),
  };
}

export async function fetchTrainingVideos(filters = {}) {
  const params = new URLSearchParams();

  if (filters.userEmail) {
    params.set("userEmail", filters.userEmail);
  }

  if (filters.phraseId) {
    params.set("phraseId", String(filters.phraseId));
  }

  if (filters.signLanguage) {
    params.set("signLanguage", filters.signLanguage);
  }

  if (filters.recognitionLevel) {
    params.set("recognitionLevel", filters.recognitionLevel);
  }

  if (filters.status) {
    params.set("status", filters.status);
  }

  if (filters.datasetSplit) {
    params.set("datasetSplit", filters.datasetSplit);
  }

  if (filters.reviewStatus) {
    params.set("reviewStatus", filters.reviewStatus);
  }

  const query = params.size ? `?${params.toString()}` : "";
  const data = await request(`/api/training-videos${query}`);

  return data.map(formatTrainingVideo);
}

export async function createTrainingVideo(formData) {
  const item = await request("/api/training-videos", {
    method: "POST",
    body: formData,
  });

  return formatTrainingVideo(item);
}

export async function deleteTrainingVideo(id) {
  return request(`/api/training-videos/${id}`, {
    method: "DELETE",
  });
}

export async function extractTrainingVideoLandmarks(id) {
  const item = await request(`/api/training-videos/${id}/extract-landmarks`, {
    method: "POST",
  });

  return formatTrainingVideo(item);
}

export async function updateTrainingVideoReview(id, payload) {
  const item = await request(`/api/training-videos/${id}/review`, {
    method: "PUT",
    body: payload,
  });

  return formatTrainingVideo(item);
}
