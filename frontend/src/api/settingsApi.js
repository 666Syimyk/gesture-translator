import { request } from "./apiClient";

export const DEFAULT_SETTINGS = {
  autoSpeakEnabled: true,
  speechRate: 1,
  speechPitch: 1,
  voiceName: "",
  uiLanguage: "ru",
  signLanguage: "rsl",
  preferredCategories: [],
  largeTextEnabled: false,
  developerModeEnabled: false,
};

function formatSettingsResponse(data) {
  return {
    user: data.user ?? null,
    settings: {
      ...DEFAULT_SETTINGS,
      ...(data.settings ?? {}),
    },
  };
}

export async function fetchSettings(userEmail) {
  const query = userEmail ? `?userEmail=${encodeURIComponent(userEmail)}` : "";
  const data = await request(`/api/settings${query}`);
  return formatSettingsResponse(data);
}

export async function updateSettings(payload) {
  const data = await request("/api/settings", {
    method: "PUT",
    body: payload,
  });

  return formatSettingsResponse(data);
}
