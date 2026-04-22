import { request } from "./apiClient";
import { DEFAULT_SETTINGS } from "./settingsApi";

function formatCategory(item) {
  return {
    id: item.id,
    slug: item.slug,
    name: item.name,
    sortOrder: item.sort_order ?? item.sortOrder ?? 0,
  };
}

function formatPhrase(item) {
  return {
    id: item.id,
    labelId: item.id,
    label: item.text,
    textValue: item.text,
    signLanguage: item.sign_language ?? "rsl",
    entryType: item.entry_type ?? item.recognition_level ?? "phrase",
    recognitionLevel: item.recognition_level ?? "phrase",
    unitCode: item.unit_code ?? "",
    description: item.description ?? "",
    referenceNotes: item.reference_notes ?? "",
    isV1: Boolean(item.is_v1),
    isLocked: Boolean(item.is_locked),
    category: item.category_name,
    categorySlug: item.category_slug,
    isFeatured: Boolean(item.is_featured),
    isActive: item.is_active ?? true,
    sortOrder: item.sort_order ?? item.sortOrder ?? 0,
  };
}

function formatSignLanguage(item) {
  return {
    id: item.id,
    code: item.code,
    name: item.name,
    isActive: Boolean(item.is_active),
    isDefault: Boolean(item.is_default),
  };
}

export async function fetchBootstrapData() {
  const data = await request("/api/meta/bootstrap");

  return {
    defaultUser: data.defaultUser ?? null,
    settings: {
      ...DEFAULT_SETTINGS,
      ...(data.settings ?? {}),
    },
    categories: (data.categories ?? []).map(formatCategory),
    phrases: (data.phrases ?? []).map(formatPhrase),
    signLanguages: (data.signLanguages ?? []).map(formatSignLanguage),
  };
}
