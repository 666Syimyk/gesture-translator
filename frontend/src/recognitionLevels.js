export const RECOGNITION_LEVEL_OPTIONS = [
  { id: "alphabet", label: "Буквы / цифры", shortLabel: "Alphabet" },
  { id: "sign", label: "Знаки", shortLabel: "Signs" },
  { id: "phrase", label: "Фразы", shortLabel: "Phrases" },
];

export const DEFAULT_RECOGNITION_LEVEL = "alphabet";

export const MODEL_SCOPE_OPTIONS = [
  { id: "unified", label: "Общая модель", shortLabel: "Unified" },
  ...RECOGNITION_LEVEL_OPTIONS,
];

export const DEFAULT_MODEL_SCOPE = "unified";

export function normalizeRecognitionLevel(value) {
  return (
    RECOGNITION_LEVEL_OPTIONS.find((option) => option.id === value)?.id ??
    DEFAULT_RECOGNITION_LEVEL
  );
}

export function normalizeModelScope(value) {
  return (
    MODEL_SCOPE_OPTIONS.find((option) => option.id === value)?.id ??
    DEFAULT_MODEL_SCOPE
  );
}

export function getRecognitionLevelLabel(value) {
  return (
    RECOGNITION_LEVEL_OPTIONS.find((option) => option.id === value)?.label ??
    "Буквы / цифры"
  );
}

export function getModelScopeLabel(value) {
  return (
    MODEL_SCOPE_OPTIONS.find((option) => option.id === value)?.label ??
    "Общая модель"
  );
}
