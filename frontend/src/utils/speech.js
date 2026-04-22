const DEFAULT_TEST_PHRASE = "Проверка озвучки. Всё работает.";
const NON_SPEAKABLE_VALUES = new Set(["...", "Жест не распознан"]);
const DIGIT_SPEECH_MAP = {
  0: "Цифра ноль",
  1: "Цифра один",
  2: "Цифра два",
  3: "Цифра три",
  4: "Цифра четыре",
  5: "Цифра пять",
  6: "Цифра шесть",
  7: "Цифра семь",
  8: "Цифра восемь",
  9: "Цифра девять",
};
const LETTER_SPEECH_MAP = {
  А: "Буква А",
  Б: "Буква Бэ",
  В: "Буква Вэ",
  Г: "Буква Гэ",
  Д: "Буква Дэ",
  Е: "Буква Е",
  Ё: "Буква Ё",
  Ж: "Буква Жэ",
  З: "Буква Зэ",
  И: "Буква И",
  Й: "Буква Й",
  К: "Буква Ка",
  Л: "Буква Эль",
  М: "Буква Эм",
  Н: "Буква Эн",
  О: "Буква О",
  П: "Буква Пэ",
  Р: "Буква Эр",
  С: "Буква Эс",
  Т: "Буква Тэ",
  У: "Буква У",
  Ф: "Буква Эф",
  Х: "Буква Ха",
  Ц: "Буква Цэ",
  Ч: "Буква Че",
  Ш: "Буква Ша",
  Щ: "Буква Ща",
  Ъ: "Буква твёрдый знак",
  Ы: "Буква Ы",
  Ь: "Буква мягкий знак",
  Э: "Буква Э",
  Ю: "Буква Ю",
  Я: "Буква Я",
};

function getSpeechSynthesisInstance() {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) {
    return null;
  }

  return window.speechSynthesis;
}

function resolveSpeechLanguage(settings = {}) {
  return settings.uiLanguage === "en" ? "en-US" : "ru-RU";
}

function normalizeSpeakableText(textToSpeak) {
  if (typeof textToSpeak !== "string") {
    return "";
  }

  return textToSpeak.trim();
}

export function getSpeakableText(textToSpeak, options = {}) {
  const text = normalizeSpeakableText(textToSpeak);
  const recognitionLevel = String(options.recognitionLevel || "").trim().toLowerCase();

  if (!text || NON_SPEAKABLE_VALUES.has(text)) {
    return "";
  }

  if (recognitionLevel === "alphabet") {
    if (DIGIT_SPEECH_MAP[text]) {
      return DIGIT_SPEECH_MAP[text];
    }

    if (LETTER_SPEECH_MAP[text]) {
      return LETTER_SPEECH_MAP[text];
    }
  }

  return text;
}

function findMatchingVoice(voices, settings = {}) {
  if (!voices.length) {
    return null;
  }

  const preferredVoiceName = settings.voiceName?.trim();
  const language = resolveSpeechLanguage(settings).toLowerCase();

  if (preferredVoiceName) {
    const exactMatch = voices.find((voice) => voice.name === preferredVoiceName);

    if (exactMatch) {
      return exactMatch;
    }
  }

  return (
    voices.find((voice) => voice.lang?.toLowerCase().startsWith(language)) ??
    voices.find((voice) => voice.default) ??
    voices[0]
  );
}

export function isSpeechSynthesisSupported() {
  return Boolean(getSpeechSynthesisInstance());
}

export function isSpeakableText(textToSpeak, options = {}) {
  return Boolean(getSpeakableText(textToSpeak, options));
}

export function getSpeechVoices() {
  const speechSynthesis = getSpeechSynthesisInstance();

  if (!speechSynthesis) {
    return [];
  }

  return speechSynthesis
    .getVoices()
    .map((voice) => ({
      name: voice.name,
      lang: voice.lang,
      default: voice.default,
    }))
    .sort((left, right) => {
      if (left.default !== right.default) {
        return left.default ? -1 : 1;
      }

      return left.name.localeCompare(right.name);
    });
}

export function speakWithSettings(textToSpeak, settings = {}, options = {}) {
  const speechSynthesis = getSpeechSynthesisInstance();
  const speakableText = getSpeakableText(textToSpeak, options);

  if (!speakableText || !speechSynthesis) {
    return false;
  }

  const utterance = new SpeechSynthesisUtterance(speakableText);
  const voices = speechSynthesis.getVoices();
  const selectedVoice = findMatchingVoice(voices, settings);

  utterance.lang = resolveSpeechLanguage(settings);
  utterance.rate = settings.speechRate ?? 1;
  utterance.pitch = settings.speechPitch ?? 1;

  if (selectedVoice) {
    utterance.voice = selectedVoice;
    utterance.lang = selectedVoice.lang || utterance.lang;
  }

  speechSynthesis.cancel();
  speechSynthesis.speak(utterance);

  return true;
}

export function testSpeech(settings = {}, text = DEFAULT_TEST_PHRASE) {
  return speakWithSettings(text, settings);
}

export function cancelSpeech() {
  const speechSynthesis = getSpeechSynthesisInstance();

  if (speechSynthesis) {
    speechSynthesis.cancel();
  }
}
