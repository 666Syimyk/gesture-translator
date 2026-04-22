export const DEFAULT_USER = {
  email: "demo@gesture-translator.local",
  passwordHash: "demo-user-no-auth",
  displayName: "Demo User",
};

export const DEFAULT_USER_SETTINGS = {
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

export const DEFAULT_SIGN_LANGUAGES = [
  {
    code: "rsl",
    name: "Русский жестовый язык",
    isActive: true,
    isDefault: true,
  },
  {
    code: "kgsl",
    name: "Кыргызский жестовый язык",
    isActive: true,
    isDefault: false,
  },
];

export const DEFAULT_PHRASE_CATEGORIES = [
  { slug: "alphabet", name: "Алфавит", sortOrder: 0 },
  { slug: "basic", name: "Базовые", sortOrder: 1 },
  { slug: "shop", name: "Магазин", sortOrder: 2 },
  { slug: "hospital", name: "Больница", sortOrder: 3 },
  { slug: "street", name: "Улица", sortOrder: 4 },
  { slug: "transport", name: "Транспорт", sortOrder: 5 },
  { slug: "home", name: "Дом", sortOrder: 6 },
  { slug: "urgent", name: "Экстренно", sortOrder: 7 },
  { slug: "user-words", name: "\u0412\u0430\u0448\u0438 \u0441\u043b\u043e\u0432\u0430", sortOrder: 8 },
];

function buildV1Phrase(categorySlug, text, sortOrder) {
  return {
    categorySlug,
    text,
    signLanguage: "rsl",
    entryType: "phrase",
    recognitionLevel: "phrase",
    unitCode: "",
    description: `Эталонная фраза словаря версии 1.0: ${text}`,
    referenceNotes:
      "Показывать фразу спокойно, с полностью видимыми руками, лицом и корпусом. Для датасета нужен чистый кадр, устойчивый темп и подтверждение экспертом.",
    isV1: true,
    isLocked: true,
    isFeatured: true,
    sortOrder,
  };
}

function buildPhrase(categorySlug, text, isFeatured, sortOrder) {
  return {
    categorySlug,
    text,
    signLanguage: "rsl",
    entryType: "phrase",
    recognitionLevel: "phrase",
    unitCode: "",
    description: "",
    referenceNotes: "",
    isV1: false,
    isLocked: false,
    isFeatured,
    sortOrder,
  };
}

const DEFAULT_ALPHABET_REFERENCE_NOTES =
  "Показывать букву чётко перед камерой, не перекрывать пальцы и удерживать форму кисти до распознавания.";
const MOTION_ALPHABET_REFERENCE_NOTES =
  "Буква включает короткое движение. Сначала покажите базовую форму кисти, затем выполните движение целиком в кадре.";
const ALPHABET_REFERENCE = {
  LETTER_A: {
    description: "Кулак, большой палец сбоку.",
  },
  LETTER_B: {
    description: "Ладонь раскрыта, пальцы вместе, большой прижат к ладони.",
  },
  LETTER_V: {
    description:
      "Три пальца вверх: указательный, средний и безымянный. Большой и мизинец прижаты.",
  },
  LETTER_G: {
    description: "Указательный палец направлен вперёд, остальные в кулаке.",
  },
  LETTER_D: {
    description: "Ладонь раскрыта, большой палец касается безымянного.",
  },
  LETTER_E: {
    description: "Пальцы слегка согнуты, как «щепотка».",
  },
  LETTER_YO: {
    description: "Как «Е», но с лёгким движением вверх.",
    referenceNotes: MOTION_ALPHABET_REFERENCE_NOTES,
  },
  LETTER_ZH: {
    description: "Растопыренные пальцы, ладонь вперёд.",
  },
  LETTER_Z: {
    description: "Указательный палец рисует «Z» в воздухе.",
    referenceNotes: MOTION_ALPHABET_REFERENCE_NOTES,
  },
  LETTER_I: {
    description: "Мизинец вверх, остальные пальцы сжаты.",
  },
  LETTER_I_SHORT: {
    description: "Как «И», но с коротким движением вниз.",
    referenceNotes: MOTION_ALPHABET_REFERENCE_NOTES,
  },
  LETTER_K: {
    description: "Указательный и средний пальцы вверх, большой между ними.",
  },
  LETTER_L: {
    description: "Большой и указательный пальцы образуют угол.",
  },
  LETTER_M: {
    description: "Большой палец под тремя пальцами.",
  },
  LETTER_N: {
    description: "Большой палец под двумя пальцами.",
  },
  LETTER_O: {
    description: "Пальцы образуют круг.",
  },
  LETTER_P: {
    description: "Ладонь направлена вниз, пальцы вместе.",
  },
  LETTER_R: {
    description: "Указательный и средний пальцы скрещены.",
  },
  LETTER_S: {
    description: "Рука полукругом, как буква «С».",
  },
  LETTER_T: {
    description: "Большой палец между указательным и средним.",
  },
  LETTER_U: {
    description: "Указательный и средний пальцы подняты вместе.",
  },
  LETTER_F: {
    description: "Круг из большого и указательного пальцев, остальные пальцы вверх.",
  },
  LETTER_KH: {
    description: "Указательный палец согнут крючком.",
  },
  LETTER_TS: {
    description: "Указательный и средний пальцы направлены вниз.",
  },
  LETTER_CH: {
    description: "Большой, указательный и средний пальцы вместе.",
  },
  LETTER_SH: {
    description: "Три пальца вверх и широко разведены.",
  },
  LETTER_SHCH: {
    description: "Как «Ш», но с движением вниз.",
    referenceNotes: MOTION_ALPHABET_REFERENCE_NOTES,
  },
  LETTER_HARD_SIGN: {
    description: "Кулак, большой палец вверх, короткое движение.",
    referenceNotes: MOTION_ALPHABET_REFERENCE_NOTES,
  },
  LETTER_YERU: {
    description: "Мизинец и большой палец разведены в стороны.",
  },
  LETTER_SOFT_SIGN: {
    description: "Ладонь ребром, пальцы вместе.",
  },
  LETTER_EH: {
    description: "Как «С», но с движением внутрь.",
    referenceNotes: MOTION_ALPHABET_REFERENCE_NOTES,
  },
  LETTER_YU: {
    description: "Круг, как у «О», и мизинец в сторону.",
  },
  LETTER_YA: {
    description: "Мизинец и большой палец разведены, как «Y».",
  },
};

function buildAlphabetLetter(text, unitCode, sortOrder) {
  const reference = ALPHABET_REFERENCE[unitCode];

  return {
    categorySlug: "alphabet",
    text,
    signLanguage: "rsl",
    entryType: "alphabet",
    recognitionLevel: "alphabet",
    unitCode,
    description: reference?.description
      ? `Эталон буквы ${text}: ${reference.description}`
      : `Буква русского дактильного алфавита: ${text}`,
    referenceNotes:
      reference?.referenceNotes ?? DEFAULT_ALPHABET_REFERENCE_NOTES,
    isV1: true,
    isLocked: true,
    isFeatured: true,
    sortOrder,
  };
}

function buildAlphabetNumber(text, unitCode, sortOrder) {
  return {
    categorySlug: "alphabet",
    text,
    signLanguage: "rsl",
    entryType: "alphabet",
    recognitionLevel: "alphabet",
    unitCode,
    description: `Цифра русского жестового набора: ${text}`,
    referenceNotes:
      "Показывать цифру стабильно 1 секунду. Кисть, лицо и верх корпуса должны быть видны в кадре. Эталон цифры подтверждается экспертом по РЖЯ.",
    isV1: true,
    isLocked: true,
    isFeatured: true,
    sortOrder,
  };
}

function buildV1Sign(categorySlug, text, unitCode, sortOrder) {
  return {
    categorySlug,
    text,
    signLanguage: "rsl",
    entryType: "sign",
    recognitionLevel: "sign",
    unitCode,
    description: `Базовый знак версии 1: ${text}`,
    referenceNotes:
      "Отдельный знак для режима «Знаки». Эталон, темп, мимика и допустимые вариации подтверждаются экспертом по РЖЯ.",
    isV1: true,
    isLocked: true,
    isFeatured: true,
    sortOrder,
  };
}

export const RSL_DACTYL_ALPHABET = [
  buildAlphabetLetter("А", "LETTER_A", 1),
  buildAlphabetLetter("Б", "LETTER_B", 2),
  buildAlphabetLetter("В", "LETTER_V", 3),
  buildAlphabetLetter("Г", "LETTER_G", 4),
  buildAlphabetLetter("Д", "LETTER_D", 5),
  buildAlphabetLetter("Е", "LETTER_E", 6),
  buildAlphabetLetter("Ё", "LETTER_YO", 7),
  buildAlphabetLetter("Ж", "LETTER_ZH", 8),
  buildAlphabetLetter("З", "LETTER_Z", 9),
  buildAlphabetLetter("И", "LETTER_I", 10),
  buildAlphabetLetter("Й", "LETTER_I_SHORT", 11),
  buildAlphabetLetter("К", "LETTER_K", 12),
  buildAlphabetLetter("Л", "LETTER_L", 13),
  buildAlphabetLetter("М", "LETTER_M", 14),
  buildAlphabetLetter("Н", "LETTER_N", 15),
  buildAlphabetLetter("О", "LETTER_O", 16),
  buildAlphabetLetter("П", "LETTER_P", 17),
  buildAlphabetLetter("Р", "LETTER_R", 18),
  buildAlphabetLetter("С", "LETTER_S", 19),
  buildAlphabetLetter("Т", "LETTER_T", 20),
  buildAlphabetLetter("У", "LETTER_U", 21),
  buildAlphabetLetter("Ф", "LETTER_F", 22),
  buildAlphabetLetter("Х", "LETTER_KH", 23),
  buildAlphabetLetter("Ц", "LETTER_TS", 24),
  buildAlphabetLetter("Ч", "LETTER_CH", 25),
  buildAlphabetLetter("Ш", "LETTER_SH", 26),
  buildAlphabetLetter("Щ", "LETTER_SHCH", 27),
  buildAlphabetLetter("Ъ", "LETTER_HARD_SIGN", 28),
  buildAlphabetLetter("Ы", "LETTER_YERU", 29),
  buildAlphabetLetter("Ь", "LETTER_SOFT_SIGN", 30),
  buildAlphabetLetter("Э", "LETTER_EH", 31),
  buildAlphabetLetter("Ю", "LETTER_YU", 32),
  buildAlphabetLetter("Я", "LETTER_YA", 33),
];

export const RSL_DACTYL_NUMBERS = [
  buildAlphabetNumber("0", "NUMBER_0", 101),
  buildAlphabetNumber("1", "NUMBER_1", 102),
  buildAlphabetNumber("2", "NUMBER_2", 103),
  buildAlphabetNumber("3", "NUMBER_3", 104),
  buildAlphabetNumber("4", "NUMBER_4", 105),
  buildAlphabetNumber("5", "NUMBER_5", 106),
  buildAlphabetNumber("6", "NUMBER_6", 107),
  buildAlphabetNumber("7", "NUMBER_7", 108),
  buildAlphabetNumber("8", "NUMBER_8", 109),
  buildAlphabetNumber("9", "NUMBER_9", 110),
];

export const RSL_BASE_SIGNS = [
  buildV1Sign("basic", "Здравствуйте", "SIGN_HELLO", 1),
  buildV1Sign("basic", "Да", "SIGN_YES", 2),
  buildV1Sign("basic", "Нет", "SIGN_NO", 3),
  buildV1Sign("basic", "Спасибо", "SIGN_THANKS", 4),
  buildV1Sign("basic", "Привет", "SIGN_HI", 5),
  buildV1Sign("basic", "Повторите", "SIGN_REPEAT", 6),
  buildV1Sign("basic", "Хорошо", "SIGN_GOOD", 7),
  buildV1Sign("transport", "Подождите", "SIGN_WAIT", 1),
  buildV1Sign("urgent", "Стоп", "SIGN_STOP", 2),
  buildV1Sign("urgent", "Помощь", "SIGN_HELP", 3),
  buildV1Sign("home", "Вода", "SIGN_WATER", 1),
  buildV1Sign("home", "Еда", "SIGN_FOOD", 2),
  buildV1Sign("home", "Пить", "SIGN_DRINK", 3),
  buildV1Sign("home", "Дом", "SIGN_HOME", 4),
  buildV1Sign("home", "Холодно", "SIGN_COLD", 5),
  buildV1Sign("hospital", "Больно", "SIGN_PAIN", 1),
  buildV1Sign("hospital", "Врач", "SIGN_DOCTOR", 2),
  buildV1Sign("hospital", "Плохо", "SIGN_BAD", 3),
  buildV1Sign("hospital", "Медленно", "SIGN_SLOW", 4),
  buildV1Sign("street", "Туалет", "SIGN_TOILET", 1),
  buildV1Sign("street", "Женский туалет", "SIGN_WOMENS_TOILET", 2),
  buildV1Sign("street", "Мужской туалет", "SIGN_MENS_TOILET", 3),
  buildV1Sign("user-words", "\u041f\u0440\u0438\u0432\u0435\u0442", "SIGN_USER_PRIVET", 1),
  buildV1Sign("user-words", "\u041f\u043e\u043a\u0430", "SIGN_USER_POKA", 2),
  buildV1Sign("user-words", "\u0421\u043f\u0430\u0441\u0438\u0431\u043e", "SIGN_USER_SPASIBO", 3),
  buildV1Sign("user-words", "\u041f\u043e\u0436\u0430\u043b\u0443\u0439\u0441\u0442\u0430", "SIGN_USER_POZHALUISTA", 4),
  buildV1Sign("user-words", "\u0414\u0430", "SIGN_USER_DA", 5),
  buildV1Sign("user-words", "\u041d\u0435\u0442", "SIGN_USER_NET", 6),
  buildV1Sign("user-words", "\u041c\u0430\u043c\u0430", "SIGN_USER_MAMA", 7),
  buildV1Sign("user-words", "\u041f\u0430\u043f\u0430", "SIGN_USER_PAPA", 8),
  buildV1Sign("user-words", "\u0414\u0440\u0443\u0433", "SIGN_USER_DRUG", 9),
  buildV1Sign("user-words", "\u041b\u044e\u0431\u043e\u0432\u044c", "SIGN_USER_LYUBOV", 10),
];

export const FIRST_WAVE_DATASET_PHRASES = [
  buildV1Phrase("basic", "Здравствуйте", 1),
  buildV1Phrase("basic", "Да", 2),
  buildV1Phrase("basic", "Нет", 3),
  buildV1Phrase("basic", "Привет", 8),
  buildV1Phrase("basic", "Пока", 9),
  buildV1Phrase("basic", "Дружба", 10),
  buildV1Phrase("basic", "Мужчина", 11),
  buildV1Phrase("basic", "Женщина", 12),
  buildV1Phrase("basic", "Солнце", 13),
  buildV1Phrase("transport", "Подождите", 1),
  buildV1Phrase("basic", "Повторите", 5),
  buildV1Phrase("home", "Дом", 0),
  buildV1Phrase("urgent", "Мне нужна помощь", 1),
  buildV1Phrase("home", "Мне нужна вода", 1),
  buildV1Phrase("home", "Мне нужна еда", 2),
  buildV1Phrase("hospital", "Мне плохо", 1),
  buildV1Phrase("hospital", "Больно", 2),
  buildV1Phrase("hospital", "Позвоните врачу", 3),
  buildV1Phrase("street", "Где туалет?", 1),
  buildV1Phrase("basic", "Я не понимаю", 6),
  buildV1Phrase("basic", "Говорите медленнее", 7),
  buildV1Phrase("home", "Я хочу домой", 3),
  buildV1Phrase("urgent", "Стоп", 2),
  buildV1Phrase("basic", "Спасибо", 4),
  buildV1Phrase("street", "Подойдите сюда", 2),
  buildV1Phrase("home", "Мне холодно", 4),
  buildV1Phrase("home", "Мне жарко", 5),
];

export const DEFAULT_PHRASES = [
  ...RSL_DACTYL_ALPHABET,
  ...RSL_DACTYL_NUMBERS,
  ...RSL_BASE_SIGNS,
];
