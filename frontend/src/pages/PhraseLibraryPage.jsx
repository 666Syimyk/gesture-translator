import { useMemo, useState } from "react";
import {
  GlassCard,
  PrimaryButton,
  SecondaryButton,
  StatusBadge,
  TitleBlock,
} from "../components/PremiumUI";
import { cancelSpeech, speakWithSettings } from "../utils/speech";
import {
  DEFAULT_RECOGNITION_LEVEL,
  getRecognitionLevelLabel,
  RECOGNITION_LEVEL_OPTIONS,
} from "../recognitionLevels";

const ALL_CATEGORY = "Все";

export default function PhraseLibraryPage({
  phraseCategories,
  phraseLibrary,
  settings,
  isContentLoading,
  contentError,
  addToHistory,
  setRecognizedText,
  setScreen,
  onStartTraining,
}) {
  const [selectedCategory, setSelectedCategory] = useState(ALL_CATEGORY);
  const [selectedRecognitionLevel, setSelectedRecognitionLevel] = useState(
    DEFAULT_RECOGNITION_LEVEL,
  );
  const [searchValue, setSearchValue] = useState("");

  const categories = useMemo(() => {
    const preferredCategories = settings.preferredCategories ?? [];
    const sortedCategories = [...phraseCategories].sort((left, right) => {
      const leftPreferred = preferredCategories.includes(left.name) ? 0 : 1;
      const rightPreferred = preferredCategories.includes(right.name) ? 0 : 1;

      if (leftPreferred !== rightPreferred) {
        return leftPreferred - rightPreferred;
      }

      return left.sortOrder - right.sortOrder;
    });

    return [ALL_CATEGORY, ...sortedCategories.map((category) => category.name)];
  }, [phraseCategories, settings.preferredCategories]);

  const filteredPhrases = useMemo(() => {
    const query = searchValue.trim().toLowerCase();
    const preferredCategories = settings.preferredCategories ?? [];

    const scopedByCategory =
      selectedCategory === ALL_CATEGORY
        ? phraseLibrary
        : phraseLibrary.filter((item) => item.category === selectedCategory);

    const scopedByLevel = scopedByCategory.filter(
      (item) => item.recognitionLevel === selectedRecognitionLevel,
    );

    const scopedBySearch = query
      ? scopedByLevel.filter((item) => item.label.toLowerCase().includes(query))
      : scopedByLevel;

    return [...scopedBySearch].sort((left, right) => {
      const leftPreferred = preferredCategories.includes(left.category) ? 0 : 1;
      const rightPreferred = preferredCategories.includes(right.category) ? 0 : 1;

      if (leftPreferred !== rightPreferred) {
        return leftPreferred - rightPreferred;
      }

      if (left.category !== right.category) {
        return left.category.localeCompare(right.category, "ru");
      }

      return left.sortOrder - right.sortOrder;
    });
  }, [
    phraseLibrary,
    searchValue,
    selectedCategory,
    selectedRecognitionLevel,
    settings.preferredCategories,
  ]);

  const isAlphabetMode = selectedRecognitionLevel === "alphabet";

  function handleSpeak(phrase) {
    speakWithSettings(phrase.label, settings);
  }

  function handleUseInCamera(phrase) {
    setRecognizedText(phrase.label);
    setScreen("camera");
  }

  function handleAddToHistory(phrase) {
    addToHistory("phrase", phrase.label);
  }

  function handleStartTraining(phrase) {
    onStartTraining?.(phrase);
  }

  return (
    <div className="space-y-4">
      <TitleBlock
        eyebrow="База фраз"
        title="Словарь"
        subtitle="Каталог букв, цифр, знаков и готовых фраз для камеры, озвучки и сбора датасета."
        action={
          <StatusBadge className="border-cyan-400/20 bg-cyan-400/10 text-cyan-300">
            {settings.signLanguage === "kgsl" ? "KGSL" : "РЖЯ"}
          </StatusBadge>
        }
      />

      <GlassCard>
        <div className="text-sm font-bold text-white">Поиск и фильтр</div>
        <div className="mt-1 text-sm text-slate-300">
          Быстро находите нужную букву, цифру, знак или фразу и сразу используйте их в
          рабочем режиме.
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          {RECOGNITION_LEVEL_OPTIONS.map((level) => (
            <button
              key={level.id}
              type="button"
              onClick={() => setSelectedRecognitionLevel(level.id)}
              className={`rounded-full px-4 py-2 text-sm font-bold transition ${
                selectedRecognitionLevel === level.id
                  ? "bg-cyan-400 text-slate-950"
                  : "border border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
              }`}
            >
              {level.label}
            </button>
          ))}
        </div>

        <input
          value={searchValue}
          onChange={(event) => setSearchValue(event.target.value)}
          placeholder={
            isAlphabetMode ? "Поиск буквы или цифры" : "Поиск знака или фразы"
          }
          className="mt-4 w-full rounded-[20px] border border-white/10 bg-black/20 px-4 py-3 text-sm text-white outline-none placeholder:text-slate-500"
        />

        {contentError ? (
          <div className="mt-4 rounded-[20px] border border-red-400/20 bg-red-500/10 p-4 text-sm text-red-200">
            {contentError}
          </div>
        ) : isContentLoading ? (
          <div className="mt-4 rounded-[20px] bg-white/5 p-4 text-sm text-slate-300">
            Загружаю библиотеку из базы...
          </div>
        ) : (
          <>
            <div className="mt-4 flex flex-wrap gap-2">
              {categories.map((category) => (
                <button
                  key={category}
                  type="button"
                  onClick={() => setSelectedCategory(category)}
                  className={`rounded-full px-4 py-2 text-sm font-bold transition ${
                    selectedCategory === category
                      ? "bg-cyan-400 text-slate-950"
                      : "border border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
                  }`}
                >
                  {category}
                </button>
              ))}
            </div>

            <div className="mt-4 space-y-3">
              {filteredPhrases.length === 0 ? (
                <div className="rounded-[20px] bg-white/5 p-4 text-sm text-slate-300">
                  По текущему фильтру ничего не найдено.
                </div>
              ) : isAlphabetMode ? (
                <GlassCard className="bg-white/4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-300/70">
                        Буквы и цифры
                      </div>
                      <div className="mt-2 text-lg font-bold text-white">
                        Таблица символов для режима «Буквы / цифры»
                      </div>
                      <div className="mt-2 text-sm leading-6 text-slate-300">
                        Нажмите на символ, чтобы отправить его в камеру. Для
                        обучения используйте запись примеров по каждой букве и цифре.
                      </div>
                    </div>
                    <StatusBadge className="border-cyan-400/20 bg-cyan-400/10 text-cyan-300">
                      {filteredPhrases.length} символов
                    </StatusBadge>
                  </div>

                  <div className="mt-4 grid grid-cols-4 gap-3 sm:grid-cols-6">
                    {filteredPhrases.map((phrase) => (
                      <button
                        key={phrase.id}
                        type="button"
                        onClick={() => handleUseInCamera(phrase)}
                        className="rounded-[22px] border border-white/10 bg-black/20 p-4 text-center transition hover:bg-black/30"
                      >
                        <div className="text-3xl font-black leading-none text-white">
                          {phrase.label}
                        </div>
                        <div className="mt-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-300/70">
                          {(phrase.unitCode || phrase.label).replace("LETTER_", "")}
                        </div>
                      </button>
                    ))}
                  </div>
                </GlassCard>
              ) : (
                filteredPhrases.map((phrase) => (
                  <GlassCard key={phrase.id} className="bg-white/4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-300/70">
                          {phrase.category}
                        </div>
                        <div
                          className={`mt-2 font-bold leading-snug text-white ${
                            settings.largeTextEnabled ? "text-2xl" : "text-lg"
                          }`}
                        >
                          {phrase.label}
                        </div>
                        {phrase.description ? (
                          <div className="mt-2 text-sm leading-6 text-slate-300">
                            {phrase.description}
                          </div>
                        ) : null}
                        {phrase.referenceNotes ? (
                          <div className="mt-2 text-xs leading-5 text-slate-400">
                            {phrase.referenceNotes}
                          </div>
                        ) : null}
                      </div>

                      <div className="flex flex-col items-end gap-2">
                        <StatusBadge
                          className={
                            phrase.isActive
                              ? "border-emerald-400/20 bg-emerald-400/10 text-emerald-300"
                              : "border-white/10 bg-white/5 text-slate-300"
                          }
                        >
                          {phrase.isActive ? "Активна" : "Скрыта"}
                        </StatusBadge>
                        {phrase.isV1 ? (
                          <StatusBadge className="border-cyan-400/20 bg-cyan-400/10 text-cyan-300">
                            V1
                          </StatusBadge>
                        ) : null}
                        <StatusBadge className="border-white/10 bg-white/5 text-slate-200">
                          {getRecognitionLevelLabel(phrase.recognitionLevel)}
                        </StatusBadge>
                        {phrase.isLocked ? (
                          <StatusBadge className="border-white/10 bg-white/5 text-slate-200">
                            Locked
                          </StatusBadge>
                        ) : null}
                      </div>
                    </div>

                    <div className="mt-4 grid grid-cols-2 gap-2">
                      <SecondaryButton
                        onClick={() => handleSpeak(phrase)}
                        className="px-3 py-3 text-sm"
                      >
                        Озвучить
                      </SecondaryButton>
                      <SecondaryButton
                        onClick={() => handleAddToHistory(phrase)}
                        className="px-3 py-3 text-sm"
                      >
                        В историю
                      </SecondaryButton>
                      <PrimaryButton
                        onClick={() => handleUseInCamera(phrase)}
                        className="px-3 py-3 text-sm"
                      >
                        В камеру
                      </PrimaryButton>
                      <SecondaryButton
                        onClick={() => handleStartTraining(phrase)}
                        className="px-3 py-3 text-sm"
                      >
                        Записать
                      </SecondaryButton>
                    </div>
                  </GlassCard>
                ))
              )}
            </div>
          </>
        )}
      </GlassCard>

      <SecondaryButton onClick={cancelSpeech} className="w-full">
        Остановить озвучку
      </SecondaryButton>
    </div>
  );
}
