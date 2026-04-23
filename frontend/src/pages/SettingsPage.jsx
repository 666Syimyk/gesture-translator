import { useEffect, useState } from "react";
import { DEFAULT_SETTINGS } from "../api/settingsApi";
import { getApiBaseUrl, setApiBaseUrl } from "../api/apiClient";
import {
  cancelSpeech,
  getSpeechVoices,
  isSpeechSynthesisSupported,
  testSpeech,
} from "../utils/speech";
import {
  GlassCard,
  PrimaryButton,
  StatusBadge,
  TitleBlock,
} from "../components/PremiumUI";

const UI_LANGUAGE_OPTIONS = [
  { value: "ru", label: "Русский" },
  { value: "en", label: "English" },
];

const SIGN_LANGUAGE_OPTIONS = [
  { value: "rsl", label: "РЖЯ" },
  { value: "kgsl", label: "KGSL" },
];

function ToggleRow({ label, value, onToggle, activeText, inactiveText, tone = "cyan" }) {
  const activeClass =
    tone === "emerald"
      ? "bg-emerald-400/10 text-emerald-300"
      : tone === "amber"
        ? "bg-amber-400/10 text-amber-300"
        : "bg-cyan-400/10 text-cyan-300";

  return (
    <button
      type="button"
      onClick={onToggle}
      className="flex w-full items-center justify-between rounded-[18px] bg-black/20 px-4 py-4"
    >
      <span className="font-semibold text-slate-200">{label}</span>
      <span
        className={`rounded-full px-3 py-1 text-xs font-bold ${
          value ? activeClass : "bg-white/10 text-slate-400"
        }`}
      >
        {value ? activeText : inactiveText}
      </span>
    </button>
  );
}

function PillGroup({ options, value, onChange }) {
  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
          className={`rounded-full px-4 py-2 text-sm font-bold ${
            value === option.value
              ? "bg-cyan-400 text-slate-950"
              : "border border-white/10 bg-white/5 text-slate-300"
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export default function SettingsPage({
  settings,
  categories,
  onSave,
  isSaving,
  saveError,
}) {
  const [draftSettings, setDraftSettings] = useState({
    ...DEFAULT_SETTINGS,
    ...settings,
  });
  const [saveMessage, setSaveMessage] = useState("");
  const [speechSupported] = useState(() => isSpeechSynthesisSupported());
  const [voiceOptions, setVoiceOptions] = useState(() =>
    isSpeechSynthesisSupported() ? getSpeechVoices() : [],
  );
  const [speechStatus, setSpeechStatus] = useState("");
  const [apiBaseUrlDraft, setApiBaseUrlDraft] = useState(() => getApiBaseUrl());
  const [apiBaseUrlStatus, setApiBaseUrlStatus] = useState("");

  useEffect(() => {
    if (!speechSupported) {
      return undefined;
    }

    function syncVoices() {
      setVoiceOptions(getSpeechVoices());
    }

    syncVoices();
    window.speechSynthesis.addEventListener("voiceschanged", syncVoices);

    return () => {
      window.speechSynthesis.removeEventListener("voiceschanged", syncVoices);
    };
  }, [speechSupported]);

  function updateSetting(key, value) {
    setDraftSettings((prev) => ({
      ...prev,
      [key]: value,
    }));
  }

  function togglePreferredCategory(categoryName) {
    setDraftSettings((prev) => {
      const nextCategories = prev.preferredCategories.includes(categoryName)
        ? prev.preferredCategories.filter((item) => item !== categoryName)
        : [...prev.preferredCategories, categoryName];

      return {
        ...prev,
        preferredCategories: nextCategories,
      };
    });
  }

  async function handleSave() {
    try {
      await onSave(draftSettings);
      setSaveMessage("Настройки сохранены");
    } catch {
      setSaveMessage("");
    }
  }

  function handleTestSpeech() {
    if (!speechSupported) {
      setSpeechStatus("Озвучка недоступна в этом браузере");
      return;
    }

    const hasStarted = testSpeech(draftSettings);
    setSpeechStatus(
      hasStarted
        ? "Тестовая фраза отправлена в озвучку"
        : "Не удалось запустить озвучку",
    );
  }

  function handleApplyApiBaseUrl() {
    setApiBaseUrl(apiBaseUrlDraft);
    setApiBaseUrlStatus("Saved. Reloading…");
    window.location.reload();
  }

  function handleClearApiBaseUrl() {
    setApiBaseUrl("");
    setApiBaseUrlDraft("");
    setApiBaseUrlStatus("Cleared. Reloading…");
    window.location.reload();
  }

  return (
    <div className="space-y-4">
      <TitleBlock
        eyebrow="Персонализация"
        title="Настройки"
        subtitle="Озвучка, интерфейс, жестовый язык и служебный режим."
        action={
          <StatusBadge className="border-cyan-400/20 bg-cyan-400/10 text-cyan-300">
            Личный режим
          </StatusBadge>
        }
      />

      <GlassCard>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-sm font-bold text-white">Озвучка</div>
            <div className="mt-1 text-sm text-slate-300">
              Управление автоозвучкой, голосом и тестом звука.
            </div>
          </div>
          <StatusBadge
            className={
              speechSupported
                ? "border-emerald-400/20 bg-emerald-500/10 text-emerald-300"
                : "border-red-400/20 bg-red-500/10 text-red-200"
            }
          >
            {speechSupported ? "Озвучка доступна" : "Озвучка недоступна"}
          </StatusBadge>
        </div>

        <div className="mt-4 space-y-3">
          <ToggleRow
            label="Автоозвучка"
            value={draftSettings.autoSpeakEnabled}
            onToggle={() =>
              updateSetting("autoSpeakEnabled", !draftSettings.autoSpeakEnabled)
            }
            activeText="Включена"
            inactiveText="Выключена"
            tone="emerald"
          />

          <div className="rounded-[18px] bg-black/20 px-4 py-4">
            <div className="flex items-center justify-between text-sm font-semibold text-slate-200">
              <span>Скорость речи</span>
              <span>{draftSettings.speechRate.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="0.7"
              max="1.5"
              step="0.1"
              value={draftSettings.speechRate}
              onChange={(event) =>
                updateSetting("speechRate", Number(event.target.value))
              }
              className="mt-3 w-full accent-cyan-400"
            />
          </div>

          <div className="rounded-[18px] bg-black/20 px-4 py-4">
            <div className="flex items-center justify-between text-sm font-semibold text-slate-200">
              <span>Высота голоса</span>
              <span>{draftSettings.speechPitch.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="0.7"
              max="1.5"
              step="0.1"
              value={draftSettings.speechPitch}
              onChange={(event) =>
                updateSetting("speechPitch", Number(event.target.value))
              }
              className="mt-3 w-full accent-cyan-400"
            />
          </div>

          <div className="rounded-[18px] bg-black/20 px-4 py-4">
            <div className="text-sm font-semibold text-slate-200">Системный голос</div>
            <select
              value={draftSettings.voiceName}
              onChange={(event) => updateSetting("voiceName", event.target.value)}
              disabled={!speechSupported}
              className="mt-3 w-full rounded-[16px] border border-white/10 bg-slate-950 px-4 py-3 text-sm text-white outline-none disabled:cursor-not-allowed disabled:opacity-60"
            >
              <option value="">Автовыбор по языку</option>
              {voiceOptions.map((voice) => (
                <option key={`${voice.name}-${voice.lang}`} value={voice.name}>
                  {voice.name}
                  {voice.lang ? ` (${voice.lang})` : ""}
                  {voice.default ? " - default" : ""}
                </option>
              ))}
            </select>
            <div className="mt-2 text-xs text-slate-500">
              Если оставить автовыбор, приложение само возьмёт подходящий голос.
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <PrimaryButton
              onClick={handleTestSpeech}
              disabled={!speechSupported}
              className="w-full"
            >
              Проверить звук
            </PrimaryButton>
            <PrimaryButton
              onClick={() => {
                cancelSpeech();
                setSpeechStatus("Озвучка остановлена");
              }}
              disabled={!speechSupported}
              className="w-full bg-white/10 text-white shadow-none hover:scale-100"
            >
              Стоп озвучки
            </PrimaryButton>
          </div>

          {speechStatus ? (
            <div className="rounded-[18px] border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-300">
              {speechStatus}
            </div>
          ) : null}
        </div>
      </GlassCard>

      <GlassCard>
        <div className="text-sm font-bold text-white">Backend</div>
        <div className="mt-1 text-sm text-slate-300">
          GitHub Pages hosts only the frontend. If you see “Request failed”, set the backend URL (must be HTTPS on mobile).
        </div>

        <div className="mt-4 grid gap-3">
          <label className="grid gap-2">
            <span className="text-sm font-semibold text-slate-200">API base URL</span>
            <input
              value={apiBaseUrlDraft}
              onChange={(event) => setApiBaseUrlDraft(event.target.value)}
              placeholder="https://your-backend.example.com"
              className="w-full rounded-[14px] border border-white/10 bg-black/20 px-4 py-3 text-sm text-slate-100 outline-none placeholder:text-slate-500 focus:border-cyan-400/40"
              inputMode="url"
              autoCapitalize="none"
              autoCorrect="off"
              spellCheck={false}
            />
          </label>

          <div className="grid grid-cols-2 gap-3">
            <PrimaryButton onClick={handleApplyApiBaseUrl} className="w-full">
              Save & reload
            </PrimaryButton>
            <button
              type="button"
              onClick={handleClearApiBaseUrl}
              className="w-full rounded-[18px] border border-white/10 bg-white/5 px-4 py-3 text-sm font-bold text-slate-200 hover:bg-white/10"
            >
              Clear
            </button>
          </div>

          {apiBaseUrlStatus ? (
            <div className="rounded-[18px] bg-black/20 px-4 py-3 text-sm text-slate-300">
              {apiBaseUrlStatus}
            </div>
          ) : null}
        </div>
      </GlassCard>

      <GlassCard>
        <div className="text-sm font-bold text-white">Интерфейс</div>
        <div className="mt-4 grid gap-3">
          <ToggleRow
            label="Крупный текст"
            value={draftSettings.largeTextEnabled}
            onToggle={() =>
              updateSetting("largeTextEnabled", !draftSettings.largeTextEnabled)
            }
            activeText="Да"
            inactiveText="Нет"
          />

          <ToggleRow
            label="Developer mode"
            value={draftSettings.developerModeEnabled}
            onToggle={() =>
              updateSetting(
                "developerModeEnabled",
                !draftSettings.developerModeEnabled,
              )
            }
            activeText="Включен"
            inactiveText="Скрыт"
            tone="amber"
          />

          <div className="rounded-[18px] bg-black/20 px-4 py-4">
            <div className="text-sm font-semibold text-slate-200">Язык интерфейса</div>
            <PillGroup
              options={UI_LANGUAGE_OPTIONS}
              value={draftSettings.uiLanguage}
              onChange={(value) => updateSetting("uiLanguage", value)}
            />
          </div>

          <div className="rounded-[18px] bg-black/20 px-4 py-4">
            <div className="text-sm font-semibold text-slate-200">Жестовый язык</div>
            <PillGroup
              options={SIGN_LANGUAGE_OPTIONS}
              value={draftSettings.signLanguage}
              onChange={(value) => updateSetting("signLanguage", value)}
            />
          </div>
        </div>
      </GlassCard>

      <GlassCard>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-sm font-bold text-white">Любимые категории</div>
            <div className="mt-1 text-sm text-slate-300">
              Эти категории будут подниматься выше в словаре и быстрых карточках.
            </div>
          </div>
          <StatusBadge className="text-cyan-300">
            {draftSettings.preferredCategories.length} выбрано
          </StatusBadge>
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          {categories.map((category) => {
            const isActive = draftSettings.preferredCategories.includes(category.name);

            return (
              <button
                key={category.id}
                type="button"
                onClick={() => togglePreferredCategory(category.name)}
                className={`rounded-full px-4 py-2 text-sm font-bold transition ${
                  isActive
                    ? "bg-cyan-400 text-slate-950"
                    : "border border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
                }`}
              >
                {category.name}
              </button>
            );
          })}
        </div>
      </GlassCard>

      {saveError ? (
        <div className="rounded-[20px] border border-red-400/20 bg-red-500/10 p-4 text-sm text-red-200">
          {saveError}
        </div>
      ) : null}

      {saveMessage ? (
        <div className="rounded-[20px] border border-emerald-400/20 bg-emerald-500/10 p-4 text-sm text-emerald-200">
          {saveMessage}
        </div>
      ) : null}

      <PrimaryButton onClick={handleSave} disabled={isSaving} className="w-full">
        {isSaving ? "Сохраняю..." : "Сохранить настройки"}
      </PrimaryButton>
    </div>
  );
}
