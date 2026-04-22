import { useEffect, useRef, useState } from "react";
import {
  GlassCard,
  PrimaryButton,
  SecondaryButton,
  StatusBadge,
  TitleBlock,
} from "../components/PremiumUI";

const QUICK_RESPONSES = [
  { id: 1, label: "Сейчас я вам помогу", category: "Поддержка" },
  { id: 2, label: "Подождите минуту", category: "Поддержка" },
  { id: 3, label: "Пожалуйста, повторите ещё раз", category: "Уточнение" },
  { id: 4, label: "Покажите, что вам нужно", category: "Уточнение" },
  { id: 5, label: "Вам нужна вода?", category: "Забота" },
  { id: 6, label: "Я вызову врача", category: "Забота" },
];

function mapRecognitionError(errorCode) {
  switch (errorCode) {
    case "not-allowed":
    case "service-not-allowed":
      return "Микрофон недоступен. Разрешите доступ в браузере.";
    case "audio-capture":
      return "Не удалось получить звук с микрофона.";
    case "no-speech":
      return "Речь не обнаружена. Попробуйте сказать фразу ещё раз.";
    case "network":
      return "Ошибка сети при распознавании речи.";
    case "aborted":
      return "Запись остановлена.";
    default:
      return `Ошибка распознавания: ${errorCode}`;
  }
}

export default function VoicePage({
  spokenText,
  setSpokenText,
  addToHistory,
  settings,
}) {
  const speechRecognitionSupported =
    typeof window !== "undefined" &&
    !!(window.SpeechRecognition || window.webkitSpeechRecognition);

  const recognitionRef = useRef(null);
  const [isListening, setIsListening] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState(
    speechRecognitionSupported
      ? "Готов к записи"
      : "Браузер не поддерживает распознавание речи",
  );
  const [interimText, setInterimText] = useState("");
  const [isSupported] = useState(speechRecognitionSupported);

  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.onstart = null;
        recognitionRef.current.onresult = null;
        recognitionRef.current.onerror = null;
        recognitionRef.current.onend = null;
        recognitionRef.current.stop();
      }
    };
  }, []);

  function saveVoiceText(text) {
    const cleanText = text.trim();

    if (!cleanText) {
      return;
    }

    setInterimText("");
    setSpokenText(cleanText);
    addToHistory("voice", cleanText);
    setVoiceStatus("Фраза распознана");
  }

  function startVoiceRecognition() {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setVoiceStatus("Браузер не поддерживает распознавание речи");
      return;
    }

    if (isListening) {
      return;
    }

    if (recognitionRef.current) {
      recognitionRef.current.onstart = null;
      recognitionRef.current.onresult = null;
      recognitionRef.current.onerror = null;
      recognitionRef.current.onend = null;
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "ru-RU";
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setIsListening(true);
      setInterimText("");
      setVoiceStatus("Слушаю и распознаю речь...");
    };

    recognition.onresult = (event) => {
      let finalText = "";
      let nextInterimText = "";

      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const transcript = event.results[i][0].transcript;

        if (event.results[i].isFinal) {
          finalText += transcript;
        } else {
          nextInterimText += transcript;
        }
      }

      if (finalText.trim()) {
        saveVoiceText(finalText);
      } else if (nextInterimText.trim()) {
        const draft = nextInterimText.trim();
        setInterimText(draft);
        setSpokenText(draft);
        setVoiceStatus("Слушаю... подтверждаю фразу");
      }
    };

    recognition.onerror = (event) => {
      setIsListening(false);
      setInterimText("");
      setVoiceStatus(mapRecognitionError(event.error));
      recognitionRef.current = null;
    };

    recognition.onend = () => {
      setIsListening(false);

      if (recognitionRef.current === recognition) {
        recognitionRef.current = null;
      }

      setVoiceStatus((currentStatus) => {
        if (
          currentStatus === "Слушаю и распознаю речь..." ||
          currentStatus === "Слушаю... подтверждаю фразу"
        ) {
          return "Готов к повторной записи";
        }

        return currentStatus;
      });
    };

    recognitionRef.current = recognition;
    recognition.start();
  }

  function stopVoiceRecognition() {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }

    setIsListening(false);
    setInterimText("");
    setVoiceStatus("Запись остановлена");
  }

  function clearVoiceText() {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }

    setIsListening(false);
    setInterimText("");
    setSpokenText("...");
    setVoiceStatus(
      isSupported ? "Готов к записи" : "Браузер не поддерживает распознавание речи",
    );
  }

  return (
    <div className="space-y-4">
      <TitleBlock
        eyebrow="Распознавание"
        title="Голос"
        subtitle="Речь в текст для второй стороны общения"
        action={
          <StatusBadge
            className={
              isListening
                ? "border-emerald-400/20 bg-emerald-400/10 text-emerald-300"
                : ""
            }
          >
            {isListening ? "Слушаю" : "Ожидание"}
          </StatusBadge>
        }
      />

      <GlassCard className="bg-[linear-gradient(135deg,rgba(255,255,255,0.06),rgba(16,185,129,0.06))] p-5">
        <div className="text-sm text-slate-400">Статус</div>
        <div className="mt-1 text-lg font-bold text-white">{voiceStatus}</div>

        <div className="mt-5 grid grid-cols-2 gap-3">
          <PrimaryButton
            onClick={startVoiceRecognition}
            disabled={!isSupported || isListening}
          >
            Начать запись
          </PrimaryButton>

          <SecondaryButton onClick={stopVoiceRecognition} disabled={!isListening}>
            Стоп
          </SecondaryButton>
        </div>

        <SecondaryButton onClick={clearVoiceText} className="mt-3 w-full">
          Очистить текст
        </SecondaryButton>
      </GlassCard>

      <GlassCard className="bg-[linear-gradient(180deg,rgba(255,255,255,0.035),rgba(255,255,255,0.02))]">
        <div className="flex items-center justify-between gap-3">
          <div className="mono text-[11px] font-semibold uppercase tracking-[0.24em] text-[var(--text-soft)]">
            Распознанный текст
          </div>
          {interimText ? (
            <div className="rounded-full border border-emerald-400/20 bg-emerald-500/10 px-3 py-1 text-xs font-bold text-emerald-300">
              Черновик
            </div>
          ) : null}
        </div>

        <div
          className={`mt-4 font-black leading-tight ${
            settings.largeTextEnabled ? "text-5xl text-white" : "text-4xl text-white"
          }`}
        >
          {spokenText}
        </div>

        <div className="mt-4 text-sm leading-7 text-[var(--text-soft)]">
          Говорите короткими фразами и делайте небольшую паузу после окончания
          речи, чтобы приложение успело подтвердить результат.
        </div>
      </GlassCard>

      <GlassCard>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-sm font-bold text-white">Быстрые ответы</div>
            <div className="mt-1 text-sm text-slate-300">
              Готовые фразы для второй стороны диалога, если нужно ответить сразу.
            </div>
          </div>
          <StatusBadge className="text-cyan-300">
            {QUICK_RESPONSES.length} фраз
          </StatusBadge>
        </div>

        <div className="mt-4 grid gap-2">
          {QUICK_RESPONSES.map((phrase) => (
            <button
              key={phrase.id}
              type="button"
              onClick={() => saveVoiceText(phrase.label)}
              className="rounded-[22px] border border-white/8 bg-white/[0.035] px-4 py-3 text-left transition duration-300 hover:-translate-y-[1px] hover:bg-white/[0.06]"
            >
              <div className="mono text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-300/70">
                {phrase.category}
              </div>
              <div className="mt-2 text-sm font-bold text-slate-100">
                {phrase.label}
              </div>
            </button>
          ))}
        </div>
      </GlassCard>
    </div>
  );
}
