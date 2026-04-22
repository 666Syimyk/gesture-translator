import {
  GlassCard,
  SecondaryButton,
  StatusBadge,
  TitleBlock,
} from "../components/PremiumUI";

function getTypeLabel(type) {
  if (type === "gesture") {
    return "Жест";
  }

  if (type === "voice") {
    return "Голос";
  }

  return "Фраза";
}

export default function HistoryPage({
  history,
  clearHistory,
  isLoading,
  error,
  settings,
}) {
  return (
    <div className="space-y-4">
      <TitleBlock
        eyebrow="Последние действия"
        title="История"
        subtitle="Жесты, голос и словарь собираются в одном журнале."
        action={
          <SecondaryButton onClick={clearHistory} className="px-4 py-2 text-sm">
            Очистить
          </SecondaryButton>
        }
      />

      <GlassCard>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-sm font-bold text-white">Состояние журнала</div>
            <div className="mt-1 text-sm text-slate-300">
              Backend хранит общую историю сообщений и возвращает её после
              перезагрузки.
            </div>
          </div>
          <StatusBadge className="text-cyan-300">{history.length} записей</StatusBadge>
        </div>
      </GlassCard>

      {error ? (
        <div className="rounded-[20px] border border-red-400/20 bg-red-500/10 p-4 text-sm text-red-200">
          {error}
        </div>
      ) : null}

      {isLoading ? (
        <GlassCard className="text-sm text-slate-300">Загрузка истории...</GlassCard>
      ) : history.length === 0 ? (
        <GlassCard className="text-sm text-slate-300">
          История пока пустая.
        </GlassCard>
      ) : (
        <div className="space-y-3">
          {history.map((item) => (
            <GlassCard key={item.id}>
              <div className="flex items-center justify-between gap-3">
                <StatusBadge className="border-cyan-400/20 bg-cyan-400/10 text-cyan-300">
                  {getTypeLabel(item.type)}
                </StatusBadge>
                <div className="text-xs font-semibold text-slate-500">
                  {item.time}
                </div>
              </div>

              <div
                className={`mt-3 font-black leading-tight text-white ${
                  settings.largeTextEnabled ? "text-3xl" : "text-2xl"
                }`}
              >
                {item.text}
              </div>
            </GlassCard>
          ))}
        </div>
      )}
    </div>
  );
}
