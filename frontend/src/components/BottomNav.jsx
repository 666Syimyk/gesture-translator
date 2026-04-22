const NAV_ITEMS = [
  { key: "camera", label: "Камера" },
  { key: "voice", label: "Голос" },
  { key: "training", label: "Обучение" },
];

export default function BottomNav({ screen, setScreen, className = "" }) {
  return (
    <div
      className={`app-surface rounded-full p-2 shadow-[0_18px_60px_rgba(0,0,0,0.35)] ${className}`}
    >
      <div className="grid grid-cols-3 gap-2">
        {NAV_ITEMS.map((item) => {
          const isActive = screen === item.key;

          return (
            <button
              key={item.key}
              type="button"
              onClick={() => setScreen(item.key)}
              aria-current={isActive ? "page" : undefined}
              className={`rounded-full px-2 py-3 text-center text-sm font-semibold transition duration-300 ${
                isActive
                  ? "bg-white/[0.085] text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]"
                  : "text-[var(--text-soft)] hover:bg-white/[0.05] hover:text-white"
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
