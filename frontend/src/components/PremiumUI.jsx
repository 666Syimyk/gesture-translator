export function TitleBlock({ eyebrow, title, subtitle, action }) {
  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
      <div className="min-w-0">
        {eyebrow ? (
          <div className="mono text-[11px] font-semibold uppercase tracking-[0.28em] text-[var(--accent-strong)]/80">
            {eyebrow}
          </div>
        ) : null}
        <h2 className="mt-2 max-w-3xl text-[clamp(1.6rem,2.2vw,2.6rem)] font-extrabold tracking-[-0.04em] text-white">
          {title}
        </h2>
        {subtitle ? (
          <p className="mt-3 max-w-3xl text-sm leading-7 text-[var(--text-soft)]">
            {subtitle}
          </p>
        ) : null}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  );
}

export function PrimaryButton({
  children,
  onClick,
  className = "",
  type = "button",
  disabled = false,
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`group relative overflow-hidden rounded-full border border-white/10 bg-[linear-gradient(135deg,rgba(138,246,208,0.95),rgba(88,217,255,0.95))] px-5 py-4 text-base font-extrabold text-slate-950 shadow-[0_18px_40px_rgba(88,217,255,0.22)] transition duration-300 hover:-translate-y-[1px] hover:shadow-[0_22px_50px_rgba(88,217,255,0.28)] disabled:cursor-not-allowed disabled:opacity-55 disabled:hover:translate-y-0 ${className}`}
    >
      <span className="pointer-events-none absolute inset-0 bg-[linear-gradient(120deg,transparent,rgba(255,255,255,0.45),transparent)] opacity-0 transition duration-500 group-hover:opacity-100" />
      <span className="relative flex items-center justify-center gap-2">{children}</span>
    </button>
  );
}

export function SecondaryButton({
  children,
  onClick,
  className = "",
  type = "button",
  disabled = false,
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`group relative overflow-hidden rounded-full border border-white/10 bg-white/[0.045] px-5 py-4 text-base font-semibold text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] backdrop-blur-xl transition duration-300 hover:-translate-y-[1px] hover:border-white/16 hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-55 disabled:hover:translate-y-0 ${className}`}
    >
      <span className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.12),transparent_60%)] opacity-0 transition duration-300 group-hover:opacity-100" />
      <span className="relative flex items-center justify-center gap-2">{children}</span>
    </button>
  );
}

export function GlassCard({ children, className = "" }) {
  return (
    <div
      className={`group app-surface glow-ring relative rounded-[30px] p-5 transition duration-300 hover:-translate-y-[2px] hover:border-white/10 ${className}`}
    >
      <div className="pointer-events-none absolute inset-x-8 top-0 h-px bg-[linear-gradient(90deg,transparent,rgba(138,246,208,0.55),transparent)] opacity-80" />
      <div className="pointer-events-none absolute -right-14 top-0 h-28 w-28 rounded-full bg-[var(--accent-glow)] blur-3xl opacity-35 transition duration-500 group-hover:opacity-55" />
      <div className="relative">{children}</div>
    </div>
  );
}

export function StatusBadge({ children, className = "" }) {
  return (
    <div
      className={`mono inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-soft)] backdrop-blur-xl ${className}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current opacity-80" />
      <span>{children}</span>
    </div>
  );
}
