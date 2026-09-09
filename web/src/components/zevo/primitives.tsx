// ZEVO shared instrument primitives. Import these across pages so the whole
// console shares one visual language (bezels, kickers, gauges, readouts).
import { useId, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Link } from "react-router-dom";

export function Kicker({
  children, className = "", strong = false,
}: {
  children: ReactNode; className?: string; strong?: boolean;
}) {
  // `!` needed: .kicker is declared after @tailwind utilities and would
  // otherwise win the colour at equal specificity.
  return <div className={`kicker ${strong ? "!text-ink" : ""} ${className}`}>{children}</div>;
}

export function Bezel({
  children,
  className = "",
  flat = false,
  id,
  onClick,
}: {
  children: ReactNode;
  className?: string;
  flat?: boolean;
  /** Set when the panel is a deep-link target (e.g. /levels#L3). */
  id?: string;
  /** Set when the whole panel is the target — a task card opening its own
   *  details. Give it a cursor and a hover border where you use it. */
  onClick?: (e: React.MouseEvent<HTMLDivElement>) => void;
}) {
  return (
    <div id={id} onClick={onClick} className={`${flat ? "bezel-flat" : "bezel"} ${className}`}>
      {children}
    </div>
  );
}

/** Page header: title + one-line blurb.
 *
 *  The tracked kicker that used to sit above the title is gone — the title
 *  itself now carries that voice, set in the same Spline Sans Mono the kicker
 *  used. Colour and size are unchanged from the old display-face title. */
export function PageHead({
  title,
  subtitle,
  subtitleTitle,
  right,
}: {
  /** Usually a string; a node when the name carries something beside it, e.g.
   *  a run's name followed by its id. */
  title: ReactNode;
  subtitle?: ReactNode;
  /** Hover text for the subtitle, where the short form shown is a trim of
   *  something longer and the whole of it is still worth reaching. */
  subtitleTitle?: string;
  right?: ReactNode;
}) {
  return (
    <div className="mb-8 flex flex-wrap items-end justify-between gap-4 animate-zevo-in">
      <div className="min-w-0">
        <h1 className="font-mono text-[2.1rem] font-semibold leading-none tracking-tight text-ink">
          {title}
        </h1>
        {subtitle && (
          <p className="mt-2 text-sm text-slate-400" title={subtitleTitle}>{subtitle}</p>
        )}
      </div>
      {right && <div className="flex shrink-0 items-center gap-2">{right}</div>}
    </div>
  );
}

/** An instrument readout: label above, large tabular numeral, optional unit + hint.
 *  `size="lg"` is the console's dial scale; every other page uses the default.
 *  The label needs `!` because .kicker is declared after @tailwind utilities and
 *  would otherwise win the font-size at equal specificity. */
export function Readout({
  label,
  value,
  unit,
  hint,
  tone = "ink",
  size = "md",
  strongLabel = false,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  hint?: ReactNode;
  tone?: "ink" | "brass" | "phosphor" | "coral" | "sky";
  size?: "md" | "card" | "stat" | "statLg" | "lg";
  // A readout that heads a card of its own, rather than labelling a field
  // inside one, wants its label to read as a heading.
  strongLabel?: boolean;
}) {
  const toneCls = {
    ink: "text-ink",
    brass: "text-brass-300",
    phosphor: "text-phosphor-300",
    coral: "text-coral-300",
    sky: "text-skyx-300",
  }[tone];
  const s = {
    md: { label: "", value: "text-4xl", unit: "text-sm", hint: "text-2xs" },
    card: { label: "!text-sm", value: "text-4xl", unit: "text-sm", hint: "text-sm" },
    // Console label scale, but a restrained numeral — for stat clusters that sit
    // inline with a heading rather than filling a card of their own.
    stat: { label: "", value: "text-2xl", unit: "text-sm", hint: "text-sm" },
    // A stat cluster carrying a panel on its own: card-sized label, numeral one
    // step under `card` so it reads as a heading's companion, not a tile.
    statLg: { label: "!text-2xs", value: "text-3xl", unit: "text-sm", hint: "text-sm" },
    lg: { label: "!text-sm", value: "text-5xl", unit: "text-lg", hint: "text-sm" },
  }[size];
  return (
    <div>
      <Kicker strong={strongLabel} className={s.label}>{label}</Kicker>
      <div className="mt-2 flex items-baseline gap-1.5">
        <span className={`readout ${s.value} font-semibold ${toneCls}`}>{value}</span>
        {unit && <span className={`readout ${s.unit} text-slate-500`}>{unit}</span>}
      </div>
      {hint && <div className={`mt-1 ${s.hint} text-slate-500`}>{hint}</div>}
    </div>
  );
}

/** A radial gauge dial (0..1 fraction), brass needle sweeping a phosphor arc. */
export function Gauge({
  value,
  label,
  display,
  size = 128,
  tone = "phosphor",
  displayScale = 0.19,
  bounded = true,
}: {
  value: number;
  /** Omit for a bare dial — used where the surrounding card already names it. */
  label?: string;
  display: string;
  size?: number;
  tone?: "phosphor" | "brass" | "coral";
  /** Numeral size as a fraction of `size`. */
  displayScale?: number;
  /** False for raw-scale metrics whose unknown domain cannot drive a 0..1 arc. */
  bounded?: boolean;
}) {
  const clamped = Math.max(0, Math.min(1, isFinite(value) ? value : 0));
  const r = size / 2 - 12;
  const cx = size / 2;
  const cy = size / 2;
  // 270° sweep starting at 135°.
  const start = 135;
  const total = 270;
  const arc = (frac: number) => {
    const a0 = (start * Math.PI) / 180;
    const a1 = ((start + total * frac) * Math.PI) / 180;
    const x0 = cx + r * Math.cos(a0);
    const y0 = cy + r * Math.sin(a0);
    const x1 = cx + r * Math.cos(a1);
    const y1 = cy + r * Math.sin(a1);
    const large = total * frac > 180 ? 1 : 0;
    return `M ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1}`;
  };
  const stroke = { phosphor: "#4FD1B5", brass: "#E08D38", coral: "#F26F58" }[tone];
  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size} className="overflow-visible">
        <path d={arc(1)} fill="none" stroke="#1C2731" strokeWidth={9} strokeLinecap="round" />
        {bounded && (
          <path
            d={arc(clamped)}
            fill="none"
            stroke={stroke}
            strokeWidth={9}
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 6px ${stroke}88)` }}
          />
        )}
        <text
          x={cx}
          y={label ? cy - 2 : cy + size * 0.06}
          textAnchor="middle"
          className="readout"
          fill="#E9EEF2"
          fontSize={size * displayScale}
          fontWeight={600}
        >
          {display}
        </text>
        {label && (
          <text x={cx} y={cy + size * 0.15} textAnchor="middle" fill="#66788A" fontSize={size * 0.085} className="kicker">
            {label}
          </text>
        )}
      </svg>
    </div>
  );
}

/** A note marker: the explanation for the thing beside it, one hover away.
 *
 *  Several places on the console show a number or a label whose meaning is not
 *  self-evident — why a GPU cost is $0, what the two ends of the loop are, what
 *  an autonomy level means. Each had grown its own inline gloss ("e.g. …", a
 *  parenthetical, a whole sentence under the title), which pushed the reading
 *  order around and made the pages noisier the more they explained.
 *
 *  Drawn as an icon rather than a bordered "!" character: a text glyph inside a
 *  CSS circle never centres the same way twice — it rides the font's baseline,
 *  not the circle's centre. An "i", not a "!": the mark says "there is more to
 *  read here", which is information, not a warning, and coral already means
 *  something is wrong on these pages.
 *
 *  The explanation is drawn, not left to the browser's `title`. A native
 *  tooltip waits about a second before appearing, renders in the OS's font at
 *  the pointer, and never shows at all for a keyboard user — which for a mark
 *  whose entire job is to explain meant most people never saw the explanation.
 *  It is portalled to `body` and positioned from the trigger's own rect, since
 *  every place one of these sits is inside a panel that clips its overflow.
 *
 *  Pass `to` to make it a link as well — used where the full answer is a page
 *  of its own rather than a sentence.
 */
/** A filled disc with the "i" knocked out of it.
 *
 *  An outlined circle with an outlined "i" inside is two hairlines fighting for
 *  the same 15 pixels, and the letter is the one that loses — at this size the
 *  dot and the stem read as noise inside the ring. Inverting it gives the mark
 *  one solid shape and lets the counter do the work, and knocking the letter
 *  THROUGH the disc (rather than painting it) keeps it legible on the panel,
 *  the canvas and the raised surfaces alike, since it is simply transparent.
 */
function InfoGlyph({ size }: { size: number }) {
  const id = useId();
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden focusable="false">
      <mask id={id}>
        <rect width="16" height="16" fill="#fff" />
        <circle cx="8" cy="4.55" r="1.15" fill="#000" />
        <rect x="6.9" y="6.9" width="2.2" height="5.4" rx="1.1" fill="#000" />
      </mask>
      <circle cx="8" cy="8" r="7.4" fill="currentColor" mask={`url(#${id})`} />
    </svg>
  );
}

export function Note({
  children,
  to,
  size = 15,
  className = "",
  tooltipClassName = "",
}: {
  /** The explanation. Shown on hover, on focus, and to screen readers. */
  children: string;
  /** Optional route to open on click. */
  to?: string;
  /** Match the text it annotates: 15 beside body copy, 14 in a table head. */
  size?: number;
  className?: string;
  tooltipClassName?: string;
}) {
  const ref = useRef<HTMLElement | null>(null);
  const [box, setBox] = useState<{ top: number; left: number; below: boolean } | null>(null);

  function show() {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    // Below by default; above when the panel would run off the bottom.
    const below = r.bottom + 96 < window.innerHeight;
    setBox({
      top: below ? r.bottom + 8 : r.top - 8,
      left: Math.min(Math.max(r.left + r.width / 2, 160), window.innerWidth - 160),
      below,
    });
  }
  const hide = () => setBox(null);

  // `align-middle` on an inline-flex box centres it on the text's own middle
  // line, which is what puts it level with the words either side of it. The
  // earlier baseline nudge was a constant, so it only ever looked right at one
  // font size.
  const shared =
    "inline-flex shrink-0 items-center justify-center align-middle " +
    "text-slate-400 transition-colors hover:text-brass-200 focus:outline-none " +
    `focus-visible:text-brass-200 ${className}`;
  const handlers = {
    onMouseEnter: show,
    onMouseLeave: hide,
    onFocus: show,
    onBlur: hide,
  };
  const glyph = <InfoGlyph size={size} />;

  const panel = box
    ? createPortal(
        <div
          role="tooltip"
          style={{
            top: box.top,
            left: box.left,
            transform: box.below ? "translateX(-50%)" : "translate(-50%, -100%)",
          }}
          className={`pointer-events-none fixed z-[100] max-w-xs rounded-bezel border border-hair
                     bg-panel px-3 py-2 text-2xs leading-relaxed text-slate-200 shadow-lg ${tooltipClassName}`}
        >
          {children}
        </div>,
        document.body,
      )
    : null;

  if (to) {
    return (
      <>
        <Link
          ref={ref as React.Ref<HTMLAnchorElement>}
          to={to}
          aria-label={children}
          className={shared}
          {...handlers}
        >
          {glyph}
        </Link>
        {panel}
      </>
    );
  }
  return (
    <>
      <span
        ref={ref as React.Ref<HTMLSpanElement>}
        role="note"
        tabIndex={0}
        aria-label={children}
        className={`${shared} cursor-help`}
        {...handlers}
      >
        {glyph}
      </span>
      {panel}
    </>
  );
}

/**
 * One labelled block in a dialog. `section` is for the ones that name a half of
 * the dialog — they carry the accent and the size, so the eye finds them before
 * the text under them.
 *
 * Shared rather than page-local because the settings table appears in two
 * dialogs (a task's own, and the leaderboard's key to the Setting column) and
 * they have to be framed the same way or the same table reads as two things.
 */
export function Detail({ label, children, className = "", section = false, hideLabel = false }: {
  label: string; children: React.ReactNode; className?: string; section?: boolean; hideLabel?: boolean;
}) {
  return (
    <div className={`min-w-0 ${className}`}>
      {!hideLabel && (
        <div className={
          section
            ? "font-mono text-sm uppercase tracking-[0.16em] text-brass-300"
            : "font-mono text-2xs uppercase tracking-[0.16em] text-slate-500"
        }>
          {label}
        </div>
      )}
      <div className={`${hideLabel ? "" : section ? "mt-3" : "mt-1.5"} min-w-0`}>{children}</div>
    </div>
  );
}
