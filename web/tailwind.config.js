/** @type {import('tailwindcss').Config} */
//
// ═══════════════════════════════════════════════════════════════════════════
//  ZEVO — a mission console for self-improving models.
//  Aesthetic: analog observatory / instrument console. Deep observatory-night
//  canvas, sodium-brass primary, phosphor-mint live signals, coral alerts.
//
//  Every colour is a CSS variable (see styles.css): the dark observatory ramps
//  live on :root and a light re-cut on :root[data-theme="light"], so one
//  class like text-brass-300 is readable on both grounds without a rewrite.
//
//  Two layers of tokens:
//   1. SEMANTIC tokens (canvas/panel/ink/brass/phosphor/…) — used by the hand
//      built ZEVO shell + signature screens for precise, readable intent.
//   2. A REMAP of Tailwind's stock scales (slate/amber/green/rose/sky/…) onto
//      the same observatory ramps, so any page authored against the original
//      dark-slate theme inherits the console look coherently without a rewrite.
// ═══════════════════════════════════════════════════════════════════════════
//
const v = (name) => `rgb(var(--${name}) / <alpha-value>)`;
const NIGHT = { 50: v("night-50"), 100: v("night-100"), 200: v("night-200"), 300: v("night-300"), 400: v("night-400"), 500: v("night-500"), 600: v("night-600"), 700: v("night-700"), 800: v("night-800"), 900: v("night-900"), 950: v("night-950") };
const BRASS = { 100: v("brass-100"), 200: v("brass-200"), 300: v("brass-300"), 400: v("brass-400"), 500: v("brass-500"), 600: v("brass-600"), 700: v("brass-700") };
const PHOSPHOR = { 100: v("phosphor-100"), 200: v("phosphor-200"), 300: v("phosphor-300"), 400: v("phosphor-400"), 500: v("phosphor-500"), 600: v("phosphor-600") };
const CORAL = { 100: v("coral-100"), 200: v("coral-200"), 300: v("coral-300"), 400: v("coral-400"), 500: v("coral-500"), 600: v("coral-600") };
const SKYX = { 100: v("skyx-100"), 200: v("skyx-200"), 300: v("skyx-300"), 400: v("skyx-400"), 500: v("skyx-500"), 600: v("skyx-600") };
const LILAC = { 100: v("lilac-100"), 200: v("lilac-200"), 300: v("lilac-300"), 400: v("lilac-400"), 500: v("lilac-500"), 600: v("lilac-600") };
const GOLD = { 300: v("gold-300"), 400: v("gold-400"), 500: v("gold-500") };

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // ── Semantic (preferred in ZEVO-native code) ──
        canvas: NIGHT[950],
        panel: NIGHT[900],
        raised: v("raised"),
        hair: NIGHT[800],
        ink: NIGHT[100],
        dim: NIGHT[500],
        brass: BRASS,
        phosphor: PHOSPHOR,
        coral: CORAL,
        skyx: SKYX,
        lilac: LILAC,
        gold: GOLD,
        // Translucent overlays (bg-white/5 hover tints) must tint TOWARD the
        // ink, so "white" is light on the dark theme and near-black on light.
        white: v("overlay"),

        // ── Remap of stock scales so inherited pages read as the console ──
        slate: NIGHT,
        gray: NIGHT,
        neutral: NIGHT,
        zinc: NIGHT,
        stone: NIGHT,
        amber: BRASS,
        yellow: BRASS,
        orange: BRASS,
        green: PHOSPHOR,
        emerald: PHOSPHOR,
        teal: PHOSPHOR,
        lime: PHOSPHOR,
        rose: CORAL,
        red: CORAL,
        pink: CORAL,
        sky: SKYX,
        blue: SKYX,
        cyan: SKYX,
        indigo: LILAC,
        violet: LILAC,
        purple: LILAC,
        fuchsia: LILAC,

        // Status palette used by badges / lamps / DAG nodes.
        status: {
          todo:        NIGHT[500],
          ready:       SKYX[400],
          in_progress: BRASS[500],
          running:     BRASS[400],   // in flight — distinct from done
          done:        PHOSPHOR[400],
          success:     PHOSPHOR[400],
          failed:      CORAL[400],
          degraded:    LILAC[400],
          skipped:     NIGHT[500],
        },
      },
      fontFamily: {
        display: ['"Bricolage Grotesque"', "ui-sans-serif", "system-ui", "sans-serif"],
        sans: ['"Instrument Sans"', "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ['"Spline Sans Mono"', "ui-monospace", "Menlo", "monospace"],
      },
      fontSize: {
        "2xs": ["0.8125rem", "1.05rem"], // 13px micro-labels
        xs:   ["0.9375rem", "1.3rem"],
        sm:   ["1.0625rem", "1.5rem"],
        base: ["1.1875rem", "1.7rem"],
        lg:   ["1.3125rem", "1.8rem"],
        xl:   ["1.5rem",    "2rem"],
      },
      letterSpacing: {
        kicker: "0.22em",
      },
      borderRadius: {
        bezel: "14px",
      },
      boxShadow: {
        // Instrument bezel: soft top-light + grounded drop.
        bezel: "var(--shadow-bezel)",
        lamp: "0 0 0 1px rgb(var(--overlay) / 0.04)",
        "glow-brass": "0 0 12px -2px rgba(224,141,56,0.55)",
        "glow-phosphor": "0 0 12px -2px rgba(79,209,181,0.6)",
        "glow-coral": "0 0 12px -2px rgba(242,111,88,0.6)",
      },
      keyframes: {
        "lamp-pulse": {
          "0%,100%": { opacity: "1" },
          "50%": { opacity: "0.35" },
        },
        "zevo-in": {
          from: { opacity: "0", transform: "translateY(8px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        sweep: {
          from: { transform: "translateX(-100%)" },
          to: { transform: "translateX(100%)" },
        },
        "dash-flow": {
          to: { "stroke-dashoffset": "-16" },
        },
      },
      animation: {
        "lamp-pulse": "lamp-pulse 1.8s ease-in-out infinite",
        "zevo-in": "zevo-in 0.5s cubic-bezier(0.16,1,0.3,1) both",
        sweep: "sweep 1.4s ease-in-out infinite",
        "dash-flow": "dash-flow 0.7s linear infinite",
      },
    },
  },
  plugins: [],
};
