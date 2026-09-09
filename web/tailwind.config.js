/** @type {import('tailwindcss').Config} */
//
// ═══════════════════════════════════════════════════════════════════════════
//  ZEVO — a mission console for self-improving models.
//  Aesthetic: analog observatory / instrument console. Deep observatory-night
//  canvas, sodium-brass primary, phosphor-mint live signals, coral alerts.
//
//  Two layers of tokens:
//   1. SEMANTIC tokens (canvas/panel/ink/brass/phosphor/…) — used by the hand
//      built ZEVO shell + signature screens for precise, readable intent.
//   2. A REMAP of Tailwind's stock scales (slate/amber/green/rose/sky/…) onto
//      the same observatory ramps, so any page authored against the original
//      dark-slate theme inherits the console look coherently without a rewrite.
// ═══════════════════════════════════════════════════════════════════════════
//
const NIGHT = {
  50:  "#F3F7FA",
  100: "#E9EEF2",
  200: "#D3DDE5",
  300: "#B2C0CC",
  400: "#8B9CAB",
  500: "#66788A", // dim label text
  600: "#3B4C5B",
  700: "#293745", // raised hairline
  800: "#1C2731", // hairline / border
  900: "#10161E", // panel
  950: "#0A0E13", // deepest / canvas
};
const BRASS = {
  100: "#FBEDD6",
  200: "#F7D9AE",
  300: "#F2C185",
  400: "#ECA758",
  500: "#E08D38", // primary action / sodium lamp
  600: "#C6762A",
  700: "#9E5C1F",
};
const PHOSPHOR = {
  100: "#D6F7EE",
  200: "#A9EEDC",
  300: "#7DE8CE",
  400: "#4FD1B5", // live / running / healthy
  500: "#2BB89C",
  600: "#1E8F79",
};
const CORAL = {
  100: "#FBD9D1",
  200: "#F7B7A9",
  300: "#F79A88",
  400: "#F26F58", // alert / failed
  500: "#E24E36",
  600: "#B93A26",
};
const SKYX = {
  100: "#D6EDF9",
  200: "#AEDCF1",
  300: "#8FCDEB",
  400: "#5FB0DD", // ready / info
  500: "#3D93C6",
  600: "#2C6F98",
};
const LILAC = {
  100: "#E7DDF7",
  200: "#D3C2EF",
  300: "#C3B0E8",
  400: "#A98FD6", // degraded / auto-improve
  500: "#8F6FC4",
  600: "#6F52A0",
};
const GOLD = {
  300: "#EED08A",
  400: "#E6BB5B",
  500: "#E0B23C", // chart accent
};

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // ── Semantic (preferred in ZEVO-native code) ──
        canvas: NIGHT[950],
        panel: NIGHT[900],
        raised: "#141C25",
        hair: NIGHT[800],
        ink: NIGHT[100],
        dim: NIGHT[500],
        brass: BRASS,
        phosphor: PHOSPHOR,
        coral: CORAL,
        skyx: SKYX,
        lilac: LILAC,
        gold: GOLD,

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
          halted:      CORAL[400],
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
        bezel: "inset 0 1px 0 0 rgba(255,255,255,0.045), 0 1px 2px rgba(0,0,0,0.5), 0 14px 34px -18px rgba(0,0,0,0.7)",
        lamp: "0 0 0 1px rgba(255,255,255,0.04)",
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
