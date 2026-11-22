// Theme = which colour variable set the document uses. The stylesheet owns
// both sets; this module only flips the attribute and remembers the choice.
// index.html applies the same rule inline before first paint.
import { useEffect, useState } from "react";

export type Theme = "dark" | "light";
const KEY = "zevo-theme";

function read(): Theme {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "light" || saved === "dark") return saved;
    return matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  } catch {
    return "dark";
  }
}

export function applyTheme(theme: Theme) {
  const root = document.documentElement;
  if (theme === "light") root.setAttribute("data-theme", "light");
  else root.removeAttribute("data-theme");
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    /* private mode: the choice lasts for this page only */
  }
}

export function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setThemeState] = useState<Theme>(read);
  useEffect(() => {
    const mq = matchMedia("(prefers-color-scheme: light)");
    // Follow the OS only while the reader has not chosen for themselves.
    const follow = () => {
      let chosen: string | null = null;
      try {
        chosen = localStorage.getItem(KEY);
      } catch {
        /* unreadable storage counts as no choice */
      }
      if (chosen) return;
      const next: Theme = mq.matches ? "light" : "dark";
      setThemeState(next);
      if (next === "light") document.documentElement.setAttribute("data-theme", "light");
      else document.documentElement.removeAttribute("data-theme");
    };
    mq.addEventListener("change", follow);
    return () => mq.removeEventListener("change", follow);
  }, []);
  const setTheme = (t: Theme) => {
    applyTheme(t);
    setThemeState(t);
  };
  return [theme, setTheme];
}

// Light-theme styles: a palette and a type set, one attribute each. The
// stylesheet carries the blocks; this module flips the attributes.
export const LIGHT_VARIANTS = ["cool", "paper", "plate", "shade"] as const;
export type LightVariant = (typeof LIGHT_VARIANTS)[number];
export const TYPE_SETS = ["console", "site"] as const;
export type TypeSet = (typeof TYPE_SETS)[number];

function attrState<T extends string>(attr: string, key: string, allowed: readonly T[]) {
  const read = (): T => {
    try {
      const saved = localStorage.getItem(key);
      if (saved && (allowed as readonly string[]).includes(saved)) return saved as T;
    } catch {
      /* fall through to the default */
    }
    return allowed[0];
  };
  const apply = (value: T) => {
    document.documentElement.setAttribute(attr, value);
    try {
      localStorage.setItem(key, value);
    } catch {
      /* page-only */
    }
  };
  return [read, apply] as const;
}

export const [readLightVariant, applyLightVariant] = attrState<LightVariant>("data-light", "zevo-light-variant", LIGHT_VARIANTS);
export const [readTypeSet, applyTypeSet] = attrState<TypeSet>("data-type", "zevo-type", TYPE_SETS);
