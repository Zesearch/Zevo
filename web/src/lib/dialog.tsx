import { createContext, useContext, useEffect, useId, useRef } from "react";

export const DialogOwner = createContext("");
export const useDialogOwner = () => useContext(DialogOwner);
const stack: Array<{ id: string; layer: number }> = [];
const focusable = 'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** One focus boundary shared by dialogs, the command palette, and portalled controls. */
export function useDialogFocus(open: boolean, close: () => void, layer = 50) {
  const id = useId();
  const ref = useRef<HTMLDivElement>(null);
  const closeRef = useRef(close);
  closeRef.current = close;
  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    const entry = { id, layer };
    stack.push(entry);
    const isTop = () => [...stack].sort((a, b) => a.layer - b.layer).at(-1) === entry;
    const portals = () => Array.from(document.querySelectorAll<HTMLElement>(`[data-dialog-owner="${CSS.escape(id)}"]`));
    const contains = (node: Node | null) => !!node && (ref.current?.contains(node) || portals().some((p) => p.contains(node)));
    const elements = () => [ref.current, ...portals()].flatMap((root) => root ? Array.from(root.querySelectorAll<HTMLElement>(focusable)) : []).filter((el) => el.getClientRects().length > 0 && !el.closest('[inert]'));
    const focusFirst = () => (elements()[0] ?? ref.current)?.focus();
    const timer = requestAnimationFrame(() => { if (isTop() && !contains(document.activeElement)) focusFirst(); });
    const key = (event: KeyboardEvent) => {
      if (!isTop() || event.defaultPrevented) return;
      if (event.key === "Escape") {
        event.preventDefault(); event.stopPropagation(); closeRef.current();
      }
      if (event.key !== "Tab") return;
      const list = elements();
      const at = list.indexOf(document.activeElement as HTMLElement);
      if (!list.length) { event.preventDefault(); ref.current?.focus(); }
      else if (event.shiftKey && at <= 0) { event.preventDefault(); list.at(-1)?.focus(); }
      else if (!event.shiftKey && (at < 0 || at === list.length - 1)) { event.preventDefault(); list[0].focus(); }
    };
    const focus = (event: FocusEvent) => {
      if (isTop() && !contains(event.target as Node)) focusFirst();
    };
    document.addEventListener("keydown", key);
    document.addEventListener("focusin", focus);
    return () => {
      cancelAnimationFrame(timer);
      document.removeEventListener("keydown", key);
      document.removeEventListener("focusin", focus);
      stack.splice(stack.indexOf(entry), 1);
      const hadFocus = contains(document.activeElement) || document.activeElement === document.body;
      requestAnimationFrame(() => {
        if (hadFocus && previous?.isConnected && (!document.activeElement || document.activeElement === document.body)) previous.focus();
      });
    };
  }, [open, id, layer]);
  return { id, ref };
}
