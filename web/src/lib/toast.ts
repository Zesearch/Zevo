// Tiny global toast bus, same shape as lib/commands.ts: any component can
// `toast("...")` and the single <ToastHost/> mounted at the shell renders it,
// without a context provider or prop-drilling. Replaces the raw window.alert
// error popups with a non-blocking, dismissible notice that matches the console
// aesthetic.

export type ToastKind = "error" | "success" | "info";

export type ToastMessage = {
  id: string;
  message: string;
  kind: ToastKind;
};

const target = new EventTarget();
let seq = 0;

/** Show a toast. Returns its id (so a caller could dismiss it early). */
export function toast(message: string, kind: ToastKind = "info"): string {
  const id = `toast-${++seq}`;
  target.dispatchEvent(
    new CustomEvent("zevo-toast", { detail: { id, message, kind } }),
  );
  return id;
}

/** Subscribe to toasts; returns an unsubscribe. Used only by <ToastHost/>. */
export function onToast(handler: (t: ToastMessage) => void): () => void {
  const listener = (e: Event) => handler((e as CustomEvent).detail as ToastMessage);
  target.addEventListener("zevo-toast", listener);
  return () => target.removeEventListener("zevo-toast", listener);
}
