// The single sink for the toast bus (lib/toast.ts). Mounted once at the shell.
// Renders a bottom-right stack of non-blocking notices that auto-dismiss and
// can be closed. Styled with the console tokens: coral for errors, phosphor for
// success, sky for info.
import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Info, X } from "lucide-react";
import { onToast, type ToastKind, type ToastMessage } from "../../lib/toast";

const TTL_MS = 6000;

const KIND: Record<ToastKind, { icon: typeof Info; border: string; ictint: string }> = {
  error:   { icon: AlertTriangle, border: "border-coral-500/40",    ictint: "text-coral-300" },
  success: { icon: CheckCircle2,  border: "border-phosphor-500/40", ictint: "text-phosphor-300" },
  info:    { icon: Info,          border: "border-skyx-500/40",     ictint: "text-skyx-300" },
};

export function ToastHost() {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  useEffect(
    () =>
      onToast((t) => {
        setToasts((prev) => [...prev, t]);
        window.setTimeout(
          () => setToasts((prev) => prev.filter((x) => x.id !== t.id)),
          TTL_MS,
        );
      }),
    [],
  );

  const dismiss = (id: string) =>
    setToasts((prev) => prev.filter((x) => x.id !== id));

  if (!toasts.length) return null;

  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-[120] flex w-[min(24rem,calc(100vw-2rem))] flex-col gap-2">
      {toasts.map((t) => {
        const k = KIND[t.kind];
        const Icon = k.icon;
        return (
          <div
            key={t.id}
            role="alert"
            className={`pointer-events-auto flex items-start gap-2.5 rounded-xl border ${k.border} bg-panel/95 px-3 py-2.5 shadow-lg backdrop-blur transition`}
          >
            <Icon size={15} className={`mt-0.5 shrink-0 ${k.ictint}`} />
            <p className="min-w-0 flex-1 whitespace-pre-wrap break-words text-xs leading-relaxed text-ink">
              {t.message}
            </p>
            <button
              type="button"
              onClick={() => dismiss(t.id)}
              aria-label="Dismiss"
              className="shrink-0 rounded p-0.5 text-dim transition hover:text-ink"
            >
              <X size={13} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
