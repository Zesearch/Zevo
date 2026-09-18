import { X } from "lucide-react";
import { DialogOwner, useDialogFocus } from "../lib/dialog";

export function Modal({
  open,
  title,
  onClose,
  children,
  width = "max-w-xl",
  kicker = "",
  badge,
  footer,
}: {
  open: boolean;
  title: string;
  /** Sits beside the title — a status that qualifies the name, not a row. */
  badge?: React.ReactNode;
  onClose: () => void;
  children: React.ReactNode;
  width?: string;
  /** Actions that must stay in view while the body scrolls: rendered under
   *  the scroller, so a long form never hides its own submit button. */
  footer?: React.ReactNode;
  /** Small label above the title. Off by default: the title already says what
   *  the dialog is, and "Console" above every one of them said nothing. */
  kicker?: string;
}) {
  const { id, ref } = useDialogFocus(open, onClose);

  if (!open) return null;
  return (
    <DialogOwner.Provider value={id}><div
      className="fixed inset-0 z-50 grid place-items-center bg-canvas/80 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        ref={ref} role="dialog" aria-modal="true" aria-labelledby={`${id}-title`} tabIndex={-1}
        className={`bezel relative flex w-full ${width} max-h-[85vh] flex-col animate-zevo-in overflow-hidden p-6`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Keep dialog chrome outside the content scroller. Wide tables may
            scroll horizontally, but the title rule and close button must
            always remain aligned to the visible dialog viewport. */}
        <div className="mb-5 flex shrink-0 items-center justify-between border-b border-hair pb-4">
          <div className="min-w-0">
            {kicker && <div className="kicker">{kicker}</div>}
            <div className="mt-1 flex flex-wrap items-center gap-3">
              <h2 id={`${id}-title`} className="font-display text-lg font-semibold tracking-tight text-ink">
                {title}
              </h2>
              {badge}
            </div>
          </div>
          <button className="btn shrink-0 !px-2 !py-2" onClick={onClose} aria-label="close">
            <X size={15} />
          </button>
        </div>
        <div className="min-h-0 overflow-auto overscroll-contain">
          {children}
        </div>
        {footer && (
          <div className="mt-4 shrink-0 border-t border-hair pt-4">{footer}</div>
        )}
      </div>
    </div></DialogOwner.Provider>
  );
}
