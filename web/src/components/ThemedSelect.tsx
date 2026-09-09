import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Check, ChevronDown } from "lucide-react";
import { createPortal } from "react-dom";

export type ThemedSelectOption = {
  value: string;
  label: ReactNode;
  disabled?: boolean;
};

/**
 * Browser-independent single-choice menu.
 *
 * Native selects inherit Safari/Chrome platform chrome and require placeholder
 * options. This control keeps placeholder text in the closed button and puts
 * only the caller's actual options in the portalled menu. Portalling also keeps
 * the list visible inside the launch modal's scroll container.
 */
export function ThemedSelect({
  value,
  onChange,
  options,
  placeholder = "",
  ariaLabel,
  disabled = false,
  className = "",
  buttonClassName = "",
}: {
  value: string;
  onChange: (value: string) => void;
  options: ThemedSelectOption[];
  placeholder?: ReactNode;
  ariaLabel: string;
  disabled?: boolean;
  className?: string;
  buttonClassName?: string;
}) {
  const control = useRef<HTMLDivElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [position, setPosition] = useState({ left: 0, top: 0, width: 0, above: false });
  const selectedIndex = options.findIndex((o) => o.value === value);
  const selected = selectedIndex >= 0 ? options[selectedIndex] : null;
  const enabled = useMemo(() => options.map((o, i) => o.disabled ? -1 : i).filter((i) => i >= 0), [options]);

  const place = () => {
    const rect = control.current?.getBoundingClientRect();
    if (!rect) return;
    const roomBelow = window.innerHeight - rect.bottom;
    setPosition({
      left: rect.left,
      top: roomBelow >= 190 ? rect.bottom + 4 : rect.top - 4,
      width: rect.width,
      above: roomBelow < 190,
    });
  };

  useEffect(() => {
    if (!open) return;
    place();
    const away = (e: PointerEvent) => {
      const node = e.target as Node;
      if (!control.current?.contains(node) && !menu.current?.contains(node)) setOpen(false);
    };
    const move = () => place();
    document.addEventListener("pointerdown", away);
    window.addEventListener("resize", move);
    window.addEventListener("scroll", move, true);
    return () => {
      document.removeEventListener("pointerdown", away);
      window.removeEventListener("resize", move);
      window.removeEventListener("scroll", move, true);
    };
  }, [open]);

  const choose = (index: number) => {
    const option = options[index];
    if (!option || option.disabled) return;
    onChange(option.value);
    setActive(index);
    setOpen(false);
  };

  const moveActive = (delta: number) => {
    if (!enabled.length) return;
    const at = enabled.indexOf(active);
    const next = at < 0
      ? (delta > 0 ? 0 : enabled.length - 1)
      : Math.min(Math.max(at + delta, 0), enabled.length - 1);
    setActive(enabled[next]);
  };

  return (
    <div ref={control} className={`relative ${className}`}>
      <button
        type="button"
        disabled={disabled}
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => {
          if (!open) setActive(selectedIndex >= 0 ? selectedIndex : (enabled[0] ?? 0));
          setOpen((v) => !v);
        }}
        onKeyDown={(e) => {
          if (e.key === "Escape") { setOpen(false); return; }
          if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            e.preventDefault();
            if (!open) setOpen(true);
            else moveActive(e.key === "ArrowDown" ? 1 : -1);
          } else if (e.key === "Enter" && open) {
            e.preventDefault();
            choose(active);
          }
        }}
        className={`flex w-full items-center justify-between gap-2 text-left disabled:cursor-not-allowed disabled:opacity-60 ${buttonClassName}`}
      >
        <span className={`min-w-0 truncate ${selected ? "" : "text-slate-600"}`}>
          {selected?.label ?? placeholder}
        </span>
        <ChevronDown size={14} className={`shrink-0 text-slate-500 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && !disabled && createPortal(
        <div
          ref={menu}
          role="listbox"
          aria-label={ariaLabel}
          style={{
            left: position.left,
            width: position.width,
            ...(position.above
              ? { bottom: window.innerHeight - position.top }
              : { top: position.top }),
          }}
          className="fixed z-[100] max-h-64 overflow-y-auto rounded-md border border-hair bg-raised p-1 shadow-2xl"
        >
          {options.map((option, index) => (
            <button
              key={`${option.value}:${index}`}
              type="button"
              role="option"
              aria-selected={index === selectedIndex}
              disabled={option.disabled}
              onMouseEnter={() => !option.disabled && setActive(index)}
              onClick={() => choose(index)}
              className={`flex w-full items-start justify-between gap-3 rounded px-2.5 py-2 text-left font-mono text-xs leading-relaxed transition disabled:opacity-40 ${
                index === active ? "bg-brass-500/15 text-brass-200" : "text-slate-300 hover:bg-canvas/70"
              }`}
            >
              <span className="min-w-0">{option.label}</span>
              {index === selectedIndex && <Check size={13} className="mt-0.5 shrink-0 text-brass-300" />}
            </button>
          ))}
        </div>,
        document.body,
      )}
    </div>
  );
}
