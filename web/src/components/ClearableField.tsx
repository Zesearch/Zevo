import { createContext, forwardRef, useContext, useRef, type InputHTMLAttributes, type TextareaHTMLAttributes, type ForwardedRef } from "react";
import { X } from "lucide-react";

export const ClearFieldsContext = createContext(false);
function assignRef<T>(ref: ForwardedRef<T>, value: T | null) {
  if (typeof ref === "function") ref(value);
  else if (ref) ref.current = value;
}
function clearField(field: HTMLInputElement | HTMLTextAreaElement | null) {
  if (!field) return;
  const prototype = field instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  // Use the native setter so React receives a normal input change.
  Object.getOwnPropertyDescriptor(prototype, "value")?.set?.call(field, "");
  field.dispatchEvent(new Event("input", { bubbles: true }));
  field.focus();
}
function fieldName(field: HTMLElement | null, fallback?: string) {
  return field?.getAttribute("aria-label") || field?.closest("label")?.textContent?.trim()
    || field?.parentElement?.parentElement?.querySelector(".field-label")?.textContent?.trim() || fallback || "field";
}
export const ClearableInput = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function ClearableInput(props, forwardedRef) {
  const enabled = useContext(ClearFieldsContext);
  const ref = useRef<HTMLInputElement | null>(null);
  const clearable = enabled && !props.disabled && !props.readOnly && !!props.onChange
    && !["checkbox", "radio", "file", "hidden", "range", "button", "submit", "color"].includes(props.type || "text");
  const filled = props.value !== undefined && props.value !== null && String(props.value).length > 0;
  const input = <input {...props} ref={value => { ref.current = value; assignRef(forwardedRef, value); }} className={`${props.className || ""}${clearable && filled ? " !pr-11" : ""}`} />;
  if (!clearable) return input;
  return <span className="clearable-field relative block w-full min-w-0 flex-1">
    {input}
    {filled && <button type="button" className="field-clear" aria-label={`Clear ${fieldName(ref.current, props.placeholder)}`} title="Clear this field" onMouseDown={event => event.preventDefault()} onClick={() => clearField(ref.current)}><X size={14} /></button>}
  </span>;
});
export const ClearableTextarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(function ClearableTextarea(props, forwardedRef) {
  const enabled = useContext(ClearFieldsContext);
  const ref = useRef<HTMLTextAreaElement | null>(null);
  const clearable = enabled && !props.disabled && !props.readOnly && !!props.onChange;
  const filled = props.value !== undefined && props.value !== null && String(props.value).length > 0;
  const input = <textarea {...props} ref={value => { ref.current = value; assignRef(forwardedRef, value); }} className={`${props.className || ""}${clearable && filled ? " !pr-11" : ""}`} />;
  if (!clearable) return input;
  return <span className="clearable-field relative block w-full min-w-0">
    {input}
    {filled && <button type="button" className="field-clear field-clear-multiline" aria-label={`Clear ${fieldName(ref.current, props.placeholder)}`} title="Clear this field" onMouseDown={event => event.preventDefault()} onClick={() => clearField(ref.current)}><X size={14} /></button>}
  </span>;
});
