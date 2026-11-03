export function FieldHint({ children, detail }: { children: React.ReactNode; detail?: string }) {
  return <span title={detail} className="ml-2 text-xs font-normal normal-case tracking-normal text-slate-500">· {children}</span>;
}
