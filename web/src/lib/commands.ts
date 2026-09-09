// Tiny global command bus so the ⌘K palette and the console bar can trigger
// app-level actions (open the Launch-run modal) from anywhere,
// without prop-drilling through the shell.

export type ZevoCommand = "open-new-run" | "open-new-run-single" | "open-palette";

export type RunLaunchPreset = {
  taskName?: string;
  settingId?: string;
  inputs?: Record<string, string>;
};

const target = new EventTarget();

// `arg` carries the one bit of context a command needs — which agent the
// single-agent launch should open on. Anything richer belongs in the route.
export function fireCommand(cmd: ZevoCommand, arg?: string, preset?: RunLaunchPreset) {
  target.dispatchEvent(new CustomEvent("zevo-cmd", { detail: { cmd, arg, preset } }));
}

export function onCommand(
  handler: (cmd: ZevoCommand, arg?: string, preset?: RunLaunchPreset) => void,
): () => void {
  const listener = (e: Event) => {
    const d = (e as CustomEvent).detail as {
      cmd: ZevoCommand; arg?: string; preset?: RunLaunchPreset;
    };
    handler(d.cmd, d.arg, d.preset);
  };
  target.addEventListener("zevo-cmd", listener);
  return () => target.removeEventListener("zevo-cmd", listener);
}
