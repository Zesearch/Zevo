import { Component, type ErrorInfo, type ReactNode } from "react";

/**
 * Catches render errors in the routed page so one bad component shows a
 * fallback (with the actual error) instead of blanking the whole app to a
 * black screen. `resetKey` (e.g. the current pathname) clears the error when
 * it changes, so navigating away recovers without a manual reload.
 */
type Props = { children: ReactNode; resetKey?: string };
type State = { error: Error | null };

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Surface in the console for debugging (and so it's visible in container logs
    // if the user copies it).
    console.error("[ErrorBoundary] page crashed:", error, info.componentStack);
  }

  componentDidUpdate(prev: Props) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div className="p-8">
        <div className="max-w-2xl rounded-xl border border-rose-500/30 bg-rose-500/5 p-5">
          <h1 className="text-lg font-semibold text-rose-200">This page hit an error</h1>
          <p className="mt-1 text-sm text-slate-400">
            The rest of the app is fine. Navigate elsewhere, or reload after the run advances.
          </p>
          <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md bg-slate-950/60 p-3 font-mono text-[15px] text-rose-300">
            {error.message}
            {error.stack ? "\n\n" + error.stack : ""}
          </pre>
          <button
            onClick={() => this.setState({ error: null })}
            className="mt-3 rounded-md border border-slate-700 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-800"
          >
            Try again
          </button>
        </div>
      </div>
    );
  }
}
