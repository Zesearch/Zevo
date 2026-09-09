import { useEffect, useMemo, useRef, useState } from "react";
import { TranscriptEventRow, type TranscriptEvent } from "./TranscriptEventRow";
import { fmtCost, fmtTokens } from "../lib/format";
import { api } from "../lib/api";
import type { HeartbeatDTO } from "../lib/api";

export type EventEnvelope = TranscriptEvent & { seq: number };


function buildWsUrl(path: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${path}`;
}

export function LiveTranscript({
  heartbeatId,
  className = "",
  height = "max-h-[36rem]",
  onEvents,
  seekTs = "",
  seekNonce = 0,
}: {
  heartbeatId: string;
  className?: string;
  /** Called with the cleaned, ordered feed whenever it changes. */
  onEvents?: (events: EventEnvelope[]) => void;
  height?: string;
  /** Scroll to the first event at or after this timestamp and flash it. The
   *  phase chips above the transcript set it, so a phase reads as a section
   *  heading for the stretch of activity that follows it. */
  seekTs?: string;
  /** Bumped on every chip click so clicking the SAME chip twice jumps twice.
   *  Without it the seek is keyed only by timestamp and a repeat click, having
   *  the same value, is indistinguishable from no click at all. */
  seekNonce?: number;
}) {
  const [events, setEvents] = useState<EventEnvelope[]>([]);
  const [meta, setMeta] = useState<HeartbeatDTO | null>(null);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  // The page above renders its own summary of this same feed. Hand the events
  // up rather than letting it open a second socket for the same heartbeat.
  const onEventsRef = useRef(onEvents);
  onEventsRef.current = onEvents;
  // Start at the TOP (read from the beginning); only stick to the bottom once the
  // user scrolls down there. So false by default, not true.
  const wasAtBottomRef = useRef<boolean>(false);

  // Reset all per-heartbeat state when the selected heartbeat changes — otherwise
  // clicking a different agent would APPEND its events onto the previous one's, so
  // the transcript would show every heartbeat you've clicked (and seq, which is
  // per-heartbeat, would collide across them). Each heartbeat starts fresh.
  useEffect(() => {
    setEvents([]);
    setMeta(null);
    setError(null);
    wasAtBottomRef.current = false;
  }, [heartbeatId]);

  // Fetch metadata for the header (live status, driver/model). The refresh
  // stops itself once the heartbeat is finished — the row is immutable from
  // then on, and so is the transcript, so nothing here should keep polling.
  const liveRef = useRef<boolean | null>(null);
  useEffect(() => {
    if (!heartbeatId) return;
    let stopped = false;
    liveRef.current = null;
    const interval = setInterval(() => void once(), 5000);
    async function once() {
      try {
        const j = await api<HeartbeatDTO>(`/heartbeats/${encodeURIComponent(heartbeatId)}`);
        if (stopped) return;
        setMeta(j);
        liveRef.current = j.is_live;
        if (!j.is_live) clearInterval(interval);
      } catch (e) {
        if (!stopped) setError(String(e));
      }
    }
    void once();
    return () => {
      stopped = true;
      clearInterval(interval);
    };
  }, [heartbeatId]);

  // Open WebSocket. Falls back to polling /events if WS fails.
  useEffect(() => {
    if (!heartbeatId) return;
    let cancelled = false;
    let ws: WebSocket | null = null;
    let pollTimer: ReturnType<typeof setInterval> | null = null;
    let lastSeq = 0;

    const startPollingFallback = () => {
      if (pollTimer) return;
      pollTimer = setInterval(async () => {
        try {
          const rows = await api<EventEnvelope[]>(
            `/heartbeats/${encodeURIComponent(heartbeatId)}/events?since_seq=${lastSeq}`,
          );
          if (cancelled) return;
          if (rows.length) {
            setEvents((prev) => [...prev, ...rows]);
            lastSeq = rows[rows.length - 1].seq;
          } else if (liveRef.current === false && pollTimer) {
            // Finished heartbeat, nothing new arriving: the transcript is
            // complete, so stop hitting the API every 1.5s forever.
            clearInterval(pollTimer);
            pollTimer = null;
          }
        } catch {
          /* keep trying */
        }
      }, 1500);
    };

    try {
      ws = new WebSocket(buildWsUrl(`/api/ws/heartbeats/${encodeURIComponent(heartbeatId)}`));
      ws.onmessage = (msg) => {
        if (cancelled) return;
        try {
          const ev: EventEnvelope = JSON.parse(msg.data);
          setEvents((prev) => [...prev, ev]);
          lastSeq = Math.max(lastSeq, ev.seq);
        } catch {
          /* ignore */
        }
      };
      ws.onerror = () => {
        if (cancelled) return;
        startPollingFallback();
      };
      ws.onclose = () => {
        if (cancelled) return;
        // If heartbeat is still live (no finished event), fall back to polling.
        startPollingFallback();
      };
    } catch (e) {
      setError(String(e));
      startPollingFallback();
    }

    return () => {
      cancelled = true;
      if (ws) try { ws.close(); } catch { /* */ }
      if (pollTimer) clearInterval(pollTimer);
    };
  }, [heartbeatId]);

  // Sticky-tail auto-scroll.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && wasAtBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [events.length]);

  // Jump to the event a phase chip points at. Phases and transcript rows come
  // from different tables, so there is no id to match on — the timestamp is
  // the join. Seeking also cancels sticky-tail, or the next event would yank
  // the reader back to the bottom of a live feed.
  const HL = ["ring-1", "ring-brass-500/60", "bg-brass-500/5"];
  const litRef = useRef<HTMLElement | null>(null);
  const seekDoneRef = useRef("");
  const wantTs = seekTs;
  const wantN = seekNonce;
  useEffect(() => {
    if (!wantTs) return;
    const token = `${wantTs}#${wantN}`;
    // `events.length` is in the deps so a seek fired before the rows exist can
    // still land once they arrive. Without this guard it would ALSO re-scroll
    // on every incoming event of a live feed, dragging the reader back to the
    // target each time something new arrived.
    if (seekDoneRef.current === token) return;
    const el = scrollRef.current;
    if (!el) return;
    const row = el.querySelector<HTMLElement>(`[data-ts="${CSS.escape(wantTs)}"]`)
      ?? [...el.querySelectorAll<HTMLElement>("[data-ts]")].find(
        (n) => (n.dataset.ts ?? "") >= wantTs);
    if (!row) return;
    seekDoneRef.current = token;
    wasAtBottomRef.current = false;
    // NEVER measure a sticky element's own position. While a section heading is
    // pinned to the top of the scrollport its `offsetTop` reports the PINNED
    // position, not where it sits in the document — so scrolling back to an
    // earlier heading computed "roughly where we already are" and the view
    // crept up a few pixels instead of jumping. Jumping forward looked fine
    // only because a heading below the fold is not pinned yet.
    //
    // Take the static position from the last ordinary row above it instead;
    // event rows never stick, so their geometry is stable. Walk past any
    // adjacent headings (empty sections) to reach one.
    let top: number;
    if (row.classList.contains("sticky")) {
      let prev = row.previousElementSibling as HTMLElement | null;
      while (prev && prev.classList.contains("sticky")) {
        prev = prev.previousElementSibling as HTMLElement | null;
      }
      top = prev ? prev.offsetTop - el.offsetTop + prev.offsetHeight : 0;
    } else {
      top = row.offsetTop - el.offsetTop;
    }
    el.scrollTo({ top: Math.max(0, top - 8), behavior: "smooth" });
    // Put out the previous highlight before lighting this one. The cleanup
    // used to cancel only the fade TIMER, so a second click inside 1.6s left
    // the first row lit for good and the transcript accumulated highlights.
    litRef.current?.classList.remove(...HL);
    row.classList.add(...HL);
    litRef.current = row;
    const id = setTimeout(() => {
      row.classList.remove(...HL);
      if (litRef.current === row) litRef.current = null;
    }, 1600);
    return () => clearTimeout(id);
  }, [wantTs, wantN, events.length]);

  // A different activation replaces every row, so drop the stale reference.
  useEffect(() => {
    litRef.current = null;
    seekDoneRef.current = "";
  }, [heartbeatId]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    wasAtBottomRef.current =
      Math.abs(el.scrollHeight - el.scrollTop - el.clientHeight) < 16;
  };

  // Dedup by seq (WS + REST fallback can both deliver the same event).
  const sorted = useMemo(() => {
    const seen = new Set<number>();
    const out: EventEnvelope[] = [];
    for (const e of events) {
      if (seen.has(e.seq)) continue;
      // __PROGRESS__/__PHASE__/__CONFIG__ marker rows are redundant noise here
      // (train/infer surface them in the monitor chart + progress bar), so we
      // drop them from the transcript.
      if (e.type === "progress" || e.type === "phase" || e.type === "config") continue;
      seen.add(e.seq);
      out.push(e);
    }
    out.sort((a, b) => a.seq - b.seq);
    return out;
  }, [events]);

  useEffect(() => {
    onEventsRef.current?.(sorted);
  }, [sorted]);


  if (!heartbeatId) {
    return <div className="text-xs text-dim">No heartbeat selected.</div>;
  }

  return (
    <div className={`overflow-hidden rounded-bezel border border-hair bg-canvas ${className}`}>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-hair bg-panel/60 px-3 py-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="kicker">Telemetry · {sorted.length}</span>
          {meta && (
            <>
              <code className="font-mono text-[14px] text-dim">
                {meta.id.slice(0, 8)}
              </code>
              <span className="text-slate-600">·</span>
              <code className="font-mono text-[14px] text-dim">
                {meta.driver}/{meta.model}
              </code>
              {meta.is_live ? (
                <span className="inline-flex items-center gap-1.5 rounded-full border border-phosphor-500/30 bg-phosphor-500/10 px-2 py-0.5 font-mono text-[13px] uppercase tracking-wider text-phosphor-200">
                  <span className="lamp lamp-live" />
                  live
                </span>
              ) : (
                <span
                  className={`rounded px-1.5 py-0.5 font-mono text-[13px] ${
                    meta.exit_code === 0
                      ? "bg-phosphor-500/15 text-phosphor-300"
                      : "bg-coral-500/15 text-coral-300"
                  }`}
                >
                  exit {meta.exit_code}
                </span>
              )}
              {(meta.input_tokens + meta.output_tokens) > 0 && (
                <>
                  <span className="text-slate-600">·</span>
                  <code className="font-mono text-[14px] text-dim">
                    {fmtTokens(meta.input_tokens + meta.output_tokens)} tok
                  </code>
                  {meta.estimated_cost_usd > 0 && (
                    <code className="font-mono text-[14px] text-brass-300/80">
                      {fmtCost(meta.estimated_cost_usd)}
                    </code>
                  )}
                </>
              )}
            </>
          )}
        </div>
        {/* No transport indicator: whether events arrive over the WebSocket or
            the REST fallback is ours to handle, not something to report. It
            also said nothing new — the server holds the socket open exactly
            while the heartbeat runs, so it only ever restated `live`/`exit N`
            beside it. */}
      </div>
      {error && (
        <div className="border-b border-coral-500/20 bg-coral-500/10 px-3 py-2 text-xs text-coral-300">
          {error}
        </div>
      )}
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className={`divide-y divide-hair/60 overflow-auto ${height}`}
      >
        {sorted.length === 0 ? (
          <div className="px-3 py-6 text-center font-mono text-xs text-dim">
            {meta?.is_live ? "(waiting for first event)" : "(no events captured)"}
          </div>
        ) : (
          sorted.map((ev) => (
            <div key={`${ev.seq}-${ev.type}`} data-ts={ev.ts} className="transition-colors">
              <TranscriptEventRow ev={ev} />
            </div>
          ))
        )}
      </div>
    </div>
  );
}
