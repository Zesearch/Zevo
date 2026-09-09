import { useEffect, useState } from "react";

export type RunWsMessage = {
  type: "snapshot";
  tickets: Record<
    string,
    {
      agent_id: string;
      lane: "optimization" | "held_out_test";
      iteration: number;
      status: string;
      summary: string;
      created_at: string;
      execution_event_rows: number;
    }
  >;
};

export function useRunWebsocket(runId: string | undefined): RunWsMessage | null {
  const [msg, setMsg] = useState<RunWsMessage | null>(null);

  useEffect(() => {
    if (!runId) return;
    const url =
      (window.location.protocol === "https:" ? "wss://" : "ws://") +
      window.location.host +
      `/api/ws/runs/${runId}`;
    const sock = new WebSocket(url);
    sock.onmessage = (ev) => {
      try {
        setMsg(JSON.parse(ev.data));
      } catch {
        /* ignore */
      }
    };
    return () => sock.close();
  }, [runId]);

  return msg;
}
