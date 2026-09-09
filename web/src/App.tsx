import { useEffect, useRef, useState } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { ConsoleBar } from "./components/zevo/ConsoleBar";
import { InstrumentRail } from "./components/zevo/InstrumentRail";
import { CommandPalette } from "./components/zevo/CommandPalette";
import { ToastHost } from "./components/zevo/ToastHost";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { NewRunModal, type RunLaunchMode } from "./components/NewRunModal";
import { onCommand, type RunLaunchPreset } from "./lib/commands";
import { DashboardPage } from "./pages/DashboardPage";
import { RunsPage } from "./pages/RunsPage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { AgentsPage } from "./pages/AgentsPage";
import { AgentDetailPage } from "./pages/AgentDetailPage";
import { TicketDetailPage } from "./pages/TicketDetailPage";
import { ModelsPage } from "./pages/ModelsPage";
import { FilesPage } from "./pages/FilesPage";
import { HardwarePage } from "./pages/HardwarePage";
import { SettingsPage } from "./pages/SettingsPage";
import { LeaderboardPage } from "./pages/LeaderboardPage";
import { TasksPage } from "./pages/TasksPage";
import { LevelsPage } from "./pages/LevelsPage";

// Stamped once at module load. A footer that re-derives the year on every
// render is a clock nobody asked for.
const BUILD_YEAR = new Date().getFullYear();

export default function App() {
  const location = useLocation();
  const mainRef = useRef<HTMLElement>(null);
  useEffect(() => {
    mainRef.current?.scrollTo({ top: 0 });
  }, [location.pathname]);

  // Global launch modals, driven from the command palette / console bar so a run
  // can be launched from any page.
  const [newRun, setNewRun] = useState(false);
  // Which mode the dialog opens on — an agent page wants the single-agent form.
  // Auto is the default: a plain "new run" derives the scoring contract
  // form. A preset carries a predefined task and its saved setting, which
  // Auto cannot take (it always starts a fresh custom task), so that opens
  // Full Pipeline where the preset's fields live.
  const [runMode, setRunMode] = useState<RunLaunchMode>("auto");
  const [runAgent, setRunAgent] = useState<string | undefined>();
  const [runPreset, setRunPreset] = useState<RunLaunchPreset | undefined>();
  useEffect(
    () =>
      onCommand((c, arg, preset) => {
        if (c === "open-new-run") {
          setRunMode(preset ? "full_pipeline" : "auto"); setRunAgent(undefined); setRunPreset(preset); setNewRun(true);
        }
        if (c === "open-new-run-single") {
          setRunMode("single_stage"); setRunAgent(arg); setRunPreset(undefined); setNewRun(true);
        }
      }),
    [],
  );

  return (
    <div className="scanlines flex h-screen w-full flex-col">
      <ConsoleBar />
      <div className="flex min-h-0 flex-1">
        <InstrumentRail />
        <main ref={mainRef} className="flex flex-1 flex-col overflow-auto">
          <div className="flex flex-1 flex-col">
          <ErrorBoundary resetKey={location.pathname}>
            <Routes>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/runs" element={<RunsPage />} />
              <Route path="/runs/:runId" element={<RunDetailPage />} />
              {/* The Tickets list is out of the UI for now — an agent's
                  "Tickets & heartbeats" tab covers the same ground with the
                  transcripts attached. Single tickets still open by id, and a
                  saved /tickets link lands on the dashboard via the catch-all. */}
              <Route path="/tickets/:ticketId" element={<TicketDetailPage />} />
              <Route path="/agents" element={<AgentsPage />} />
              <Route path="/agents/:agentId" element={<AgentDetailPage />} />
              <Route path="/models" element={<ModelsPage />} />
              <Route path="/files" element={<FilesPage />} />
              <Route path="/hardware" element={<HardwarePage />} />
              <Route path="/tasks" element={<TasksPage />} />
              <Route path="/levels" element={<LevelsPage />} />
              <Route path="/leaderboard" element={<LeaderboardPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              {/* One path per page, no aliases. The pages formerly at
                  /registry, /datasets and /secrets are gone rather than
                  redirected: the catch-all below lands an old bookmark on the
                  dashboard, and a redirect that outlives the rename is just a
                  second name for the page that nobody prunes. */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </ErrorBoundary>
          </div>
          {/* Inside the scroller, so it sits under the page's own content
              rather than pinning a bar to the window at every viewport. No rule
              above it: the pages below already end on their own boundaries, and
              a second line right under one read as a mistake. */}
          <footer className="px-[max(1.5rem,1.5vw)] pb-6 pt-4 text-2xs leading-relaxed text-slate-600">
            © {BUILD_YEAR} Zevo · Created by Zesearch NLP Group
          </footer>
        </main>
      </div>

      <CommandPalette />
      <ToastHost />
      {/* Mounted only while open, so every launch starts from a blank dialog.
          Keeping it mounted meant last time's objective, task name and inputs
          were still in state when you reopened, and the reset ran a frame
          later — you saw the previous run's form before it cleared. Modal
          renders null when closed, so there is no exit animation to lose. */}
      {newRun && (
        <NewRunModal
          open
          initialMode={runMode}
          initialAgent={runAgent}
          initialTaskName={runPreset?.taskName}
          initialSettingId={runPreset?.settingId}
          initialInputs={runPreset?.inputs}
          onClose={() => setNewRun(false)}
        />
      )}
    </div>
  );
}
