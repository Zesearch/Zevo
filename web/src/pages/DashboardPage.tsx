import { Rocket } from "lucide-react";
import { fireCommand } from "../lib/commands";
import { PageHead } from "../components/zevo/primitives";
import { VitalsGrid } from "../components/zevo/VitalsGrid";
import { ConsoleOverview } from "../components/zevo/ConsoleOverview";

// The console answers two questions and nothing else: where the system stands
// right now (the vitals strip), and whether the loop is actually producing
// better models (the overview below it). Listing runs, tickets, agents and
// models is the job of their own pages — this one measures rather than
// enumerates.
export function DashboardPage() {
  return (
    <div className="w-full px-[max(1.5rem,1.5vw)] py-8">
      <PageHead
        title="Dashboard"
        subtitle="Overview of runs, costs, and model improvements achieved by Zevo"
        right={
          <>
            <button onClick={() => fireCommand("open-new-run")} className="btn btn-brass">
              <Rocket size={14} /> Launch run
            </button>
          </>
        }
      />

      {/* ── Where the system stands ── */}
      <VitalsGrid />

      {/* ── Is the loop actually working, and what is winning? ── */}
      <div className="mt-5">
        <ConsoleOverview />
      </div>
    </div>
  );
}
