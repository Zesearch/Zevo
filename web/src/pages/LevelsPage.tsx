import { useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { ChevronLeft } from "lucide-react";
import { Bezel, PageHead } from "../components/zevo/primitives";

/**
 * Autonomy levels — what L1..L4 mean.
 *
 * The Tasks list wears a level chip on every row but has no room to say what a
 * level IS, and the definition is the same for every task, so it lives here
 * once rather than being restated eighteen times. Chips on that page link in,
 * anchored at the level they carry.
 */

const DIMS = ["Training data", "Training model", "Training method"] as const;

const OWNER_TONE = {
  user: "border-skyx-500/30 bg-skyx-500/10 text-skyx-300",
  zevo: "border-brass-500/30 bg-brass-500/10 text-brass-300",
};

// Same blue→yellow ramp the Tasks chips use, so a level keeps its colour across
// the two pages. Spelled out because Tailwind can't see interpolated names.
const LEVELS = [
  {
    id: "L1",
    name: "Entry autonomy",
    chip: "border-[#5FB0DD66] bg-[#5FB0DD1f] text-[#8FCDEB]",
    owners: [true, true, true],
    blurb: "Optimize inside the user-pinned data, model and method",
  },
  {
    id: "L2",
    name: "Constraint autonomy",
    chip: "border-[#55CE9166] bg-[#55CE911f] text-[#7DE8CE]",
    owners: [false, true, true],
    blurb: "Exhaust the current Data branch before selecting the next Data",
  },
  {
    id: "L3",
    name: "Partial autonomy",
    chip: "border-[#9BD06466] bg-[#9BD0641f] text-[#BFE08F]",
    owners: [false, true, false],
    blurb: "Search Data inside the current Method; exhaust it before changing Method",
  },
  {
    id: "L4",
    name: "Full autonomy",
    chip: "border-[#E6BB5B66] bg-[#E6BB5B1f] text-[#E6BB5B]",
    owners: [false, false, false],
    blurb: "Search Data, then Method, then Base model as each level is exhausted",
  },
] as const;

export function LevelsPage() {
  const { hash } = useLocation();

  // Arriving from a level chip should land on that level, not at the top.
  useEffect(() => {
    if (!hash) return;
    document.getElementById(hash.slice(1))?.scrollIntoView({ block: "center" });
  }, [hash]);

  return (
    <div className="w-full px-[max(1.5rem,1.5vw)] py-8">
      <Link to="/tasks" className="inline-flex items-center gap-1.5 text-sm text-slate-400 transition hover:text-brass-300">
        <ChevronLeft size={15} /> Tasks
      </Link>

      <div className="mt-2">
        <PageHead
          title="Autonomy Level"
          subtitle="Four levels of control between the user and Zevo"
        />
      </div>

      {/* The whole ladder as one table: four rows, three columns of who-decides.
          It was four stacked cards, which made you hold L1 in your head to see
          what L2 changed. Side by side, the column that flips is the answer. */}
      <Bezel className="overflow-x-auto">
        {/* The rule the table encodes, stated once before the table states it
          per level. It used to be the page subtitle, which made that one line
          three times longer than every other page's. */}
      <p className="mb-4 px-4 pt-1 text-sm leading-relaxed text-slate-400">
        Every task rests on three decisions: the training data, the model, and the method.
        Each decision is either given by the user or made by Zevo. The more Zevo decides,
        the higher the autonomy level. Zevo keeps the current branch until its inner search
        is exhausted, then advances one level at a time.
      </p>

      {/* Fixed widths on the first four columns so the meaning column absorbs the
            slack. Left to auto layout the three one-word columns each took a
            quarter of a wide screen and the table was mostly air. */}
        <table className="w-full min-w-[880px] table-fixed text-sm">
          <thead>
            <tr className="border-b border-hair text-left">
              {[["Level", "w-[20rem]", ""], ...DIMS.map((d) => [d, "w-[11rem]", "text-center"] as const), ["What Zevo decides", "w-[30rem]", ""]].map(([h, w, align]) => (
                <th key={h} className={`table-label px-4 py-3 ${w} ${align}`}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-hair">
            {LEVELS.map((lv) => (
              <tr key={lv.id} id={lv.id} className="scroll-mt-24 target:bg-raised">
                <td className="whitespace-nowrap px-4 py-4 align-top">
                  <span className="flex items-center gap-4">
                    <span className={`rounded border px-2.5 py-1 font-mono text-sm ${lv.chip}`}>{lv.id}</span>
                    <span className="font-mono text-sm text-slate-300">{lv.name}</span>
                  </span>
                </td>
                {lv.owners.map((byUser, i) => (
                  <td key={i} className="px-4 py-4 text-center align-top">
                    <span className={`rounded border px-2.5 py-1 font-mono text-sm ${byUser ? OWNER_TONE.user : OWNER_TONE.zevo}`}>
                      {byUser ? "User" : "Zevo"}
                    </span>
                  </td>
                ))}
                <td className="px-4 py-4 align-top text-sm text-slate-400">{lv.blurb}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Bezel>

    </div>
  );
}
