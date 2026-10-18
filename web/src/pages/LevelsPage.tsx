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

const DIMS = ["Training data", "Base model", "Training method"] as const;

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
    blurb: "Decide on training data; the user specifies model and method",
  },
  {
    id: "L3",
    name: "Partial autonomy",
    chip: "border-[#9BD06466] bg-[#9BD0641f] text-[#BFE08F]",
    owners: [false, true, false],
    blurb: "Decide on training data and method; the user specifies the model",
  },
  {
    id: "L4",
    name: "Full autonomy",
    chip: "border-[#E6BB5B66] bg-[#E6BB5B1f] text-[#E6BB5B]",
    owners: [false, false, false],
    blurb: "Decide on training data, model and method",
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

      <p className="mb-5 text-sm leading-relaxed text-slate-400">
        Autonomy depends on how many of the three decisions Zevo makes: training data,
        base model, and training method. Zevo can decide any combination; the remaining
        decisions stay fixed as specified by the user.
      </p>
      <div className="mb-8 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {LEVELS.map((lv, index) => (
          <div key={lv.id} id={lv.id} className="scroll-mt-24 rounded-lg border border-hair bg-raised p-4">
            <span className={`rounded border px-2.5 py-1 font-mono text-sm ${lv.chip}`}>{lv.id}</span>
            <p className="mt-3 text-sm text-slate-200">
              {["Zevo decides none", "Zevo decides any 1", "Zevo decides any 2", "Zevo decides all 3"][index]}
            </p>
          </div>
        ))}
      </div>
      <h2 className="mb-2 text-lg font-semibold text-slate-200">Common examples</h2>
      <p className="mb-4 text-sm text-slate-400">
        These are common examples of what Zevo decides. Any combination with the same number
        of decisions belongs to the same level. For example, L2 also includes Zevo deciding
        only the method; L3 also includes Zevo deciding data and model.
      </p>
      <Bezel className="overflow-x-auto">
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
              <tr key={lv.id}>
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
