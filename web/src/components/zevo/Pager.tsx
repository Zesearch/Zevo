import { ChevronLeft, ChevronRight } from "lucide-react";

/**
 * The footer every card list pages with: what you are looking at, how many fit
 * on a page, and the way to the next one.
 *
 * One component because three lists had three near-copies of it, and a pager
 * that behaves differently on Tasks than on Files is a pager you have to read
 * before you can use.
 *
 * `0` in `sizes` means "as many as fit" — the caller measures that, since only
 * it knows how tall its own cards are.
 */
export const PAGE_SIZES = [0, 25, 50, 100];

export function Pager({
  total, first, last, page, pageCount, pageSizePref, onSize, onPage,
  sizes = PAGE_SIZES,
}: {
  total: number;
  /** 1-based index of the first row shown; 0 when there are none. */
  first: number;
  last: number;
  /** 0-based, and already clamped by the caller. */
  page: number;
  pageCount: number;
  /** The chosen size, `0` for auto — not the resolved one. */
  pageSizePref: number;
  onSize: (n: number) => void;
  onPage: (n: number) => void;
  sizes?: number[];
}) {
  if (total === 0) return null;
  return (
    <div className="flex flex-wrap items-center justify-between gap-4 px-1 pt-6">
      <div className="flex items-center gap-2 font-mono text-2xs text-slate-500">
        <span>{first}–{last} of {total}</span>
        <span className="text-slate-700">·</span>
        <span className="text-slate-600">per page</span>
        {sizes.map((n) => (
          <button
            key={n}
            onClick={() => onSize(n)}
            className={`rounded px-1.5 py-0.5 transition ${
              n === pageSizePref
                ? "bg-brass-500/15 text-brass-300"
                : "text-slate-500 hover:bg-raised hover:text-slate-300"
            }`}
          >
            {n || "auto"}
          </button>
        ))}
      </div>

      {pageCount > 1 && (
        <div className="flex items-center gap-2">
          <button
            onClick={() => onPage(page - 1)}
            disabled={page === 0}
            className="btn px-2 disabled:opacity-30"
            title="Previous page"
          >
            <ChevronLeft size={14} />
          </button>
          <span className="font-mono text-2xs text-slate-500">
            page {page + 1} / {pageCount}
          </span>
          <button
            onClick={() => onPage(page + 1)}
            disabled={page >= pageCount - 1}
            className="btn px-2 disabled:opacity-30"
            title="Next page"
          >
            <ChevronRight size={14} />
          </button>
        </div>
      )}
    </div>
  );
}
