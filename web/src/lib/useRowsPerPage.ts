import { useEffect, useState } from "react";

/**
 * How many rows fit on screen below `el`, so a page fills the window and its
 * pager sits at the bottom of it.
 *
 * A fixed page size is wrong on both ends of the same list: 25 rows leaves a
 * laptop scrolling and a tall monitor half empty, and either way the pager ends
 * up somewhere arbitrary. Measuring instead means the page is as long as the
 * window is, and paging is what you do when you reach the bottom of it.
 *
 * Takes the ELEMENT, not a ref object. A list is not in the DOM on the first
 * render — it is behind a loading branch — and a ref object never changes
 * identity, so an effect keyed on one runs once against `null`, measures
 * nothing, and the page size stays stuck at the minimum forever. An element
 * held in state changes when it mounts, which is exactly when there is
 * something to measure.
 *
 * `rowHeight` is the full stride of one row including whatever separates it
 * from the next, a measured value from the rendered list rather than the CSS
 * height.
 *
 * What sits BELOW the list — the pager, the page's bottom padding, the site
 * footer — is measured rather than declared. It was a per-page constant, and a
 * constant that is too small produces exactly the bug this hook exists to
 * prevent: one row too many, and the pager lands under the fold. It also has to
 * be re-tuned every time anything is added under a list. The measurement below
 * is independent of how many rows are currently shown, so it converges instead
 * of oscillating.
 */
export function useRowsPerPage(
  el: HTMLElement | null,
  rowHeight: number,
  { min = 3, max = 100 }: { min?: number; max?: number } = {},
): number {
  const [rows, setRows] = useState(min);

  useEffect(() => {
    if (!el) return;

    /** Height of everything rendered after `node`, up to `stop`.
     *
     *  Summed element by element rather than taken as
     *  `scrollHeight - listBottom`: that difference also contains the slack the
     *  pager's `mt-auto` absorbs and, on a short page, the empty space under
     *  the content. Both of those grow as the list shrinks, so feeding them
     *  back in drove the row count down to its floor and kept it there. An
     *  element's own height does not depend on how many rows are above it.
     */
    function heightBelow(node: HTMLElement, stop: HTMLElement): number {
      let total = 0;
      let cur: HTMLElement | null = node;
      while (cur && cur !== stop) {
        for (let sib = cur.nextElementSibling; sib; sib = sib.nextElementSibling) {
          const box = sib as HTMLElement;
          const cs = getComputedStyle(box);
          // Bottom margin counts, top margin does not. The pager is held at the
          // foot of the page by `margin-top: auto`, and getComputedStyle reports
          // an auto margin as the pixels it actually resolved to -- which IS the
          // slack this function is measuring how much of. Feeding it back in
          // made fewer rows produce more slack produce fewer rows, all the way
          // down to the floor. Anything wanting a fixed gap above the pager
          // should use padding, which lives inside the rect.
          total += box.getBoundingClientRect().height + (parseFloat(cs.marginBottom) || 0);
        }
        const parent: HTMLElement | null = cur.parentElement;
        if (parent && parent !== stop) {
          total += parseFloat(getComputedStyle(parent).paddingBottom) || 0;
        }
        cur = parent;
      }
      return total;
    }

    function measure() {
      if (!el) return;
      // Measured against the window, not against a scrolling ancestor. Finding
      // "the scroller" by walking up for overflow-y kept landing on the wrapper
      // that gives a wide table a horizontal scrollbar: `overflow-x: auto`
      // computes `overflow-y: auto` too, so that box looked like the viewport
      // and its height was the table's own — which pinned every list to the
      // minimum row count.
      // The 8px is slack, not a fudge for a wrong measurement: the sum above is
      // of border boxes, so hairline borders on the wrappers between the list
      // and the page edge are not in it, and a row's stride is fractional
      // (51.4px, not 52). Erring by a few pixels in this direction costs a row
      // only when the fit was that close; erring the other way puts the pager
      // under the fold, which is the whole thing this exists to avoid.
      const available =
        window.innerHeight - el.getBoundingClientRect().top - heightBelow(el, document.body) - 8;
      setRows(Math.max(min, Math.min(max, Math.floor(available / rowHeight))));
    }

    measure();
    // And again as the page settles. The first pass runs before the layout
    // around the list is final: switching to a tab whose data is still loading,
    // or a first render where the pager is not on the page yet, both leave the
    // list measuring against a page shorter than the one it ends up in.
    // ResizeObserver is meant to catch that and mostly does, but not reliably
    // for a <tbody>, so these two re-measures are the belt to its braces.
    const raf = requestAnimationFrame(measure);
    const settle = [
      window.setTimeout(measure, 150),
      window.setTimeout(measure, 600),
    ];
    window.addEventListener("resize", measure);
    // The list's top moves when anything above it reflows: a search box
    // wrapping to two lines, a group header appearing. The PARENT is observed
    // rather than the list itself because a list can be a <tbody>, and table
    // sections do not reliably report their own resizes.
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(measure) : null;
    ro?.observe(el);
    if (el.parentElement) ro?.observe(el.parentElement);
    // The page container inside the scroller. `document.body` never changes
    // height here -- the page scrolls inside <main>, not the document -- so
    // watching it alone meant nothing fired when a tab's content arrived and
    // the list stayed at whatever it measured before the data landed.
    const page = el.closest("main")?.firstElementChild;
    if (page) ro?.observe(page);
    ro?.observe(document.body);
    return () => {
      cancelAnimationFrame(raf);
      settle.forEach(window.clearTimeout);
      window.removeEventListener("resize", measure);
      ro?.disconnect();
    };
  }, [el, rowHeight, min, max]);

  return rows;
}
