import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Tailwind-styled GitHub-flavoured markdown renderer, on the ZEVO
 * observatory palette (used in transcripts + summaries).
 *
 * The playbook agent documents lean heavily on tables, fenced code blocks,
 * headings, and inline code, so this component maps each element to a
 * dark-console style that matches the rest of the instrument. No
 * @tailwindcss/typography dependency — we style the elements directly
 * so the bundle stays lean.
 */
export function Markdown({ children }: { children: string }) {
  return (
    <div className="text-sm leading-relaxed text-slate-200">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="mb-3 mt-5 border-b border-hair pb-2 font-display text-xl font-semibold tracking-tight text-ink first:mt-0">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="mb-2 mt-5 border-b border-hair/60 pb-1 font-display text-lg font-semibold tracking-tight text-ink">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="mb-2 mt-4 font-display text-base font-semibold text-brass-200">{children}</h3>
          ),
          h4: ({ children }) => (
            <h4 className="mb-1 mt-3 font-mono text-2xs uppercase tracking-[0.18em] text-slate-500">
              {children}
            </h4>
          ),
          p: ({ children }) => <p className="my-2.5">{children}</p>,
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noreferrer"
               className="text-brass-300 underline decoration-brass-300/30 hover:decoration-brass-300">
              {children}
            </a>
          ),
          strong: ({ children }) => <strong className="font-semibold text-ink">{children}</strong>,
          em: ({ children }) => <em className="italic text-slate-100">{children}</em>,
          ul: ({ children }) => <ul className="my-2.5 ml-5 list-disc space-y-1 marker:text-slate-600">{children}</ul>,
          ol: ({ children }) => <ol className="my-2.5 ml-5 list-decimal space-y-1 marker:text-slate-600">{children}</ol>,
          li: ({ children }) => <li className="pl-1">{children}</li>,
          blockquote: ({ children }) => (
            <blockquote className="my-3 border-l-2 border-brass-500/40 bg-brass-500/5 py-1 pl-3 italic text-slate-400">
              {children}
            </blockquote>
          ),
          hr: () => <hr className="my-5 border-hair" />,
          code: ({ className, children, ...props }) => {
            const isBlock = !!className;  // fenced blocks carry language-* class
            if (isBlock) {
              return (
                <code className={`${className} font-mono text-[16px]`} {...props}>
                  {children}
                </code>
              );
            }
            return (
              <code className="rounded bg-raised px-1.5 py-0.5 font-mono text-[16px] text-brass-200" {...props}>
                {children}
              </code>
            );
          },
          pre: ({ children }) => (
            <pre className="bezel-flat my-3 overflow-x-auto p-3 font-mono text-[16px] leading-relaxed text-slate-300">
              {children}
            </pre>
          ),
          table: ({ children }) => (
            <div className="bezel-flat my-3 overflow-x-auto">
              <table className="w-full border-collapse text-xs">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-raised/60">{children}</thead>,
          th: ({ children }) => (
            <th className="table-label border-b border-hair px-3 py-2 text-left font-semibold">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border-b border-hair/60 px-3 py-2 align-top text-slate-300">{children}</td>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}

/**
 * The same markup, rendered INLINE — for a table cell or a definition row
 * where the surrounding grid owns the spacing and the type size.
 *
 * The agents write ordinary markdown into their prose fields, so a plain-text
 * render shows the source: `__config__` keeps its underscores, every
 * `identifier` keeps its backticks, and emphasis arrives as literal asterisks.
 * The block renderer above cannot be used instead — its wrapper sets its own
 * font size and its paragraphs carry margins, which breaks a row that is
 * supposed to sit on one baseline with its label.
 *
 * So paragraphs collapse to fragments and only the inline elements are styled.
 * Anything block-level (a list, a table) is left to the block renderer; this
 * one is for a sentence.
 */
export function InlineMarkdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => <>{children}</>,
        strong: ({ children }) => <strong className="font-semibold text-ink">{children}</strong>,
        em: ({ children }) => <em className="italic">{children}</em>,
        a: ({ href, children }) => (
          <a href={href} target="_blank" rel="noreferrer"
             className="text-brass-300 underline decoration-brass-300/30 hover:decoration-brass-300">
            {children}
          </a>
        ),
        // Slightly smaller than the prose around it: these are identifiers
        // dropped mid-sentence, and at matching size the monospace face reads
        // as bigger than the text it sits in.
        code: ({ children }) => (
          <code className="rounded bg-raised px-1 py-px font-mono text-[0.92em] text-brass-200">
            {children}
          </code>
        ),
      }}
    >
      {children}
    </ReactMarkdown>
  );
}
