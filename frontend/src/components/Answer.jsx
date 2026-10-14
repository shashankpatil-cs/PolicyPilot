import { useRef, useState } from 'react';

// Splitting on a capturing group puts the markers at the odd indices.
const MARKER = /(\[S\d+(?:\s*,\s*S\d+)*\])/;

function pageText(c) {
  const label = c.page_label && String(c.page_label) !== String(c.page_number) ? ` (printed p. ${c.page_label})` : '';
  return `PDF page ${c.page_number}${label}`;
}

function Inline({ text, bySid, onPick }) {
  return text.split(MARKER).map((part, i) => {
    if (i % 2 === 0) return <span key={i}>{part}</span>;
    const ids = part.match(/S\d+/g) || [];
    return (
      <span key={i} className="chips">
        {ids.map((sid) => {
          const c = bySid[sid];
          return (
            <button
              key={sid}
              type="button"
              className={`chip ${c ? '' : 'chip-bad'}`}
              title={c ? c.quote : 'Unverified source'}
              onClick={() => c && onPick(sid)}
            >
              {c ? `${c.report_year} · p.${c.page_number}` : sid}
            </button>
          );
        })}
      </span>
    );
  });
}

function Body({ text, bySid, onPick }) {
  const blocks = [];
  let list = null;
  text.split('\n').forEach((raw, idx) => {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    if (bullet) {
      if (!list) {
        list = [];
        blocks.push({ type: 'ul', items: list, key: idx });
      }
      list.push(bullet[1]);
    } else {
      list = null;
      if (line.trim()) blocks.push({ type: 'p', text: line, key: idx });
    }
  });
  return blocks.map((b) =>
    b.type === 'ul' ? (
      <ul key={b.key}>
        {b.items.map((it, i) => (
          <li key={i}><Inline text={it} bySid={bySid} onPick={onPick} /></li>
        ))}
      </ul>
    ) : (
      <p key={b.key}><Inline text={b.text} bySid={bySid} onPick={onPick} /></p>
    ),
  );
}

export default function Answer({ message, onFeedback }) {
  const [active, setActive] = useState(null);
  const refs = useRef({});
  const citations = message.citations || [];
  const bySid = Object.fromEntries(citations.map((c) => [c.source_id, c]));
  const meta = message.meta || {};
  const insufficient = message.sufficient === false;

  function pick(sid) {
    setActive(sid);
    refs.current[sid]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  return (
    <div className={`answer ${insufficient ? 'answer-insufficient' : ''}`}>
      {insufficient && <div className="badge warn">Insufficient evidence</div>}
      <div className="answer-text">
        <Body text={message.content} bySid={bySid} onPick={pick} />
      </div>

      {citations.length > 0 && (
        <section className="sources">
          <h4>Sources <span className="badge ok">verified against report text</span></h4>
          {citations.map((c) => (
            <article
              key={c.source_id}
              ref={(el) => (refs.current[c.source_id] = el)}
              className={`source ${active === c.source_id ? 'source-active' : ''}`}
            >
              <header>
                <strong>Report {c.report_year}</strong>
                <span>{pageText(c)}</span>
              </header>
              <blockquote>“{c.quote}”</blockquote>
              <footer>{c.report_title}</footer>
            </article>
          ))}
        </section>
      )}

      <footer className="answer-foot">
        <span className="muted small">
          {meta.mode === 'research' ? 'Research agent' : 'Standard'}
          {meta.retrieved ? ` · ${meta.retrieved.length} passages retrieved` : ''}
          {meta.latency_ms ? ` · ${(meta.latency_ms / 1000).toFixed(1)}s` : ''}
        </span>
        <span className="feedback">
          <button className={message.feedback === 1 ? 'on' : ''} onClick={() => onFeedback(message, 1)} aria-label="Helpful" title="Helpful">👍</button>
          <button className={message.feedback === -1 ? 'on' : ''} onClick={() => onFeedback(message, -1)} aria-label="Not helpful" title="Not helpful">👎</button>
        </span>
      </footer>

      {meta.retrieved?.length > 0 && (
        <details className="details">
          <summary>Retrieval details</summary>
          <table>
            <thead><tr><th>ID</th><th>Report</th><th>PDF page</th><th>Relevance</th></tr></thead>
            <tbody>
              {meta.retrieved.map((r) => (
                <tr key={r.source_id}>
                  <td>{r.source_id}</td><td>{r.report_year}</td><td>{r.page_number}</td><td>{Number(r.relevance).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {meta.steps?.length > 0 && (
            <p className="muted small">Agent tool calls: {meta.steps.map((s) => s.tool).join(' → ')}</p>
          )}
        </details>
      )}
    </div>
  );
}
