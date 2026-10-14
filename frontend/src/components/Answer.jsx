import { useRef, useState } from 'react';

// Splitting on a capturing group puts the markers at the odd indices.
const MARKER = /(\[S\d+(?:\s*,\s*S\d+)*\])/;

function pageText(c) {
  const label = c.page_label && String(c.page_label) !== String(c.page_number) ? ` (printed p. ${c.page_label})` : '';
  return `PDF Page ${c.page_number}${label}`;
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
              title={c ? `Click to inspect: "${c.quote.slice(0, 80)}..."` : 'Unverified source'}
              onClick={() => onPick(sid)}
            >
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: 3 }}>
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
              </svg>
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
  // Default to false: only the answer is shown until the user clicks to view sources
  const [showSources, setShowSources] = useState(false);
  const refs = useRef({});
  const citations = message.citations || [];
  const bySid = Object.fromEntries(citations.map((c) => [c.source_id, c]));
  const meta = message.meta || {};
  const insufficient = message.sufficient === false;

  function pick(sid) {
    setActive(sid);
    setShowSources(true);
    setTimeout(() => {
      refs.current[sid]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 60);
  }

  return (
    <div className={`answer ${insufficient ? 'answer-insufficient' : ''}`}>
      {insufficient && (
        <div className="insufficient-banner">
          <span className="badge warn">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: 4, verticalAlign: -1 }}>
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            Strict Validation Notice
          </span>
          <span className="insufficient-desc">The evidence retrieved did not meet strict confidence and citation thresholds for this query.</span>
        </div>
      )}

      {/* Primary Answer Content (always visible by default) */}
      <div className="answer-text">
        <Body text={message.content} bySid={bySid} onPick={pick} />
      </div>

      {/* Collapsible Sources Dropdown (Collapsed by default) */}
      {citations.length > 0 && (
        <div className="sources-container">
          <button
            type="button"
            className={`sources-toggle-btn ${showSources ? 'active' : ''}`}
            onClick={() => setShowSources(!showSources)}
            aria-expanded={showSources}
          >
            <div className="sources-toggle-title">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
              </svg>
              <span>Verified Sources ({citations.length})</span>
              <span className="badge ok verified-pill">
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: 2 }}>
                  <polyline points="20 6 9 17 4 12" />
                </svg>
                verbatim matched
              </span>
            </div>
            <span className="chevron-indicator">
              {showSources ? (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="18 15 12 9 6 15" />
                </svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="6 9 12 15 18 9" />
                </svg>
              )}
            </span>
          </button>

          {showSources && (
            <section className="sources-dropdown-content">
              {citations.map((c) => (
                <article
                  key={c.source_id}
                  ref={(el) => (refs.current[c.source_id] = el)}
                  className={`source-card ${active === c.source_id ? 'source-active' : ''}`}
                >
                  <header className="source-header">
                    <div className="source-meta">
                      <span className="source-tag">{c.source_id}</span>
                      <strong className="source-year">Report {c.report_year}</strong>
                      <span className="source-page">{pageText(c)}</span>
                    </div>
                    <span className="source-report-title">{c.report_title}</span>
                  </header>
                  <blockquote className="source-quote">“{c.quote}”</blockquote>
                </article>
              ))}
            </section>
          )}
        </div>
      )}

      {/* Answer Footer with metadata & modern feedback buttons */}
      <footer className="answer-foot">
        <div className="meta-pill-group">
          <span className="mode-badge">
            <span className="mode-dot" />
            {meta.mode === 'research' ? 'Research Agent' : 'Standard Pipeline'}
          </span>
          {meta.retrieved && (
            <span className="meta-info">
              {meta.retrieved.length} chunks analyzed
            </span>
          )}
          {meta.latency_ms && (
            <span className="meta-info">
              {(meta.latency_ms / 1000).toFixed(1)}s
            </span>
          )}
        </div>

        <div className="feedback-group">
          <button
            type="button"
            className={`btn-feedback ${message.feedback === 1 ? 'active-like' : ''}`}
            onClick={() => onFeedback(message, 1)}
            aria-label="Helpful"
            title="Helpful response"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
            </svg>
          </button>
          <button
            type="button"
            className={`btn-feedback ${message.feedback === -1 ? 'active-dislike' : ''}`}
            onClick={() => onFeedback(message, -1)}
            aria-label="Not helpful"
            title="Not helpful response"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3" />
            </svg>
          </button>
        </div>
      </footer>

      {/* Retrieval Details Collapsible */}
      {meta.retrieved?.length > 0 && (
        <details className="retrieval-details">
          <summary>
            <span>Retrieval & Reasoning Inspection</span>
          </summary>
          <div className="details-table-wrap">
            <table className="details-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Year</th>
                  <th>PDF Page</th>
                  <th>Cosine Similarity</th>
                </tr>
              </thead>
              <tbody>
                {meta.retrieved.map((r) => (
                  <tr key={r.source_id}>
                    <td><code>{r.source_id}</code></td>
                    <td><strong>{r.report_year}</strong></td>
                    <td>p. {r.page_number}</td>
                    <td>
                      <span className="similarity-bar-wrap">
                        <span className="similarity-bar" style={{ width: `${Math.min(100, Math.round(Number(r.relevance) * 100))}%` }} />
                        <span>{Number(r.relevance).toFixed(3)}</span>
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {meta.steps?.length > 0 && (
              <div className="agent-steps-trail">
                <strong>Agent Execution Trail:</strong> {meta.steps.map((s, idx) => (
                  <span key={idx} className="step-tag">{s.tool}</span>
                ))}
              </div>
            )}
          </div>
        </details>
      )}
    </div>
  );
}
