export default function AuthShell({ title, subtitle, children, footer }) {
  return (
    <div className="auth-page">
      <aside className="auth-aside">
        <div className="auth-brand-lockup">
          <div className="brand-badge large">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <div>
            <h1 className="auth-brand-name">PolicyPilot</h1>
            <span className="auth-brand-tag">RBI Intelligence AI Platform</span>
          </div>
        </div>

        <p className="auth-hero-desc">
          Intelligence and synthesis across years of official Reserve Bank of India Annual Reports. Every finding is mathematically verified with strict source citations.
        </p>

        <ul className="auth-feature-list">
          <li>
            <div className="feature-icon">✓</div>
            <div>
              <strong>Strict Citation Grounding</strong>
              <p>Every claim is backed by verbatim report text, never hallucinations.</p>
            </div>
          </li>
          <li>
            <div className="feature-icon">✓</div>
            <div>
              <strong>Multi-Year Comparative Analysis</strong>
              <p>Analyze inflation trends, regulatory policies, and macro indicators across reports.</p>
            </div>
          </li>
          <li>
            <div className="feature-icon">✓</div>
            <div>
              <strong>GPU-Accelerated Embeddings</strong>
              <p>Fast BGE-M3 local dense retrieval combined with PostgreSQL full-text search.</p>
            </div>
          </li>
        </ul>

        <div className="auth-aside-footer">
          <span>Enterprise RAG Platform · Powered by PolicyPilot</span>
        </div>
      </aside>

      <main className="auth-main">
        <div className="auth-card">
          <div className="auth-card-header">
            <h2>{title}</h2>
            {subtitle && <p className="auth-card-sub">{subtitle}</p>}
          </div>
          {children}
          {footer && <div className="auth-card-footer">{footer}</div>}
        </div>
      </main>
    </div>
  );
}
