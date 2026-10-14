import { config } from '../config';

export default function AuthShell({ title, subtitle, children, footer }) {
  return (
    <div className="auth-page">
      <aside className="auth-aside">
        <div className="brand-mark" aria-hidden="true">₹</div>
        <h1>{config.appName}</h1>
        <p>
          Ask questions across multiple years of RBI Annual Reports. Every answer is drawn only from the
          reports and carries report-year and page citations you can check.
        </p>
        <ul>
          <li>Grounded in retrieved passages, never memory</li>
          <li>Citations verified against the source text</li>
          <li>Says so when the evidence isn&apos;t there</li>
        </ul>
      </aside>
      <main className="auth-main">
        <div className="auth-card">
          <h2>{title}</h2>
          {subtitle && <p className="muted">{subtitle}</p>}
          {children}
          {footer && <p className="auth-footer">{footer}</p>}
        </div>
      </main>
    </div>
  );
}
