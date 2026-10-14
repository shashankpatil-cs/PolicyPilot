import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';
import { useAuth } from '../auth';
import { config } from '../config';

const fmtSize = (b) => (b >= 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`);

function StatusBadge({ report }) {
  const map = { indexed: 'ok', indexing: 'info', failed: 'bad', uploaded: 'warn' };
  return <span className={`badge ${map[report.status] || ''}`}>{report.status}</span>;
}

function Progress({ report }) {
  if (report.status !== 'indexing') return null;
  const pct = report.chunks_total ? Math.round((report.chunks_done / report.chunks_total) * 100) : 0;
  return (
    <div className="progress" title={`${report.chunks_done} / ${report.chunks_total} chunks`}>
      <div style={{ width: `${report.chunks_total ? pct : 5}%` }} />
      <span>{report.chunks_total ? `${pct}%` : 'extracting…'}</span>
    </div>
  );
}

export default function Admin() {
  const { user, logout } = useAuth();
  const [reports, setReports] = useState([]);
  const [stats, setStats] = useState(null);
  const [title, setTitle] = useState('');
  const [year, setYear] = useState('');
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const fileInput = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const [r, s] = await Promise.all([api.adminReports(), api.adminStats()]);
      setReports(r);
      setStats(s);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const anyIndexing = reports.some((r) => r.status === 'indexing');
  useEffect(() => {
    if (!anyIndexing) return undefined;
    const t = setInterval(refresh, config.pollMs);
    return () => clearInterval(t);
  }, [anyIndexing, refresh]);

  async function upload(e) {
    e.preventDefault();
    if (!file) return;
    setError('');
    setNotice('');
    setUploading(true);
    try {
      const form = new FormData();
      form.append('title', title);
      form.append('report_year', year);
      form.append('file', file);
      await api.uploadReport(form);
      setNotice('Uploaded. Indexing started in the background.');
      setTitle('');
      setYear('');
      setFile(null);
      if (fileInput.current) fileInput.current.value = '';
      refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function act(fn, confirmText) {
    if (confirmText && !window.confirm(confirmText)) return;
    setError('');
    setNotice('');
    try {
      await fn();
      refresh();
    } catch (err) {
      setError(err.message);
    }
  }

  const cards = stats && [
    ['Reports', stats.reports],
    ['Indexed', stats.reports_indexed],
    ['Chunks', stats.chunks.toLocaleString()],
    ['Users', stats.users],
    ['Conversations', stats.conversations],
    ['Messages', stats.messages],
  ];

  return (
    <div className="admin">
      <header className="admin-bar">
        <div className="side-head">
          <span className="brand-mark small" aria-hidden="true">₹</span>
          <strong>{config.appName}</strong>
          <span className="badge info">Admin</span>
        </div>
        <div className="bar-right">
          <Link className="link" to="/">Open chat</Link>
          <span className="muted small">{user.username}</span>
          <button className="link" onClick={logout}>Sign out</button>
        </div>
      </header>

      <div className="admin-body">
        {cards && (
          <section className="stat-grid">
            {cards.map(([label, value]) => (
              <div className="stat" key={label}><span>{value}</span><small>{label}</small></div>
            ))}
          </section>
        )}

        {error && <div className="alert error" role="alert">{error}</div>}
        {notice && <div className="alert ok">{notice}</div>}

        <section className="panel">
          <h3>Upload an Annual Report</h3>
          <form onSubmit={upload} className="upload-form">
            <label>
              Title
              <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="RBI Annual Report 2023-24" required />
            </label>
            <label>
              Report year
              <input value={year} onChange={(e) => setYear(e.target.value)} placeholder="2023-24" required />
            </label>
            <label className="file">
              PDF file
              <input ref={fileInput} type="file" accept="application/pdf,.pdf" onChange={(e) => setFile(e.target.files?.[0] || null)} required />
            </label>
            <button className="btn primary" disabled={uploading || !file}>{uploading ? 'Uploading…' : 'Upload & index'}</button>
          </form>
        </section>

        <section className="panel">
          <h3>Reports</h3>
          {reports.length === 0 ? (
            <p className="muted">No reports uploaded yet.</p>
          ) : (
            <div className="table-wrap">
              <table className="reports">
                <thead>
                  <tr><th>Year</th><th>Title</th><th>Status</th><th>Pages</th><th>Chunks</th><th>Size</th><th /></tr>
                </thead>
                <tbody>
                  {reports.map((r) => (
                    <tr key={r.id}>
                      <td><strong>{r.report_year}</strong></td>
                      <td>
                        {r.title}
                        <div className="muted small">{r.filename}</div>
                        {r.error && <div className="err-text small">{r.error}</div>}
                      </td>
                      <td><StatusBadge report={r} /><Progress report={r} /></td>
                      <td>{r.page_count || '—'}</td>
                      <td>{r.status === 'indexed' ? r.chunks_total.toLocaleString() : '—'}</td>
                      <td>{fmtSize(r.file_size)}</td>
                      <td className="actions">
                        <button
                          className="btn small"
                          disabled={r.status === 'indexing'}
                          onClick={() => act(() => api.reindexReport(r.id))}
                        >Re-index</button>
                        <button
                          className="btn small danger"
                          disabled={r.status === 'indexing'}
                          onClick={() => act(() => api.deleteReport(r.id), `Delete "${r.title}" and all its indexed chunks?`)}
                        >Delete</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
