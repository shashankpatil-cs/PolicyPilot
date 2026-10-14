import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';
import { useAuth } from '../auth';
import { config } from '../config';

const fmtSize = (b) => (b >= 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`);

function StatusBadge({ report }) {
  const map = { indexed: 'ok', indexing: 'info', failed: 'bad', uploaded: 'warn' };
  return (
    <span className={`status-badge-modern ${map[report.status] || ''}`}>
      <span className="dot" />
      {report.status.toUpperCase()}
    </span>
  );
}

function Progress({ report }) {
  if (report.status !== 'indexing') return null;
  const pct = report.chunks_total ? Math.round((report.chunks_done / report.chunks_total) * 100) : 0;
  return (
    <div className="progress-container" title={`${report.chunks_done} / ${report.chunks_total} chunks processed`}>
      <div className="progress-bar-bg">
        <div className="progress-bar-fill" style={{ width: `${report.chunks_total ? pct : 10}%` }} />
      </div>
      <span className="progress-label">{report.chunks_total ? `${pct}% (${report.chunks_done}/${report.chunks_total})` : 'Extracting text…'}</span>
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
    const t = setInterval(refresh, config.pollMs || 3000);
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
      setNotice(`Report "${title}" uploaded. Background extraction, chunking, and embedding pipeline started.`);
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

  const statItems = stats && [
    { label: 'Total Reports', value: stats.reports, icon: '📚' },
    { label: 'Fully Indexed', value: stats.reports_indexed, icon: '✅' },
    { label: 'Vector Chunks', value: stats.chunks.toLocaleString(), icon: '🧩' },
    { label: 'Active Users', value: stats.users, icon: '👥' },
    { label: 'Conversations', value: stats.conversations, icon: '💬' },
    { label: 'Messages Exchanged', value: stats.messages, icon: '📨' },
  ];

  return (
    <div className="admin-page">
      {/* Top Navbar */}
      <header className="admin-navbar">
        <div className="admin-brand">
          <div className="brand-badge small">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <div className="admin-brand-titles">
            <strong>PolicyPilot</strong>
            <span className="badge-admin-tag">Admin Console</span>
          </div>
        </div>

        {/* Modernized Top Action Buttons: Open Chat & Sign Out */}
        <div className="admin-nav-actions">
          <Link className="btn-modern-open-chat" to="/">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            <span>Open Chat</span>
          </Link>

          <div className="admin-user-pill">
            <span className="avatar-micro">{user.username[0].toUpperCase()}</span>
            <span>{user.username}</span>
          </div>

          <button type="button" className="btn-modern-signout-nav" onClick={logout} title="Sign Out of PolicyPilot">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
              <polyline points="16 17 21 12 16 7" />
              <line x1="21" y1="12" x2="9" y2="12" />
            </svg>
            <span>Sign Out</span>
          </button>
        </div>
      </header>

      <main className="admin-container">
        {/* Alerts */}
        {error && <div className="alert error-banner" role="alert">{error}</div>}
        {notice && <div className="alert success-banner" role="status">{notice}</div>}

        {/* High-Level System Metrics */}
        {statItems && (
          <section className="stats-dashboard">
            {statItems.map((st) => (
              <div className="stat-card" key={st.label}>
                <div className="stat-card-top">
                  <span className="stat-title">{st.label}</span>
                  <span className="stat-icon">{st.icon}</span>
                </div>
                <span className="stat-number">{st.value}</span>
              </div>
            ))}
          </section>
        )}

        {/* Modernized Upload Panel */}
        <section className="panel-card upload-panel">
          <div className="panel-header">
            <div>
              <h3>Upload & Index Annual Report</h3>
              <p className="panel-sub">Upload an RBI Annual Report PDF to automatically chunk, vectorize on CUDA GPU, and index into pgvector.</p>
            </div>
            <span className="badge info">PDF Documents (Up to 200MB)</span>
          </div>

          <form onSubmit={upload} className="modern-upload-form">
            <div className="form-grid">
              <label className="field-group">
                <span className="field-label">Document Title</span>
                <input
                  className="modern-input"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="e.g. RBI Annual Report 2023-24"
                  required
                />
              </label>

              <label className="field-group">
                <span className="field-label">Fiscal / Report Year</span>
                <input
                  className="modern-input"
                  value={year}
                  onChange={(e) => setYear(e.target.value)}
                  placeholder="e.g. 2023-24"
                  required
                />
              </label>

              <label className="field-group file-field">
                <span className="field-label">Report PDF File</span>
                <div className="file-input-wrapper">
                  <input
                    ref={fileInput}
                    type="file"
                    accept="application/pdf,.pdf"
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                    required
                  />
                  <div className="file-custom-btn">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="17 8 12 3 7 8" />
                      <line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                    <span>{file ? file.name : 'Select PDF document…'}</span>
                  </div>
                </div>
              </label>
            </div>

            <div className="upload-btn-row">
              <button
                type="submit"
                className="btn-upload-primary"
                disabled={uploading || !file || !title || !year}
              >
                {uploading ? (
                  <>
                    <span className="spinner-dots" />
                    <span>Uploading & Initializing…</span>
                  </>
                ) : (
                  <>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="16 16 12 12 8 16" />
                      <line x1="12" y1="12" x2="12" y2="21" />
                      <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" />
                      <polyline points="16 16 12 12 8 16" />
                    </svg>
                    <span>Upload & Index Report</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </section>

        {/* Reports Management Table */}
        <section className="panel-card">
          <div className="panel-header">
            <div>
              <h3>Indexed Reports Knowledge Base</h3>
              <p className="panel-sub">Manage indexed documents, inspect chunk counts, re-index or remove reports.</p>
            </div>
            <button type="button" className="btn-refresh-table" onClick={refresh} title="Refresh report list">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="23 4 23 10 17 10" />
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
              </svg>
              <span>Refresh</span>
            </button>
          </div>

          {reports.length === 0 ? (
            <div className="table-empty-state">
              <p>No RBI Annual Reports have been uploaded yet.</p>
              <small>Use the upload form above to add your first PDF report.</small>
            </div>
          ) : (
            <div className="modern-table-wrap">
              <table className="modern-reports-table">
                <thead>
                  <tr>
                    <th>Year</th>
                    <th>Document Title & Details</th>
                    <th>Status & Progress</th>
                    <th>Pages</th>
                    <th>Vector Chunks</th>
                    <th>File Size</th>
                    <th className="th-actions">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {reports.map((r) => (
                    <tr key={r.id}>
                      <td>
                        <span className="year-pill-tag">{r.report_year}</span>
                      </td>
                      <td>
                        <div className="report-title-text">{r.title}</div>
                        <div className="report-filename-text">{r.filename}</div>
                        {r.error && <div className="report-error-msg">{r.error}</div>}
                      </td>
                      <td>
                        <StatusBadge report={r} />
                        <Progress report={r} />
                      </td>
                      <td>
                        <span className="metric-text">{r.page_count ? `${r.page_count} pages` : '—'}</span>
                      </td>
                      <td>
                        <span className="metric-text-bold">
                          {r.status === 'indexed' ? `${r.chunks_total.toLocaleString()} chunks` : '—'}
                        </span>
                      </td>
                      <td>
                        <span className="metric-text">{fmtSize(r.file_size)}</span>
                      </td>
                      <td className="td-actions">
                        <button
                          type="button"
                          className="btn-action-table reindex"
                          disabled={r.status === 'indexing'}
                          onClick={() => act(() => api.reindexReport(r.id))}
                          title="Re-extract and re-index this report"
                        >
                          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="23 4 23 10 17 10" />
                            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
                          </svg>
                          <span>Re-index</span>
                        </button>
                        <button
                          type="button"
                          className="btn-action-table delete"
                          disabled={r.status === 'indexing'}
                          onClick={() => act(() => api.deleteReport(r.id), `Delete "${r.title}" and remove all its indexed chunks from PolicyPilot?`)}
                          title="Delete report"
                        >
                          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="3 6 5 6 21 6" />
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                          </svg>
                          <span>Delete</span>
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
