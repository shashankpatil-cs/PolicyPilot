import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';
import { useAuth } from '../auth';
import Answer from '../components/Answer';
import { config } from '../config';

const MODES = [
  {
    id: 'standard',
    label: 'Standard Mode',
    hint: 'Ultra-fast: vector + keyword hybrid search with direct fact extraction',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
      </svg>
    ),
  },
  {
    id: 'research',
    label: 'Research Agent',
    hint: 'In-depth multi-turn agent that cross-examines and compares across report years',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8" />
        <line x1="21" y1="21" x2="16.65" y2="16.65" />
      </svg>
    ),
  },
];

export default function Chat() {
  const { user, logout } = useAuth();
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [mode, setMode] = useState('standard');
  const [reportYears, setReportYears] = useState([]);
  const [years, setYears] = useState([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const bottomRef = useRef(null);

  const loadConversations = useCallback(() => api.conversations().then(setConversations).catch(() => {}), []);

  useEffect(() => {
    loadConversations();
    api.reportYears().then(setReportYears).catch(() => {});
  }, [loadConversations]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, pending]);

  async function open(id) {
    setError('');
    try {
      const conv = await api.conversation(id);
      setActiveId(conv.id);
      setMessages(conv.messages);
    } catch (e) {
      setError(e.message);
    }
  }

  function newChat() {
    setActiveId(null);
    setMessages([]);
    setError('');
  }

  async function remove(id, e) {
    e.stopPropagation();
    if (!window.confirm('Delete this conversation?')) return;
    await api.deleteConversation(id).catch((err) => setError(err.message));
    if (id === activeId) newChat();
    loadConversations();
  }

  async function send(e) {
    e?.preventDefault();
    const question = input.trim();
    if (!question || pending) return;
    setError('');
    setInput('');
    setPending(true);
    setMessages((m) => [...m, { id: `tmp-${Date.now()}`, role: 'user', content: question }]);
    try {
      const res = await api.ask({ question, conversation_id: activeId, mode, years: years.length ? years : null });
      setActiveId(res.conversation.id);
      setMessages((m) => [...m, res.message]);
      loadConversations();
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(false);
    }
  }

  async function feedback(message, rating) {
    try {
      const updated = await api.feedback(message.id, rating);
      setMessages((m) => m.map((x) => (x.id === message.id ? updated : x)));
    } catch (err) {
      setError(err.message);
    }
  }

  const toggleYear = (y) => setYears((cur) => (cur.includes(y) ? cur.filter((x) => x !== y) : [...cur, y]));
  const uniqueYears = [...new Set(reportYears.map((r) => r.report_year))];

  return (
    <div className="chat-layout">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="side-head">
          <div className="brand-badge">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <div className="brand-text-wrap">
            <h1 className="brand-title">PolicyPilot</h1>
            <span className="brand-subtitle">RBI Intelligence AI</span>
          </div>
        </div>

        {/* Modernized New Chat Button */}
        <div className="side-action-wrap">
          <button type="button" className="btn-modern-new" onClick={newChat}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            <span>New Chat</span>
          </button>
        </div>

        {/* Conversation History */}
        <div className="history-header">
          <span>Recent Inquiries</span>
          <span className="count-pill">{conversations.length}</span>
        </div>

        <nav className="history" aria-label="Chat history">
          {conversations.length === 0 && (
            <div className="history-empty">
              <p>No conversations yet.</p>
              <small>Ask a question about monetary policy or inflation to start.</small>
            </div>
          )}
          {conversations.map((c) => (
            <div
              key={c.id}
              className={`history-item ${c.id === activeId ? 'active' : ''}`}
              onClick={() => open(c.id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === 'Enter' && open(c.id)}
            >
              <div className="history-item-left">
                <svg className="chat-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
                <span className="title">{c.title}</span>
              </div>
              <button
                type="button"
                className="btn-delete-conv"
                onClick={(e) => remove(c.id, e)}
                aria-label="Delete conversation"
                title="Delete conversation"
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="3 6 5 6 21 6" />
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
              </button>
            </div>
          ))}
        </nav>

        {/* Sidebar Footer with Modernized Action Buttons */}
        <div className="side-foot">
          <div className="user-profile-badge">
            <span className="avatar-circle">{user.username[0].toUpperCase()}</span>
            <div className="user-info">
              <strong className="user-name">{user.username}</strong>
              <span className="user-role">{user.role === 'admin' ? 'Administrator' : 'Standard User'}</span>
            </div>
          </div>

          <div className="foot-actions">
            {user.role === 'admin' && (
              <Link className="btn-modern-secondary admin-btn" to="/admin">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 20h9" />
                  <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
                </svg>
                <span>Admin Console</span>
              </Link>
            )}
            <button type="button" className="btn-modern-signout" onClick={logout}>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                <polyline points="16 17 21 12 16 7" />
                <line x1="21" y1="12" x2="9" y2="12" />
              </svg>
              <span>Sign Out</span>
            </button>
          </div>
        </div>
      </aside>

      {/* Main Chat Workspace */}
      <main className="chat-main">
        <header className="chat-top-header">
          <div className="top-branding">
            <span className="top-title">PolicyPilot Chat</span>
            <span className="top-tag">Grounded with Citations</span>
          </div>
          <div className="top-stats">
            <span className="stat-pill">
              <span className="status-indicator-green" />
              {uniqueYears.length} Reports Indexed
            </span>
          </div>
        </header>

        <div className="thread">
          {messages.length === 0 && !pending && (
            <div className="empty-hero">
              <div className="hero-icon-circle">
                <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                  <polyline points="10 9 9 9 8 9" />
                </svg>
              </div>
              <h2>Ask PolicyPilot</h2>
              <p className="hero-desc">
                High-precision AI analysis for Reserve Bank of India Annual Reports. Every response is grounded in official PDF chunks with strict anti-hallucination verification.
              </p>

              <div className="quick-starters">
                <button
                  type="button"
                  className="starter-card"
                  onClick={() => setInput('What were the primary drivers of headline inflation during 2023-24?')}
                >
                  <span className="starter-label">Inflation & Prices</span>
                  <span className="starter-text">"What were the primary drivers of headline inflation during 2023-24?"</span>
                </button>
                <button
                  type="button"
                  className="starter-card"
                  onClick={() => setInput('Compare CPI inflation in 2022-23 vs 2023-24.')}
                >
                  <span className="starter-label">Year Comparison</span>
                  <span className="starter-text">"Compare CPI inflation in 2022-23 vs 2023-24."</span>
                </button>
                <button
                  type="button"
                  className="starter-card"
                  onClick={() => setInput('What are the key policy initiatives taken for banking sector resilience?')}
                >
                  <span className="starter-label">Banking Stability</span>
                  <span className="starter-text">"What are the key policy initiatives taken for banking sector resilience?"</span>
                </button>
              </div>
            </div>
          )}

          {messages.map((m) =>
            m.role === 'user' ? (
              <div key={m.id} className="bubble-wrapper user-wrapper">
                <div className="bubble user">{m.content}</div>
              </div>
            ) : (
              <div key={m.id} className="bubble-wrapper assistant-wrapper">
                <div className="assistant-avatar">
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M12 16v-4" />
                    <path d="M12 8h.01" />
                  </svg>
                </div>
                <div className="bubble assistant">
                  <Answer message={m} onFeedback={feedback} />
                </div>
              </div>
            ),
          )}

          {pending && (
            <div className="bubble-wrapper assistant-wrapper">
              <div className="assistant-avatar pulse">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <circle cx="12" cy="12" r="10" />
                </svg>
              </div>
              <div className="bubble assistant thinking-card">
                <div className="thinking-content">
                  <div className="loading-bars">
                    <span /><span /><span />
                  </div>
                  <div className="thinking-text">
                    <strong>{mode === 'research' ? 'Research Agent is active' : 'PolicyPilot is analyzing'}</strong>
                    <p>{mode === 'research' ? 'Formulating research plan and searching indexed reports…' : 'Retrieving dense & sparse evidence chunks and verifying citations…'}</p>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input Composer */}
        <form className="composer-card" onSubmit={send}>
          {error && <div className="alert error-banner" role="alert">{error}</div>}

          <div className="controls-bar">
            {/* Mode Switcher */}
            <div className="segmented-control" role="group" aria-label="Answer mode">
              {MODES.map((m) => (
                <button
                  type="button"
                  key={m.id}
                  title={m.hint}
                  className={`segmented-tab ${mode === m.id ? 'active' : ''}`}
                  onClick={() => setMode(m.id)}
                >
                  {m.icon}
                  <span>{m.label}</span>
                </button>
              ))}
            </div>

            {/* Year Filters */}
            {uniqueYears.length > 0 && (
              <div className="filter-pill-container" aria-label="Restrict to report years">
                <span className="filter-title">Year Filter:</span>
                <button
                  type="button"
                  className={`filter-pill ${years.length === 0 ? 'selected' : ''}`}
                  onClick={() => setYears([])}
                >
                  All Reports
                </button>
                {uniqueYears.map((y) => (
                  <button
                    type="button"
                    key={y}
                    className={`filter-pill ${years.includes(y) ? 'selected' : ''}`}
                    onClick={() => toggleYear(y)}
                  >
                    {y}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="input-box-wrapper">
            <textarea
              className="chat-textarea"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) send(e);
              }}
              placeholder="Ask PolicyPilot about monetary policy, inflation targets, forex reserves, digital banking…"
              rows={2}
            />
            <button
              type="submit"
              className="btn-send-primary"
              disabled={pending || !input.trim()}
              title="Submit query"
            >
              <span>Ask</span>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>
          <div className="composer-hints">
            <span>Press <kbd>Enter</kbd> to send, <kbd>Shift + Enter</kbd> for new line</span>
            <span className="grounded-guarantee">✓ Strictly verified against indexed RBI reports</span>
          </div>
        </form>
      </main>
    </div>
  );
}
