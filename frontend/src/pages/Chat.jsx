import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';
import { useAuth } from '../auth';
import Answer from '../components/Answer';
import { config } from '../config';

const MODES = [
  { id: 'standard', label: 'Standard', hint: 'Fast: plan → retrieve → answer → verify' },
  { id: 'research', label: 'Research agent', hint: 'Slower: an agent searches and compares across reports with tools' },
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
      <aside className="sidebar">
        <div className="side-head">
          <span className="brand-mark small" aria-hidden="true">₹</span>
          <strong>{config.appName}</strong>
        </div>
        <button className="btn new" onClick={newChat}>＋ New chat</button>
        <nav className="history" aria-label="Chat history">
          {conversations.length === 0 && <p className="muted small pad">No conversations yet.</p>}
          {conversations.map((c) => (
            <div
              key={c.id}
              className={`history-item ${c.id === activeId ? 'active' : ''}`}
              onClick={() => open(c.id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === 'Enter' && open(c.id)}
            >
              <span className="title">{c.title}</span>
              <button className="x" onClick={(e) => remove(c.id, e)} aria-label="Delete conversation">×</button>
            </div>
          ))}
        </nav>
        <div className="side-foot">
          <div className="who">
            <span className="avatar">{user.username[0].toUpperCase()}</span>
            <span>{user.username}</span>
          </div>
          {user.role === 'admin' && <Link className="link" to="/admin">Admin dashboard</Link>}
          <button className="link" onClick={logout}>Sign out</button>
        </div>
      </aside>

      <main className="chat-main">
        <div className="thread">
          {messages.length === 0 && !pending && (
            <div className="empty">
              <h2>Ask the Annual Reports</h2>
              <p className="muted">
                Answers come only from the indexed RBI Annual Reports and cite the report year and page.
                {uniqueYears.length === 0 && ' No reports are indexed yet. Ask an administrator to upload one.'}
              </p>
            </div>
          )}
          {messages.map((m) =>
            m.role === 'user' ? (
              <div key={m.id} className="bubble user">{m.content}</div>
            ) : (
              <div key={m.id} className="bubble assistant">
                <Answer message={m} onFeedback={feedback} />
              </div>
            ),
          )}
          {pending && (
            <div className="bubble assistant thinking">
              <span className="dots"><i /><i /><i /></span>
              {mode === 'research' ? 'Research agent is searching the reports…' : 'Retrieving and verifying evidence…'}
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <form className="composer" onSubmit={send}>
          {error && <div className="alert error" role="alert">{error}</div>}
          <div className="controls">
            <div className="segmented" role="group" aria-label="Answer mode">
              {MODES.map((m) => (
                <button type="button" key={m.id} title={m.hint} className={mode === m.id ? 'on' : ''} onClick={() => setMode(m.id)}>
                  {m.label}
                </button>
              ))}
            </div>
            {uniqueYears.length > 0 && (
              <div className="year-filter" aria-label="Restrict to report years">
                <span className="muted small">Reports:</span>
                <button type="button" className={`pill ${years.length === 0 ? 'on' : ''}`} onClick={() => setYears([])}>All</button>
                {uniqueYears.map((y) => (
                  <button type="button" key={y} className={`pill ${years.includes(y) ? 'on' : ''}`} onClick={() => toggleYear(y)}>{y}</button>
                ))}
              </div>
            )}
          </div>
          <div className="input-row">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) send(e);
              }}
              placeholder="Ask about monetary policy, inflation, banking sector, currency…"
              rows={2}
            />
            <button className="btn primary" disabled={pending || !input.trim()}>Ask</button>
          </div>
        </form>
      </main>
    </div>
  );
}
