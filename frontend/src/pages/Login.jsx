import { useState } from 'react';
import { Link, Navigate, useNavigate } from 'react-router-dom';
import AuthShell from '../components/AuthShell';
import { useAuth } from '../auth';

export default function Login({ admin = false }) {
  const { user, login, adminLogin } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to={user.role === 'admin' && admin ? '/admin' : '/'} replace />;

  async function submit(e) {
    e.preventDefault();
    setError('');
    setBusy(true);
    try {
      const u = admin ? await adminLogin(username, password) : await login(username, password);
      navigate(admin && u.role === 'admin' ? '/admin' : '/', { replace: true });
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthShell
      title={admin ? 'Administrator sign in' : 'Welcome back'}
      subtitle={admin ? 'Manage and index RBI Annual Report PDFs.' : 'Sign in to ask questions and see your chat history.'}
      footer={
        admin ? (
          <Link to="/login">← Back to user sign in</Link>
        ) : (
          <>
            New here? <Link to="/register">Create an account</Link> · <Link to="/admin/login">Admin sign in</Link>
          </>
        )
      }
    >
      <form onSubmit={submit} className="form">
        <label>
          Username
          <input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" autoFocus required />
        </label>
        <label>
          Password
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required />
        </label>
        {error && <div className="alert error" role="alert">{error}</div>}
        <button className="btn primary" disabled={busy}>{busy ? 'Signing in…' : 'Sign in'}</button>
      </form>
    </AuthShell>
  );
}
