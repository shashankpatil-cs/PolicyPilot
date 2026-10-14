import { Navigate, Route, Routes } from 'react-router-dom';
import { useAuth } from './auth';
import Admin from './pages/Admin';
import Chat from './pages/Chat';
import Login from './pages/Login';
import Register from './pages/Register';

function Protected({ children, admin = false }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="splash">Loading…</div>;
  if (!user) return <Navigate to={admin ? '/admin/login' : '/login'} replace />;
  if (admin && user.role !== 'admin') return <Navigate to="/" replace />;
  return children;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/admin/login" element={<Login admin />} />
      <Route path="/register" element={<Register />} />
      <Route path="/" element={<Protected><Chat /></Protected>} />
      <Route path="/admin" element={<Protected admin><Admin /></Protected>} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
