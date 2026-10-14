import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { api, getToken, setToken, setUnauthorizedHandler } from './api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(Boolean(getToken()));

  const logout = useCallback(() => {
    setToken(null);
    setUser(null);
  }, []);

  useEffect(() => {
    setUnauthorizedHandler(logout);
  }, [logout]);

  useEffect(() => {
    if (!getToken()) return;
    api
      .me()
      .then(setUser)
      .catch(() => setToken(null))
      .finally(() => setLoading(false));
  }, []);

  const accept = useCallback((res) => {
    setToken(res.access_token);
    setUser(res.user);
    return res.user;
  }, []);

  const value = useMemo(
    () => ({
      user,
      loading,
      logout,
      login: async (u, p) => accept(await api.login(u, p)),
      adminLogin: async (u, p) => accept(await api.adminLogin(u, p)),
      register: async (u, p) => accept(await api.register(u, p)),
    }),
    [user, loading, logout, accept],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export const useAuth = () => useContext(AuthContext);
