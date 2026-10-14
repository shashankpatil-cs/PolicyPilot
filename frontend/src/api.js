import { config } from './config';

let onUnauthorized = () => {};
export const setUnauthorizedHandler = (fn) => {
  onUnauthorized = fn;
};

export const getToken = () => localStorage.getItem(config.tokenKey);
export const setToken = (t) =>
  t ? localStorage.setItem(config.tokenKey, t) : localStorage.removeItem(config.tokenKey);

function messageFrom(detail, fallback) {
  if (!detail) return fallback;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
  return fallback;
}

async function request(path, { method = 'GET', body, form, auth = true } = {}) {
  const headers = {};
  const token = getToken();
  if (auth && token) headers.Authorization = `Bearer ${token}`;
  let payload;
  if (form) {
    payload = form;
  } else if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    payload = JSON.stringify(body);
  }

  let res;
  try {
    res = await fetch(`${config.apiBase}${path}`, { method, headers, body: payload });
  } catch {
    throw new Error('Cannot reach the server. Is the backend running?');
  }

  if (res.status === 204) return null;
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    if (res.status === 401 && auth) onUnauthorized();
    throw new Error(messageFrom(data?.detail, `Request failed (${res.status})`));
  }
  return data;
}

export const api = {
  register: (username, password) => request('/auth/register', { method: 'POST', body: { username, password }, auth: false }),
  login: (username, password) => request('/auth/login', { method: 'POST', body: { username, password }, auth: false }),
  adminLogin: (username, password) => request('/auth/admin/login', { method: 'POST', body: { username, password }, auth: false }),
  me: () => request('/auth/me'),

  reportYears: () => request('/chat/reports'),
  conversations: () => request('/chat/conversations'),
  conversation: (id) => request(`/chat/conversations/${id}`),
  deleteConversation: (id) => request(`/chat/conversations/${id}`, { method: 'DELETE' }),
  ask: (payload) => request('/chat/ask', { method: 'POST', body: payload }),
  feedback: (messageId, rating) => request(`/chat/messages/${messageId}/feedback`, { method: 'POST', body: { rating } }),

  adminStats: () => request('/admin/stats'),
  adminReports: () => request('/admin/reports'),
  uploadReport: (form) => request('/admin/reports', { method: 'POST', form }),
  reindexReport: (id) => request(`/admin/reports/${id}/reindex`, { method: 'POST' }),
  deleteReport: (id) => request(`/admin/reports/${id}`, { method: 'DELETE' }),
};
