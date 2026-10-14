const env = import.meta.env;

export const config = {
  apiBase: (env.VITE_API_BASE_URL || '').replace(/\/$/, ''),
  appName: env.VITE_APP_NAME,
  tokenKey: env.VITE_TOKEN_STORAGE_KEY,
  pollMs: Number(env.VITE_POLL_INTERVAL_MS),
};

const required = {
  VITE_API_BASE_URL: config.apiBase,
  VITE_APP_NAME: config.appName,
  VITE_TOKEN_STORAGE_KEY: config.tokenKey,
  VITE_POLL_INTERVAL_MS: config.pollMs,
};

export const missingConfig = Object.entries(required)
  .filter(([, v]) => !v)
  .map(([k]) => k);
