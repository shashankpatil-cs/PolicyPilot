import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const port = Number(env.VITE_DEV_PORT);
  return {
    plugins: [react()],
    server: {
      host: '0.0.0.0',
      port: port || 5173,
      strictPort: !!port,
    },
  };
});
