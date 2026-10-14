import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import { AuthProvider } from './auth';
import { config, missingConfig } from './config';
import './styles.css';

document.title = config.appName || document.title;

const root = ReactDOM.createRoot(document.getElementById('root'));

if (missingConfig.length) {
  root.render(
    <div className="fatal">
      <h1>Frontend is not configured</h1>
      <p>
        Missing in <code>frontend/.env</code>: <code>{missingConfig.join(', ')}</code>
      </p>
      <p>
        Copy <code>.env.example</code> to <code>.env</code> and restart the dev server.
      </p>
    </div>,
  );
} else {
  root.render(
    <React.StrictMode>
      <BrowserRouter>
        <AuthProvider>
          <App />
        </AuthProvider>
      </BrowserRouter>
    </React.StrictMode>,
  );
}
