import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './i18n' // Initialize i18n before App
import App from './App.tsx'
import { AuthProvider } from './contexts/AuthContext'
import { CitationProvider } from './contexts/CitationContext'

// Auto-reload on stale chunk imports after deployment (guard against infinite loop)
window.addEventListener('vite:preloadError', () => {
  const key = 'vite-preload-reloaded';
  if (!sessionStorage.getItem(key)) {
    sessionStorage.setItem(key, '1');
    window.location.reload();
  }
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AuthProvider>
      <CitationProvider>
        <App />
      </CitationProvider>
    </AuthProvider>
  </StrictMode>,
)
