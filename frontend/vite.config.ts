import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-admin': ['react-admin', 'ra-data-simple-rest'],
          'vendor-pdf': ['react-pdf', 'pdfjs-dist'],
          'vendor-markdown': ['react-markdown', 'rehype-highlight', 'rehype-raw', 'remark-gfm'],
        },
      },
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: [
      'sujbot.fjfi.cvut.cz',
      'localhost',
      '127.0.0.1'
    ]
  },
  optimizeDeps: {
    include: ['pdfjs-dist']
  }
})
