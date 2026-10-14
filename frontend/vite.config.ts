import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/health': 'http://localhost:8000',
      '/collections': 'http://localhost:8000',
      '/query': 'http://localhost:8000',
      '/ablation': 'http://localhost:8000',
      '/benchmarks': 'http://localhost:8000',
    },
  },
})
