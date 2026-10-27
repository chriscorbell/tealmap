import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const api = process.env.TEALMAP_API ?? 'http://127.0.0.1:8200'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': api,
      '/maps': api,
    },
  },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 800,
  },
})
