import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev: proxy API + WebSocket to the FastAPI server so the console uses same-origin
// URLs everywhere (it is also served by FastAPI itself from console/dist).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})
