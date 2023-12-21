import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    proxy: {
      '/api': 'http://localhost:7860',
      '/api/ws': {
        target: 'ws://localhost:7860',
        ws: true
      }
    },
  }
});