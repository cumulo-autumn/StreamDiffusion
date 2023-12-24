import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/kit/vite';
/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess({ postcss: true }),
  kit: {
    adapter: adapter({
      pages: 'public',
      assets: 'public',
      fallback: undefined,
      precompress: false,
      strict: true
    })
  }
};

export default config;
