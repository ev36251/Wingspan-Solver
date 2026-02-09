import { sveltekit } from '@sveltejs/kit/vite';
import { createLogger, defineConfig } from 'vite';

const logger = createLogger();
const originalWarn = logger.warn;
logger.warn = (msg, options) => {
	const text = typeof msg === 'string' ? msg : String((msg as { message?: unknown })?.message ?? '');
	if (
		text.includes('is not exported by "node_modules/svelte/src/runtime/ssr.js"') ||
		text.includes('is not exported by "node_modules/svelte/src/runtime/index.js"')
	) {
		return;
	}
	originalWarn(msg, options);
};

export default defineConfig({
	plugins: [sveltekit()],
	customLogger: logger,
	build: {
		rollupOptions: {
			onwarn(warning, warn) {
				if (
					warning.code === 'MISSING_EXPORT' &&
					(warning.message.includes('untrack') ||
						warning.message.includes('fork') ||
						warning.message.includes('settled')) &&
					warning.message.includes('svelte/src/runtime')
				) {
					return;
				}
				warn(warning);
			}
		}
	},
	server: {
		proxy: {
			'/api': {
				target: 'http://localhost:8000',
				changeOrigin: true
			}
		}
	}
});
