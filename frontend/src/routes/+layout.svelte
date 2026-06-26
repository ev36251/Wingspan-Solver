<script>
	import '../app.css';
	import { onMount } from 'svelte';

	let theme = 'light';

	onMount(() => {
		const saved = localStorage.getItem('wingspan_theme');
		if (saved === 'light' || saved === 'dark') {
			theme = saved;
		} else if (window.matchMedia?.('(prefers-color-scheme: dark)').matches) {
			theme = 'dark';
		}
		apply();
	});

	function apply() {
		document.documentElement.dataset.theme = theme;
	}

	function toggle() {
		theme = theme === 'dark' ? 'light' : 'dark';
		try {
			localStorage.setItem('wingspan_theme', theme);
		} catch {
			// ignore storage failures
		}
		apply();
	}
</script>

<div class="app-layout">
	<header>
		<h1>Wingspan Solver</h1>
		<button
			class="theme-toggle"
			on:click={toggle}
			title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
			aria-label="Toggle dark mode"
		>
			{theme === 'dark' ? '☀️' : '🌙'}
		</button>
	</header>
	<main>
		<slot />
	</main>
</div>

<style>
	.app-layout {
		max-width: 1440px;
		margin: 0 auto;
		padding: 16px;
	}

	header {
		margin-bottom: 20px;
		padding-bottom: 12px;
		border-bottom: 2px solid var(--border);
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
	}

	header h1 {
		color: var(--accent);
		font-size: 1.5rem;
	}

	.theme-toggle {
		font-size: 1.1rem;
		line-height: 1;
		padding: 7px 11px;
		border-radius: 999px;
	}

	@media (max-width: 640px) {
		.app-layout {
			padding: 10px;
		}
		header {
			margin-bottom: 12px;
			padding-bottom: 8px;
		}
		header h1 {
			font-size: 1.25rem;
		}
	}
</style>
