<script>
	import '../app.css';
</script>

<div class="app-layout">
	<header>
		<h1>Wingspan Solver</h1>
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
	}

	header h1 {
		color: var(--accent);
		font-size: 1.5rem;
	}
</style>
