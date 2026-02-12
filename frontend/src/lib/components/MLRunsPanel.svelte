<script lang="ts">
	import { getMLRunsDashboard } from '$lib/api/client';
	import type { MLRunsDashboardResponse, MLRunSummary } from '$lib/api/types';
	import { onMount } from 'svelte';

	let loading = false;
	let error = '';
	let dashboard: MLRunsDashboardResponse | null = null;
	let expanded = true;

	const POLL_MS = 10000;
	let timer: ReturnType<typeof setInterval> | null = null;

	async function refresh() {
		loading = true;
		error = '';
		try {
			dashboard = await getMLRunsDashboard(10, false);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load ML run diagnostics';
			if (timer) {
				clearInterval(timer);
				timer = null;
			}
		} finally {
			loading = false;
		}
	}

	function statusClass(run: MLRunSummary): string {
		if (run.status === 'in_progress') return 'status-live';
		if (run.best_exists) return 'status-best';
		return 'status-idle';
	}

	onMount(() => {
		refresh();
		timer = setInterval(refresh, POLL_MS);
		return () => {
			if (timer) clearInterval(timer);
		};
	});
</script>

<div class="ml-panel card">
	<div class="ml-header">
		<h4>ML Runs</h4>
		<div class="ml-actions">
			<button class="secondary" on:click={() => expanded = !expanded}>
				{expanded ? 'Hide' : 'Show'}
			</button>
			<button class="primary" on:click={refresh} disabled={loading}>
				{loading ? 'Refreshing...' : 'Refresh'}
			</button>
		</div>
	</div>

	{#if error}
		<div class="ml-error">{error}</div>
	{/if}

	{#if expanded}
		{#if dashboard}
			<div class="ml-meta">
				<span>Active runs: {dashboard.active_runs.length}</span>
				<span>Tracked runs: {dashboard.runs.length}</span>
			</div>
			{#if dashboard.runs.length === 0}
				<div class="ml-empty">No auto-improve runs found in `reports/ml`.</div>
			{:else}
				<div class="ml-runs">
					{#each dashboard.runs as run}
						<div class="ml-run">
							<div class="ml-run-top">
								<div class="ml-name">{run.run_name}</div>
								<span class={`ml-status ${statusClass(run)}`}>{run.status}</span>
							</div>
							<div class="ml-row">
								<span>Stage: {run.current_stage}</span>
								<span>Iters: {run.iterations_completed}/{run.iterations_detected}</span>
								<span>Best: {run.best_exists ? 'yes' : 'no'}</span>
							</div>
							{#if run.latest_iteration}
								<div class="ml-row">
									<span>{run.latest_iteration.iteration}</span>
									<span>Samples: {run.latest_iteration.samples}</span>
									<span>Mean: {run.latest_iteration.mean_player_score.toFixed(1)}</span>
								</div>
								<div class="ml-row">
									<span>Eval: {run.latest_iteration.eval.nn_wins}/{run.latest_iteration.eval.games}</span>
									<span>Margin: {run.latest_iteration.eval.nn_mean_margin.toFixed(2)}</span>
									<span>Strict mix: {(run.latest_iteration.strict_game_fraction * 100).toFixed(0)}%</span>
								</div>
							{/if}
						</div>
					{/each}
				</div>
			{/if}
		{:else}
			<div class="ml-empty">Loading run diagnostics...</div>
		{/if}
	{/if}
</div>

<style>
	.ml-panel {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	.ml-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.ml-header h4 {
		margin: 0;
		font-size: 1rem;
	}

	.ml-actions {
		display: flex;
		gap: 0.5rem;
	}

	.ml-actions button {
		font-size: 0.78rem;
		padding: 0.35rem 0.55rem;
	}

	.ml-meta, .ml-row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.8rem;
		font-size: 0.78rem;
		color: #444;
	}

	.ml-runs {
		display: grid;
		gap: 0.55rem;
	}

	.ml-run {
		border: 1px solid #ddd;
		border-radius: 8px;
		padding: 0.55rem 0.65rem;
		background: #fafafa;
		display: grid;
		gap: 0.3rem;
	}

	.ml-run-top {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.5rem;
	}

	.ml-name {
		font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
		font-size: 0.76rem;
		overflow-wrap: anywhere;
	}

	.ml-status {
		font-size: 0.68rem;
		font-weight: 700;
		padding: 0.15rem 0.4rem;
		border-radius: 999px;
	}

	.status-live {
		background: #ddf8e3;
		color: #0f6a2f;
	}

	.status-best {
		background: #fff3cf;
		color: #8a5b00;
	}

	.status-idle {
		background: #eef1f5;
		color: #4f5f70;
	}

	.ml-empty {
		font-size: 0.82rem;
		color: #666;
	}

	.ml-error {
		font-size: 0.82rem;
		color: #9c1d1d;
		background: #ffe5e5;
		border: 1px solid #ffc4c4;
		padding: 0.45rem 0.55rem;
		border-radius: 6px;
	}

	@media (max-width: 820px) {
		.ml-actions {
			flex-wrap: wrap;
		}
	}
</style>
