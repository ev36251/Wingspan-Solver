<script lang="ts">
	import { solveHeuristic } from '$lib/api/client';
	import type { SolverRecommendation } from '$lib/api/types';

	export let gameId: string;
	export let disabled = false;

	let recommendations: SolverRecommendation[] = [];
	let evaluationTime = 0;
	let loading = false;
	let error = '';

	async function solve() {
		loading = true;
		error = '';
		try {
			const data = await solveHeuristic(gameId);
			recommendations = data.recommendations;
			evaluationTime = data.evaluation_time_ms;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to get recommendations';
			recommendations = [];
		} finally {
			loading = false;
		}
	}

	const ACTION_ICONS: Record<string, string> = {
		play_bird: '🐦',
		gain_food: '🍽️',
		lay_eggs: '🥚',
		draw_cards: '🃏'
	};

	const ACTION_LABELS: Record<string, string> = {
		play_bird: 'Play Bird',
		gain_food: 'Gain Food',
		lay_eggs: 'Lay Eggs',
		draw_cards: 'Draw Cards'
	};
</script>

<div class="solver-panel card">
	<div class="panel-header">
		<h3>Solver Recommendations</h3>
		<button class="primary" on:click={solve} disabled={disabled || loading}>
			{loading ? 'Thinking...' : 'Get Recommendations'}
		</button>
	</div>

	{#if error}
		<div class="error">{error}</div>
	{/if}

	{#if recommendations.length > 0}
		<div class="timing">Evaluated in {evaluationTime.toFixed(1)}ms</div>

		<div class="recommendations">
			{#each recommendations as rec}
				<div class="rec" class:top-pick={rec.rank === 1}>
					<div class="rec-header">
						<span class="rank">#{rec.rank}</span>
						<span class="action-icon">{ACTION_ICONS[rec.action_type] || '?'}</span>
						<span class="action-type">{ACTION_LABELS[rec.action_type] || rec.action_type}</span>
						<span class="score">{rec.score.toFixed(1)} pts</span>
					</div>
					<div class="rec-desc">{rec.description}</div>
					{#if rec.reasoning}
						<div class="rec-reasoning">
							{#each rec.reasoning.split('; ') as part, pi}
								{#if pi > 0}<span class="reason-sep"> · </span>{/if}
								{#if part.startsWith('SKIP ')}
									<span class="skip-warning">{part}</span>
								{:else if part.startsWith('activate ')}
									<span class="activate-hint">{part}</span>
								{:else}
									<span>{part}</span>
								{/if}
							{/each}
						</div>
					{/if}
				</div>
			{/each}
		</div>
	{:else if !loading && !error}
		<div class="empty">Click "Get Recommendations" to analyze the current position.</div>
	{/if}
</div>

<style>
	.solver-panel {
		border-color: var(--accent);
	}

	.panel-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 12px;
	}

	.timing {
		font-size: 0.75rem;
		color: var(--text-muted);
		margin-bottom: 8px;
	}

	.error {
		background: #fef2f2;
		color: #dc2626;
		padding: 8px 12px;
		border-radius: 6px;
		font-size: 0.85rem;
		margin-bottom: 8px;
	}

	.recommendations {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.rec {
		padding: 8px 12px;
		border: 1px solid var(--border);
		border-radius: 6px;
		background: #fefdf8;
	}

	.rec.top-pick {
		border-color: var(--accent);
		background: #fef9f0;
	}

	.rec-header {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.rank {
		font-weight: 700;
		font-size: 0.85rem;
		color: var(--accent);
		min-width: 24px;
	}

	.action-icon {
		font-size: 1rem;
	}

	.action-type {
		font-weight: 500;
		font-size: 0.85rem;
	}

	.score {
		margin-left: auto;
		font-weight: 600;
		font-size: 0.85rem;
		color: var(--accent);
	}

	.rec-desc {
		font-size: 0.75rem;
		color: var(--text-muted);
		margin-top: 4px;
		padding-left: 32px;
	}

	.rec-reasoning {
		font-size: 0.7rem;
		color: var(--accent);
		margin-top: 2px;
		padding-left: 32px;
		font-style: italic;
	}

	.skip-warning {
		color: #dc2626;
		font-weight: 600;
	}

	.activate-hint {
		color: #16a34a;
	}

	.reason-sep {
		color: var(--text-muted);
	}

	.empty {
		font-size: 0.85rem;
		color: var(--text-muted);
		text-align: center;
		padding: 16px;
	}
</style>
