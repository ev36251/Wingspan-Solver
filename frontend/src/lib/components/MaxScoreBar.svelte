<script lang="ts">
	import { getMaxScore } from '$lib/api/client';
	import type { MaxScoreResponse } from '$lib/api/types';

	export let gameId: string;
	export let playerName: string;
	export let triggerRefresh: number = 0;

	let data: MaxScoreResponse | null = null;
	let loading = false;
	let error = '';
	let showBreakdown = false;

	async function fetchMaxScore() {
		if (!gameId) return;
		loading = true;
		error = '';
		try {
			data = await getMaxScore(gameId, playerName);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed';
			data = null;
		} finally {
			loading = false;
		}
	}

	$: if (triggerRefresh > 0 && gameId) {
		fetchMaxScore();
	}

	$: if (playerName && gameId && triggerRefresh > 0) {
		fetchMaxScore();
	}

	function efficiencyColor(pct: number): string {
		if (pct >= 60) return '#16a34a';
		if (pct >= 30) return '#ca8a04';
		return '#dc2626';
	}

	const LABELS: Record<string, string> = {
		bird_vp: 'Bird VP',
		eggs: 'Eggs',
		cached_food: 'Cached',
		tucked_cards: 'Tucked',
		bonus_cards: 'Bonus',
		round_goals: 'Goals',
		nectar: 'Nectar',
	};
</script>

<div class="max-score-bar">
	{#if loading}
		<span class="loading-text">...</span>
	{:else if data}
		<div class="score-readout">
			<span class="current">{data.current_score}</span>
			<span class="sep">/</span>
			<span class="max">{data.max_possible_score}</span>
			<span class="label">pts</span>
		</div>
		<span class="eff-badge" style="background: {efficiencyColor(data.efficiency_pct)}">
			{data.efficiency_pct.toFixed(0)}%
		</span>
		<button class="detail-btn" on:click={() => showBreakdown = !showBreakdown}>
			{showBreakdown ? 'Hide' : 'Details'}
		</button>
		<button class="refresh-btn" on:click={fetchMaxScore} title="Refresh score estimate">
			Refresh
		</button>
	{:else if error}
		<span class="error-text">{error}</span>
	{:else}
		<span class="empty-text">Save to see score estimate</span>
	{/if}

	{#if showBreakdown && data}
		<div class="breakdown">
			{#each Object.entries(data.breakdown) as [key, value]}
				{#if key !== 'total' && LABELS[key]}
					<div class="bk-row">
						<span class="bk-label">{LABELS[key]}</span>
						<span class="bk-val">{value}</span>
					</div>
				{/if}
			{/each}
		</div>
	{/if}
</div>

<style>
	.max-score-bar {
		display: flex;
		align-items: center;
		gap: 8px;
		position: relative;
		flex-wrap: wrap;
	}

	.score-readout {
		display: flex;
		align-items: baseline;
		gap: 2px;
	}

	.current {
		font-size: 1.1rem;
		font-weight: 800;
		color: var(--text);
	}

	.sep {
		font-size: 0.9rem;
		color: var(--text-muted);
	}

	.max {
		font-size: 1.1rem;
		font-weight: 700;
		color: var(--text-muted);
	}

	.label {
		font-size: 0.7rem;
		color: var(--text-muted);
		margin-left: 2px;
	}

	.eff-badge {
		color: white;
		font-size: 0.75rem;
		font-weight: 700;
		padding: 2px 8px;
		border-radius: 10px;
		line-height: 1.2;
	}

	.detail-btn, .refresh-btn {
		font-size: 0.7rem;
		padding: 2px 8px;
		border: 1px solid var(--border);
		background: #f5f5f5;
		border-radius: 4px;
		cursor: pointer;
		color: var(--text-muted);
	}

	.detail-btn:hover, .refresh-btn:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	.loading-text, .empty-text {
		font-size: 0.75rem;
		color: var(--text-muted);
		font-style: italic;
	}

	.error-text {
		font-size: 0.75rem;
		color: #dc2626;
	}

	.breakdown {
		position: absolute;
		top: 100%;
		left: 0;
		background: white;
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 8px 12px;
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
		z-index: 50;
		min-width: 160px;
		margin-top: 4px;
	}

	.bk-row {
		display: flex;
		justify-content: space-between;
		padding: 2px 0;
		font-size: 0.75rem;
	}

	.bk-label {
		color: var(--text-muted);
	}

	.bk-val {
		font-weight: 700;
		color: var(--text);
	}
</style>
