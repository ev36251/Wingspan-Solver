<script lang="ts">
	import { getScores } from '$lib/api/client';
	import type { ScoreBreakdown } from '$lib/api/types';

	export let gameId: string;

	let scores: Record<string, ScoreBreakdown> = {};
	let loading = false;
	let error = '';

	export async function refresh() {
		loading = true;
		error = '';
		try {
			const data = await getScores(gameId);
			scores = data.scores;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load scores';
		} finally {
			loading = false;
		}
	}

	const CATEGORIES = [
		{ key: 'bird_vp', label: 'Bird VP' },
		{ key: 'eggs', label: 'Eggs' },
		{ key: 'cached_food', label: 'Cached Food' },
		{ key: 'tucked_cards', label: 'Tucked Cards' },
		{ key: 'bonus_cards', label: 'Bonus Cards' },
		{ key: 'round_goals', label: 'Round Goals' },
		{ key: 'nectar', label: 'Nectar' },
	] as const;

	$: playerNames = Object.keys(scores);
</script>

<div class="score-sheet card">
	<div class="sheet-header">
		<h3>Score Breakdown</h3>
		<button on:click={refresh} disabled={loading}>
			{loading ? 'Loading...' : 'Refresh'}
		</button>
	</div>

	{#if error}
		<div class="error">{error}</div>
	{/if}

	{#if playerNames.length > 0}
		<table>
			<thead>
				<tr>
					<th>Category</th>
					{#each playerNames as name}
						<th>{name}</th>
					{/each}
				</tr>
			</thead>
			<tbody>
				{#each CATEGORIES as cat}
					<tr>
						<td>{cat.label}</td>
						{#each playerNames as name}
							<td class="score-cell">{scores[name][cat.key]}</td>
						{/each}
					</tr>
				{/each}
				<tr class="total-row">
					<td>Total</td>
					{#each playerNames as name}
						<td class="score-cell total">{scores[name].total}</td>
					{/each}
				</tr>
			</tbody>
		</table>
	{:else if !loading}
		<div class="empty">No scores to display yet.</div>
	{/if}
</div>

<style>
	.sheet-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 12px;
	}

	.error {
		background: #fef2f2;
		color: #dc2626;
		padding: 8px 12px;
		border-radius: 6px;
		font-size: 0.85rem;
		margin-bottom: 8px;
	}

	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.85rem;
	}

	th {
		text-align: left;
		padding: 6px 10px;
		border-bottom: 2px solid var(--border);
		font-weight: 600;
		color: var(--text);
	}

	th:not(:first-child) {
		text-align: center;
	}

	td {
		padding: 5px 10px;
		border-bottom: 1px solid #f0ece4;
	}

	.score-cell {
		text-align: center;
		font-variant-numeric: tabular-nums;
	}

	.total-row td {
		border-top: 2px solid var(--border);
		font-weight: 700;
	}

	.total {
		color: var(--accent);
		font-size: 1rem;
	}

	.empty {
		font-size: 0.85rem;
		color: var(--text-muted);
		text-align: center;
		padding: 16px;
	}
</style>
