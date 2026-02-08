<script lang="ts">
	import { solveHeuristic, solveAfterReset, searchBirds } from '$lib/api/client';
	import type { SolverRecommendation, AfterResetRecommendation } from '$lib/api/types';
	import { FOOD_ICONS } from '$lib/api/types';

	export let gameId: string;
	export let disabled = false;

	let recommendations: SolverRecommendation[] = [];
	let evaluationTime = 0;
	let loading = false;
	let error = '';
	let feederRerollAvailable = false;

	// After-reset flow state
	let showResetInput = false;
	let resetType: 'feeder' | 'tray' = 'feeder';
	let resetTotalToGain = 0;
	let resetLoading = false;
	let resetError = '';
	let afterResetRecs: AfterResetRecommendation[] = [];

	// Feeder dice input (5 dice)
	const OCEANIA_FOOD_TYPES = ['invertebrate', 'seed', 'fish', 'fruit', 'rodent', 'nectar'];
	let diceInputs: string[] = ['seed', 'seed', 'seed', 'seed', 'seed'];
	let diceIsChoice: boolean[] = [false, false, false, false, false];
	let diceChoice2: string[] = ['seed', 'seed', 'seed', 'seed', 'seed'];

	// Tray card input (3 cards)
	let trayInputs: string[] = ['', '', ''];
	let traySearchResults: { name: string }[][] = [[], [], []];
	let traySearching: boolean[] = [false, false, false];

	export async function solve() {
		loading = true;
		error = '';
		showResetInput = false;
		afterResetRecs = [];
		try {
			const data = await solveHeuristic(gameId);
			recommendations = data.recommendations;
			evaluationTime = data.evaluation_time_ms;
			feederRerollAvailable = data.feeder_reroll_available || false;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to get recommendations';
			recommendations = [];
		} finally {
			loading = false;
		}
	}

	function hasResetOption(rec: SolverRecommendation): 'feeder' | 'tray' | null {
		const desc = rec.description.toLowerCase();
		const reasoning = rec.reasoning.toLowerCase();
		if (desc.includes('reset feeder') || reasoning.includes('reset feeder')) return 'feeder';
		if (desc.includes('reset tray') || reasoning.includes('reset tray')) return 'tray';
		// Free reroll: all dice show same face, show on gain_food moves
		if (feederRerollAvailable && rec.action_type === 'gain_food') return 'feeder';
		return null;
	}

	function extractTotalToGain(rec: SolverRecommendation): number {
		const desc = rec.description;
		// Count food items: "Gain 1 seed + 1 invertebrate"
		let count = 0;
		const gainMatch = desc.match(/Gain\s+(.*?)(?:\s*\(|$)/);
		if (gainMatch) {
			const parts = gainMatch[1].split('+');
			for (const part of parts) {
				const numMatch = part.trim().match(/^(\d+)/);
				if (numMatch) count += parseInt(numMatch[1]);
			}
		}
		// For draw cards: "Draw N from deck" or "Take ... + N from deck"
		const drawMatch = desc.match(/Draw\s+(\d+)/);
		if (drawMatch) count = parseInt(drawMatch[1]);
		const takeMatch = desc.match(/\+\s*(\d+)\s*from deck/);
		if (takeMatch && !drawMatch) count = parseInt(takeMatch[1]) + 1;
		return count || 2;
	}

	function startResetInput(rec: SolverRecommendation) {
		const type = hasResetOption(rec);
		if (!type) return;
		resetType = type;
		resetTotalToGain = extractTotalToGain(rec);
		showResetInput = true;
		afterResetRecs = [];
		resetError = '';
	}

	async function submitResetInput() {
		resetLoading = true;
		resetError = '';
		try {
			if (resetType === 'feeder') {
				const dice: (string | string[])[] = diceInputs.map((d, i) =>
					diceIsChoice[i] ? [d, diceChoice2[i]] : d
				);
				const data = await solveAfterReset(gameId, 'feeder', dice, undefined, resetTotalToGain);
				afterResetRecs = data.recommendations;
			} else {
				const cards = trayInputs.filter(c => c.trim() !== '');
				if (cards.length === 0) {
					resetError = 'Enter at least one tray card';
					resetLoading = false;
					return;
				}
				const data = await solveAfterReset(gameId, 'tray', undefined, cards, resetTotalToGain);
				afterResetRecs = data.recommendations;
			}
		} catch (e) {
			resetError = e instanceof Error ? e.message : 'Failed to get after-reset recommendation';
			afterResetRecs = [];
		} finally {
			resetLoading = false;
		}
	}

	let searchTimeout: ReturnType<typeof setTimeout>;
	async function searchTrayBird(index: number) {
		const query = trayInputs[index].trim();
		if (query.length < 2) {
			traySearchResults[index] = [];
			traySearchResults = [...traySearchResults];
			return;
		}
		clearTimeout(searchTimeout);
		searchTimeout = setTimeout(async () => {
			traySearching[index] = true;
			traySearching = [...traySearching];
			try {
				const data = await searchBirds(query, 5);
				traySearchResults[index] = data.birds.map(b => ({ name: b.name }));
				traySearchResults = [...traySearchResults];
			} catch {
				traySearchResults[index] = [];
				traySearchResults = [...traySearchResults];
			} finally {
				traySearching[index] = false;
				traySearching = [...traySearching];
			}
		}, 300);
	}

	function selectTrayBird(index: number, name: string) {
		trayInputs[index] = name;
		traySearchResults[index] = [];
		traySearchResults = [...traySearchResults];
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
								{#if part.startsWith('SKIP ') || part.startsWith('WARNING')}
									<span class="skip-warning">{part}</span>
								{:else if part.startsWith('REROLL ')}
									<span class="reroll-hint">{part}</span>
								{:else if part.startsWith('FIRST ')}
									<span class="reset-order-hint">{part}</span>
								{:else if part.startsWith('activate ')}
									<span class="activate-hint">{part}</span>
								{:else}
									<span>{part}</span>
								{/if}
							{/each}
						</div>
					{/if}
					{#if hasResetOption(rec)}
						<button
							class="reset-btn"
							on:click={() => startResetInput(rec)}
						>
							After Reset: Input New {hasResetOption(rec) === 'feeder' ? 'Dice' : 'Cards'}
						</button>
					{/if}
				</div>
			{/each}
		</div>
	{:else if !loading && !error}
		<div class="empty">Click "Get Recommendations" to analyze the current position.</div>
	{/if}

	<!-- After-reset input panel -->
	{#if showResetInput}
		<div class="reset-panel">
			<div class="reset-header">
				<h4>
					{resetType === 'feeder' ? '🎲 New Feeder Dice' : '🃏 New Tray Cards'}
				</h4>
				<span class="reset-info">
					({resetTotalToGain} {resetType === 'feeder' ? 'food' : 'card'}{resetTotalToGain !== 1 ? 's' : ''} to gain)
				</span>
				<button class="close-btn" on:click={() => { showResetInput = false; afterResetRecs = []; }}>
					✕
				</button>
			</div>

			{#if resetType === 'feeder'}
				<div class="dice-inputs">
					{#each diceInputs as _, i}
						<div class="die-row">
							<span class="die-label">Die {i + 1}:</span>
							<select bind:value={diceInputs[i]}>
								{#each OCEANIA_FOOD_TYPES as ft}
									<option value={ft}>{FOOD_ICONS[ft] || ''} {ft}</option>
								{/each}
							</select>
							<label class="choice-label">
								<input type="checkbox" bind:checked={diceIsChoice[i]} />
								Choice
							</label>
							{#if diceIsChoice[i]}
								<span class="choice-or">/</span>
								<select bind:value={diceChoice2[i]}>
									{#each OCEANIA_FOOD_TYPES as ft}
										<option value={ft}>{FOOD_ICONS[ft] || ''} {ft}</option>
									{/each}
								</select>
							{/if}
						</div>
					{/each}
				</div>
			{:else}
				<div class="tray-inputs">
					{#each trayInputs as _, i}
						<div class="tray-row">
							<span class="tray-label">Card {i + 1}:</span>
							<div class="tray-search-wrapper">
								<input
									type="text"
									bind:value={trayInputs[i]}
									on:input={() => searchTrayBird(i)}
									placeholder="Search bird name..."
								/>
								{#if traySearchResults[i].length > 0}
									<div class="search-dropdown">
										{#each traySearchResults[i] as result}
											<button
												class="search-result"
												on:click={() => selectTrayBird(i, result.name)}
											>
												{result.name}
											</button>
										{/each}
									</div>
								{/if}
							</div>
						</div>
					{/each}
				</div>
			{/if}

			<button
				class="primary submit-reset"
				on:click={submitResetInput}
				disabled={resetLoading}
			>
				{resetLoading ? 'Evaluating...' : 'Get Recommendation'}
			</button>

			{#if resetError}
				<div class="error">{resetError}</div>
			{/if}

			{#if afterResetRecs.length > 0}
				<div class="after-reset-results">
					<h4>
						{resetType === 'feeder' ? '🍽️ Take This Food' : '🃏 Take This Card'}
					</h4>
					{#each afterResetRecs as rec}
						<div class="rec" class:top-pick={rec.rank === 1}>
							<div class="rec-header">
								<span class="rank">#{rec.rank}</span>
								<span class="rec-desc-inline">{rec.description}</span>
								<span class="score">{rec.score.toFixed(1)} pts</span>
							</div>
							{#if rec.reasoning}
								<div class="rec-reasoning">
									{#each rec.reasoning.split('; ') as part, pi}
										{#if pi > 0}<span class="reason-sep"> · </span>{/if}
										<span>{part}</span>
									{/each}
								</div>
							{/if}
						</div>
					{/each}
				</div>
			{/if}
		</div>
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

	.rec-desc-inline {
		font-size: 0.85rem;
		font-weight: 500;
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

	.reroll-hint {
		color: #0369a1;
		font-weight: 600;
	}

	.reset-order-hint {
		color: #7c3aed;
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

	/* Reset button on recommendations */
	.reset-btn {
		display: inline-block;
		margin-top: 6px;
		margin-left: 32px;
		padding: 4px 10px;
		font-size: 0.7rem;
		background: #e0f2fe;
		color: #0369a1;
		border: 1px solid #7dd3fc;
		border-radius: 4px;
		cursor: pointer;
		font-weight: 500;
	}

	.reset-btn:hover {
		background: #bae6fd;
	}

	/* Reset input panel */
	.reset-panel {
		margin-top: 12px;
		padding: 12px;
		border: 2px solid #7dd3fc;
		border-radius: 8px;
		background: #f0f9ff;
	}

	.reset-header {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 10px;
	}

	.reset-header h4 {
		margin: 0;
		font-size: 0.9rem;
	}

	.reset-info {
		font-size: 0.75rem;
		color: var(--text-muted);
	}

	.close-btn {
		margin-left: auto;
		background: none;
		border: none;
		cursor: pointer;
		font-size: 1rem;
		color: var(--text-muted);
		padding: 2px 6px;
	}

	.close-btn:hover {
		color: #dc2626;
	}

	/* Dice inputs */
	.dice-inputs {
		display: flex;
		flex-direction: column;
		gap: 6px;
		margin-bottom: 10px;
	}

	.die-row {
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.die-label {
		font-size: 0.8rem;
		font-weight: 500;
		min-width: 42px;
	}

	.die-row select {
		padding: 3px 6px;
		font-size: 0.8rem;
		border: 1px solid var(--border);
		border-radius: 4px;
	}

	.choice-label {
		font-size: 0.75rem;
		display: flex;
		align-items: center;
		gap: 3px;
		cursor: pointer;
	}

	.choice-label input {
		margin: 0;
	}

	.choice-or {
		font-weight: 600;
		color: var(--accent);
	}

	/* Tray inputs */
	.tray-inputs {
		display: flex;
		flex-direction: column;
		gap: 6px;
		margin-bottom: 10px;
	}

	.tray-row {
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.tray-label {
		font-size: 0.8rem;
		font-weight: 500;
		min-width: 52px;
	}

	.tray-search-wrapper {
		position: relative;
		flex: 1;
	}

	.tray-search-wrapper input {
		width: 100%;
		padding: 4px 8px;
		font-size: 0.8rem;
		border: 1px solid var(--border);
		border-radius: 4px;
		box-sizing: border-box;
	}

	.search-dropdown {
		position: absolute;
		top: 100%;
		left: 0;
		right: 0;
		background: white;
		border: 1px solid var(--border);
		border-top: none;
		border-radius: 0 0 4px 4px;
		z-index: 10;
		max-height: 120px;
		overflow-y: auto;
	}

	.search-result {
		display: block;
		width: 100%;
		text-align: left;
		padding: 4px 8px;
		font-size: 0.8rem;
		background: none;
		border: none;
		cursor: pointer;
	}

	.search-result:hover {
		background: #f0f9ff;
	}

	.submit-reset {
		width: 100%;
		margin-top: 4px;
	}

	/* After-reset results */
	.after-reset-results {
		margin-top: 10px;
		padding-top: 10px;
		border-top: 1px solid #7dd3fc;
	}

	.after-reset-results h4 {
		margin: 0 0 8px 0;
		font-size: 0.85rem;
	}
</style>
