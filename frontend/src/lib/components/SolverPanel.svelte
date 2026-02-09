<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { solveHeuristic, solveAfterReset, searchBirds } from '$lib/api/client';
	import type { SolverRecommendation, AfterResetRecommendation } from '$lib/api/types';
	import { FOOD_ICONS } from '$lib/api/types';

	const dispatch = createEventDispatcher();

	export let gameId: string;
	export let disabled = false;
	export let playerIdx: number = 0;
	export let playerName: string = '';

	let appliedRank: number | null = null;

	let recommendations: SolverRecommendation[] = [];
	let evaluationTime = 0;
	let loading = false;
	let error = '';
	let feederRerollAvailable = false;
	let solvedForPlayer = '';

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
			const data = await solveHeuristic(gameId, playerIdx);
			recommendations = data.recommendations;
			evaluationTime = data.evaluation_time_ms;
			feederRerollAvailable = data.feeder_reroll_available || false;
			solvedForPlayer = data.player_name || playerName;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to get recommendations';
			recommendations = [];
		} finally {
			loading = false;
		}
	}

	// Clear recommendations when player tab changes
	$: if (playerName !== solvedForPlayer && recommendations.length > 0) {
		recommendations = [];
		solvedForPlayer = '';
		showResetInput = false;
		afterResetRecs = [];
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

	const detailsOf = (rec: SolverRecommendation) => rec.details as Record<string, any>;
	const hasPlan = (rec: SolverRecommendation) => Array.isArray(detailsOf(rec)?.plan);
	const getPlan = (rec: SolverRecommendation) => detailsOf(rec)?.plan as string[] | undefined;
	const hasPlanDetails = (rec: SolverRecommendation) => Array.isArray(detailsOf(rec)?.plan_details);
	const getPlanDetails = (rec: SolverRecommendation) =>
		detailsOf(rec)?.plan_details as { description: string; delta?: any; power?: string[]; goal?: any }[] | undefined;
	const hasActivationAdvice = (rec: SolverRecommendation) =>
		Array.isArray(detailsOf(rec)?.activation_advice);
	const getActivationAdvice = (rec: SolverRecommendation) =>
		detailsOf(rec)?.activation_advice as string[] | undefined;
	const hasWhy = (rec: SolverRecommendation) => Array.isArray(detailsOf(rec)?.why);
	const getWhy = (rec: SolverRecommendation) => detailsOf(rec)?.why as string[] | undefined;
	const getBreakdown = (rec: SolverRecommendation) =>
		detailsOf(rec)?.breakdown as Record<string, number> | undefined;

	type BreakdownLine = { label: string; value: number; isNegative: boolean; tooltip?: string };
	let expandedBreakdownRanks = new Set<number>();

	function formatBreakdown(
		bd: Record<string, number> | undefined,
		expanded = false
	): BreakdownLine[] {
		if (!bd) return [];
		const order = [
			'bird_vp',
			'egg_capacity',
			'power_engine',
			'bonus_cards',
			'round_goal',
			'food_gain',
			'bird_unlocks',
			'engine_activation',
			'eggs',
			'cards',
			'tray_value',
			'deck_value',
			'nectar_value',
			'costs',
			'total_estimate'
		];
		const labels: Record<string, string> = {
			bird_vp: 'Bird VP',
			egg_capacity: 'Egg capacity',
			power_engine: 'Engine value',
			bonus_cards: 'Bonus card value',
			round_goal: 'Goal swing',
			food_gain: 'Food gained',
			bird_unlocks: 'Bird unlocks',
			engine_activation: 'Row activation',
			eggs: 'Egg points',
			cards: 'Card value',
			tray_value: 'Tray value',
			deck_value: 'Deck value',
			nectar_value: 'Nectar value',
			costs: 'Costs',
			total_estimate: 'Total estimate'
		};
		const tooltips: Record<string, string> = {
			bird_vp: 'Printed bird points plus immediate bird value.',
			egg_capacity: 'Value from added egg slots you can fill later.',
			power_engine: 'Estimated value from brown-power engines.',
			bonus_cards: 'Estimated alignment with active bonus cards.',
			round_goal: 'Estimated swing toward round-goal scoring tiers.',
			food_gain: 'Expected food gained from the move/activation.',
			bird_unlocks: 'Value from unlocking additional bird plays.',
			engine_activation: 'Value from activating a row this turn.',
			eggs: 'Immediate egg point value.',
			cards: 'Card draw/tuck value converted to points.',
			tray_value: 'Value of picking cards from the tray.',
			deck_value: 'Value of drawing unknown cards from the deck.',
			nectar_value: 'Nectar majority/scoring value.',
			costs: 'Penalty for food/tempo/nectar spent to execute.',
			total_estimate: 'Sum of the heuristic components shown.'
		};
		const alwaysInclude = new Set(['total_estimate', 'costs']);
		const entries = Object.entries(bd)
			.filter(([k, v]) => v !== 0 && (alwaysInclude.has(k) || Math.abs(v) >= 0.2));

		let ordered: [string, number][];
		if (expanded) {
			ordered = [...entries].sort((a, b) => {
				const ai = order.indexOf(a[0]);
				const bi = order.indexOf(b[0]);
				return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
			});
		} else {
			// Pull total first, then top 4 by absolute contribution
			const total = entries.find(([k]) => k === 'total_estimate');
			const rest = entries.filter(([k]) => k !== 'total_estimate');
			rest.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
			const top = rest.slice(0, 4);

			ordered = [];
			if (total) ordered.push(total);
			ordered.push(...top);
			// Ensure costs are visible even if small
			if (!ordered.find(([k]) => k === 'costs')) {
				const cost = entries.find(([k]) => k === 'costs');
				if (cost) ordered.push(cost);
			}
		}

		return ordered.map(([k, v]) => ({
			label: labels[k] || k,
			value: v,
			isNegative: v < 0,
			tooltip: tooltips[k]
		}));
	}

	function toggleBreakdown(rank: number) {
		const next = new Set(expandedBreakdownRanks);
		if (next.has(rank)) {
			next.delete(rank);
		} else {
			next.add(rank);
		}
		expandedBreakdownRanks = next;
	}

	function formatDelta(delta: any): string {
		if (!delta) return '';
		const parts: string[] = [];
		if (delta.food) {
			for (const [ft, v] of Object.entries(delta.food)) {
				const n = Number(v);
				if (n !== 0) parts.push(`${n > 0 ? '+' : ''}${n} ${ft}`);
			}
		}
		if (delta.eggs) parts.push(`${delta.eggs > 0 ? '+' : ''}${delta.eggs} eggs`);
		if (delta.cards) parts.push(`${delta.cards > 0 ? '+' : ''}${delta.cards} cards`);
		if (delta.cached) parts.push(`${delta.cached > 0 ? '+' : ''}${delta.cached} cached`);
		if (delta.tucked) parts.push(`${delta.tucked > 0 ? '+' : ''}${delta.tucked} tucked`);
		if (delta.nectar_spent) {
			for (const [hab, v] of Object.entries(delta.nectar_spent)) {
				const n = Number(v);
				if (n !== 0) parts.push(`${n > 0 ? '+' : ''}${n} nectar in ${hab}`);
			}
		}
		return parts.join(', ');
	}

	function applyRecommendation(rec: SolverRecommendation) {
		if (disabled) return;
		dispatch('apply', rec);
		appliedRank = rec.rank;
		setTimeout(() => { appliedRank = null; }, 2000);
	}

	function onRecKeydown(e: KeyboardEvent, rec: SolverRecommendation) {
		if (e.key === 'Enter' || e.key === ' ') {
			e.preventDefault();
			applyRecommendation(rec);
		}
	}
</script>

<div class="solver-panel card" class:disabled={disabled} aria-disabled={disabled}>
	<div class="panel-header">
		<h3>Recommendations{playerName ? ` for ${playerName}` : ''}</h3>
	</div>

	{#if error}
		<div class="error">{error}</div>
	{/if}

	{#if recommendations.length > 0}
		<div class="timing">
			Best moves for <strong>{solvedForPlayer}</strong> &middot; {evaluationTime.toFixed(1)}ms
		</div>

		<div class="recommendations">
			{#each recommendations as rec}
				<div
					class="rec clickable"
					class:top-pick={rec.rank === 1}
					class:applied={appliedRank === rec.rank}
					role="button"
					tabindex={disabled ? -1 : 0}
					on:click={() => applyRecommendation(rec)}
					on:keydown={(e) => onRecKeydown(e, rec)}
					title="Click to apply this move"
				>
					<div class="rec-header">
						<span class="rank">#{rec.rank}</span>
						<span class="action-icon">{ACTION_ICONS[rec.action_type] || '?'}</span>
						<span class="action-type">{ACTION_LABELS[rec.action_type] || rec.action_type}</span>
						{#if appliedRank === rec.rank}
							<span class="applied-badge">Applied!</span>
						{/if}
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
								{:else if part.toLowerCase().startsWith('activate ')}
									<span class="activate-hint">{part}</span>
								{:else}
									<span>{part}</span>
								{/if}
							{/each}
						</div>
					{/if}
					{#if hasWhy(rec)}
						<div class="rec-why">
							<span class="label">Why:</span>
							{#each getWhy(rec) || [] as line, li}
								{#if li > 0}<span class="reason-sep"> · </span>{/if}
								<span>{line}</span>
							{/each}
						</div>
					{/if}
					{#if getBreakdown(rec)}
						<div class="rec-breakdown">
							<span class="label">Breakdown:</span>
							{#each formatBreakdown(getBreakdown(rec), expandedBreakdownRanks.has(rec.rank)) as line, li}
								{#if li > 0}<span class="reason-sep"> · </span>{/if}
							<span class:negative={line.isNegative} title={line.tooltip || ''}>
								{line.label}: {line.value.toFixed(1)}
							</span>
							{/each}
							<button
								class="toggle-breakdown"
								type="button"
								on:click|stopPropagation={() => toggleBreakdown(rec.rank)}
							>
								{expandedBreakdownRanks.has(rec.rank) ? 'Show less' : 'Show full'}
							</button>
						</div>
					{/if}
					{#if hasPlan(rec)}
						<div class="rec-plan">
							<span class="label">Plan:</span>
							<span class="plan-steps">
								{#each getPlan(rec) || [] as step, si}
									{#if si > 0}<span class="reason-sep"> → </span>{/if}
									<span>{step}</span>
								{/each}
							</span>
						</div>
					{/if}
					{#if hasPlanDetails(rec)}
						<div class="rec-plan-details">
							{#each getPlanDetails(rec) || [] as step, si}
								<div class="plan-detail">
									<span class="step-index">{si + 1}.</span>
									<span class="step-desc">{step.description}</span>
									{#if step.delta && formatDelta(step.delta)}
										<span class="step-delta">({formatDelta(step.delta)})</span>
									{/if}
									{#if step.goal}
										<span class="step-goal">
											Goal gap {step.goal.gap_before} → {step.goal.gap_after}
										</span>
									{/if}
									{#if Array.isArray(step.power)}
										<div class="step-power">
											{#each step.power as p, pi}
												{#if pi > 0}<span class="reason-sep"> · </span>{/if}
												<span>{p}</span>
											{/each}
										</div>
									{/if}
								</div>
							{/each}
						</div>
					{/if}
					{#if hasActivationAdvice(rec)}
						<div class="rec-activation">
							<span class="label">Activation:</span>
							{#each getActivationAdvice(rec) || [] as line, li}
								{#if li > 0}<span class="reason-sep"> · </span>{/if}
								<span>{line}</span>
							{/each}
						</div>
					{/if}
					{#if hasResetOption(rec)}
						<button
							class="reset-btn"
							on:click|stopPropagation={() => startResetInput(rec)}
						>
							After Reset: Input New {hasResetOption(rec) === 'feeder' ? 'Dice' : 'Cards'}
						</button>
					{/if}
				</div>
			{/each}
		</div>
	{:else if !loading && !error}
		<div class="empty">Click "Get Recommendations" to analyze {playerName ? playerName + "'s" : 'the current'} position.</div>
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

	.rec.clickable {
		cursor: pointer;
		transition: background 0.15s, border-color 0.15s, box-shadow 0.15s;
	}

	.rec.clickable:hover {
		border-color: var(--accent);
		box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
		background: #fef9f0;
	}

	.rec.applied {
		border-color: #16a34a;
		background: #f0fdf4;
	}

	.applied-badge {
		font-size: 0.7rem;
		font-weight: 600;
		color: #16a34a;
		background: #dcfce7;
		padding: 1px 6px;
		border-radius: 4px;
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

	.rec-plan,
	.rec-activation {
		font-size: 0.7rem;
		color: var(--text-muted);
		margin-top: 4px;
		padding-left: 32px;
	}

	.rec-why {
		font-size: 0.7rem;
		color: var(--text-muted);
		margin-top: 4px;
		padding-left: 32px;
	}

	.rec-breakdown {
		font-size: 0.7rem;
		color: var(--text-muted);
		margin-top: 4px;
		padding-left: 32px;
	}

	.rec-breakdown .negative {
		color: #b91c1c;
	}

	.toggle-breakdown {
		margin-left: 8px;
		font-size: 0.7rem;
		background: transparent;
		border: none;
		color: var(--accent);
		cursor: pointer;
		padding: 0;
	}

	.rec-plan-details {
		font-size: 0.7rem;
		color: var(--text-muted);
		margin-top: 6px;
		padding-left: 32px;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.plan-detail {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: baseline;
	}

	.step-index {
		font-weight: 600;
		color: var(--accent);
	}

	.step-desc {
		font-weight: 500;
		color: var(--text);
	}

	.step-delta,
	.step-goal {
		color: var(--text-muted);
	}

	.step-power {
		font-style: italic;
		color: var(--accent);
	}

	.rec-plan .label,
	.rec-activation .label {
		font-weight: 600;
		color: var(--accent);
		margin-right: 6px;
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
