<script lang="ts">
	import { searchBirds, getBonusCards, getGoals, analyzeSetup } from '$lib/api/client';
	import { FOOD_ICONS } from '$lib/api/types';
	import type { Bird, BonusCard, Goal, SetupRecommendation } from '$lib/api/types';
	import BirdSearch from './BirdSearch.svelte';
	import { createEventDispatcher } from 'svelte';

	const dispatch = createEventDispatcher();

	// Input state
	let selectedBirds: Bird[] = [];
	let bonusCards: BonusCard[] = [];
	let allGoals: Goal[] = [];
	let selectedBonusCards: BonusCard[] = [];
	let selectedGoals: Goal[] = [];

	// Results
	let recommendations: SetupRecommendation[] = [];
	let totalCombinations = 0;
	let loading = false;
	let error = '';
	let hasAnalyzed = false;

	// Search state for bonus cards and goals
	let bonusSearchQuery = '';
	let goalSearchQuery = '';
	let showBonusDropdown = false;
	let showGoalDropdown = false;

	// Load bonus cards and goals on mount
	async function loadData() {
		try {
			const [bonusData, goalData] = await Promise.all([
				getBonusCards(),
				getGoals()
			]);
			bonusCards = bonusData.bonus_cards;
			allGoals = goalData.goals;
		} catch (e) {
			error = 'Failed to load game data';
		}
	}
	loadData();

	function handleBirdSelect(e: CustomEvent<Bird>) {
		if (selectedBirds.length >= 5) return;
		if (selectedBirds.find(b => b.name === e.detail.name)) return;
		selectedBirds = [...selectedBirds, e.detail];
	}

	function removeBird(i: number) {
		selectedBirds = selectedBirds.filter((_, idx) => idx !== i);
		hasAnalyzed = false;
	}

	$: filteredBonus = bonusSearchQuery.length > 0
		? bonusCards.filter(bc =>
			bc.name.toLowerCase().includes(bonusSearchQuery.toLowerCase()) ||
			bc.condition_text.toLowerCase().includes(bonusSearchQuery.toLowerCase())
		)
		: bonusCards;

	$: filteredGoals = goalSearchQuery.length > 0
		? allGoals.filter(g =>
			g.description.toLowerCase().includes(goalSearchQuery.toLowerCase())
		)
		: allGoals;

	function selectBonusCard(bc: BonusCard) {
		if (selectedBonusCards.length >= 2) return;
		if (selectedBonusCards.find(b => b.name === bc.name)) return;
		selectedBonusCards = [...selectedBonusCards, bc];
		bonusSearchQuery = '';
		showBonusDropdown = false;
		hasAnalyzed = false;
	}

	function removeBonusCard(i: number) {
		selectedBonusCards = selectedBonusCards.filter((_, idx) => idx !== i);
		hasAnalyzed = false;
	}

	function selectGoal(g: Goal) {
		if (selectedGoals.length >= 4) return;
		if (selectedGoals.find(sg => sg.description === g.description)) return;
		selectedGoals = [...selectedGoals, g];
		goalSearchQuery = '';
		showGoalDropdown = false;
		hasAnalyzed = false;
	}

	function removeGoal(i: number) {
		selectedGoals = selectedGoals.filter((_, idx) => idx !== i);
		hasAnalyzed = false;
	}

	async function analyze() {
		if (selectedBirds.length === 0 || selectedBonusCards.length === 0) {
			error = 'Add at least 1 bird and 1 bonus card';
			return;
		}
		loading = true;
		error = '';
		try {
			const data = await analyzeSetup(
				selectedBirds.map(b => b.name),
				selectedBonusCards.map(bc => bc.name),
				selectedGoals.map(g => g.description)
			);
			recommendations = data.recommendations;
			totalCombinations = data.total_combinations;
			hasAnalyzed = true;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Analysis failed';
		} finally {
			loading = false;
		}
	}

	function foodDisplay(bird: Bird): string {
		if (bird.food_cost.total === 0) return 'Free';
		return bird.food_cost.items.map(f => FOOD_ICONS[f] || f).join(
			bird.food_cost.is_or ? '/' : ' '
		);
	}

	function foodKeptDisplay(food: Record<string, number>): string {
		return Object.entries(food)
			.map(([type, count]) => `${FOOD_ICONS[type] || type} x${count}`)
			.join('  ');
	}
</script>

<div class="setup-advisor card">
	<h2>Setup Draft Advisor</h2>
	<p class="subtitle">Enter your dealt cards to find the best starting combination.</p>

	{#if error}
		<div class="error">{error}</div>
	{/if}

	<!-- Bird Selection -->
	<div class="section">
		<h3>Birds Dealt ({selectedBirds.length}/5)</h3>
		<BirdSearch
			placeholder="Search for a dealt bird..."
			disabled={selectedBirds.length >= 5}
			on:select={handleBirdSelect}
		/>
		{#if selectedBirds.length > 0}
			<div class="selected-list">
				{#each selectedBirds as bird, i}
					<div class="selected-item bird-item">
						<div class="item-info">
							<span class="item-name">{bird.name}</span>
							<span class="item-meta">
								{bird.victory_points}VP | {foodDisplay(bird)} | {bird.habitats.join(', ')}
							</span>
						</div>
						<button class="remove" on:click={() => removeBird(i)}>x</button>
					</div>
				{/each}
			</div>
		{/if}
	</div>

	<!-- Bonus Card Selection -->
	<div class="section">
		<h3>Bonus Cards Dealt ({selectedBonusCards.length}/2)</h3>
		<div class="search-container">
			<input
				type="text"
				bind:value={bonusSearchQuery}
				on:focus={() => showBonusDropdown = true}
				on:blur={() => setTimeout(() => showBonusDropdown = false, 200)}
				placeholder="Search bonus cards..."
				disabled={selectedBonusCards.length >= 2}
			/>
			{#if showBonusDropdown && filteredBonus.length > 0}
				<ul class="dropdown">
					{#each filteredBonus.slice(0, 15) as bc}
						<li on:mousedown|preventDefault={() => selectBonusCard(bc)}>
							<span class="item-name">{bc.name}</span>
							<span class="item-desc">{bc.condition_text}</span>
						</li>
					{/each}
				</ul>
			{/if}
		</div>
		{#if selectedBonusCards.length > 0}
			<div class="selected-list">
				{#each selectedBonusCards as bc, i}
					<div class="selected-item">
						<div class="item-info">
							<span class="item-name">{bc.name}</span>
							<span class="item-meta">{bc.condition_text}</span>
						</div>
						<button class="remove" on:click={() => removeBonusCard(i)}>x</button>
					</div>
				{/each}
			</div>
		{/if}
	</div>

	<!-- Round Goals (Optional) -->
	<div class="section">
		<h3>Round Goals ({selectedGoals.length}/4) <span class="optional">optional</span></h3>
		<div class="search-container">
			<input
				type="text"
				bind:value={goalSearchQuery}
				on:focus={() => showGoalDropdown = true}
				on:blur={() => setTimeout(() => showGoalDropdown = false, 200)}
				placeholder="Search round goals..."
				disabled={selectedGoals.length >= 4}
			/>
			{#if showGoalDropdown && filteredGoals.length > 0}
				<ul class="dropdown">
					{#each filteredGoals.slice(0, 15) as g}
						<li on:mousedown|preventDefault={() => selectGoal(g)}>
							<span class="item-name">{g.description}</span>
							<span class="item-desc">{g.game_set}</span>
						</li>
					{/each}
				</ul>
			{/if}
		</div>
		{#if selectedGoals.length > 0}
			<div class="selected-list">
				{#each selectedGoals as goal, i}
					<div class="selected-item">
						<div class="item-info">
							<span class="item-name">{goal.description}</span>
						</div>
						<button class="remove" on:click={() => removeGoal(i)}>x</button>
					</div>
				{/each}
			</div>
		{/if}
	</div>

	<!-- Analyze Button -->
	<div class="actions">
		<button class="primary" on:click={analyze} disabled={loading || selectedBirds.length === 0 || selectedBonusCards.length === 0}>
			{loading ? 'Analyzing...' : 'Analyze Draft'}
		</button>
	</div>

	<!-- Results -->
	{#if hasAnalyzed && recommendations.length > 0}
		<div class="results">
			<h3>Recommendations <span class="combo-count">({totalCombinations} combinations evaluated)</span></h3>
			{#each recommendations as rec}
				<div class="recommendation" class:top-pick={rec.rank === 1}>
					<div class="rec-header">
						<span class="rec-rank">#{rec.rank}</span>
						<span class="rec-score">Score: {rec.score}</span>
					</div>

					<div class="rec-body">
						<div class="rec-section">
							<span class="rec-label">Keep birds:</span>
							{#if rec.birds_to_keep.length === 0}
								<span class="rec-value none">None (keep all food)</span>
							{:else}
								<span class="rec-value">{rec.birds_to_keep.join(', ')}</span>
							{/if}
						</div>

						<div class="rec-section">
							<span class="rec-label">Keep food:</span>
							{#if Object.keys(rec.food_to_keep).length === 0}
								<span class="rec-value none">None (keep all birds)</span>
							{:else}
								<span class="rec-value food-display">
									{foodKeptDisplay(rec.food_to_keep)}
								</span>
							{/if}
						</div>

						<div class="rec-section">
							<span class="rec-label">Bonus card:</span>
							<span class="rec-value">{rec.bonus_card}</span>
						</div>

						<div class="rec-reasoning">{rec.reasoning}</div>
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.setup-advisor {
		max-width: 700px;
		margin: 20px auto;
	}

	h2 {
		margin-bottom: 4px;
	}

	.subtitle {
		color: var(--text-muted);
		font-size: 0.9rem;
		margin-bottom: 16px;
	}

	.error {
		background: #fef2f2;
		color: #dc2626;
		padding: 8px 12px;
		border-radius: 6px;
		font-size: 0.85rem;
		margin-bottom: 12px;
	}

	.section {
		margin-bottom: 16px;
	}

	h3 {
		font-size: 0.95rem;
		margin-bottom: 8px;
		color: var(--text);
	}

	.optional {
		font-size: 0.75rem;
		color: var(--text-muted);
		font-weight: normal;
	}

	.search-container {
		position: relative;
	}

	.search-container input {
		width: 100%;
	}

	.dropdown {
		position: absolute;
		top: 100%;
		left: 0;
		right: 0;
		background: white;
		border: 1px solid var(--border);
		border-radius: 0 0 6px 6px;
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
		list-style: none;
		max-height: 250px;
		overflow-y: auto;
		z-index: 100;
	}

	.dropdown li {
		padding: 8px 12px;
		cursor: pointer;
		border-bottom: 1px solid #f0f0f0;
	}

	.dropdown li:hover {
		background: #f5f0e8;
	}

	.dropdown .item-name {
		display: block;
		font-weight: 500;
		font-size: 0.85rem;
	}

	.dropdown .item-desc {
		display: block;
		font-size: 0.75rem;
		color: var(--text-muted);
	}

	.selected-list {
		margin-top: 8px;
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.selected-item {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 6px 10px;
		background: #f8f6f2;
		border: 1px solid #e8e4de;
		border-radius: 6px;
		font-size: 0.85rem;
	}

	.item-info {
		display: flex;
		flex-direction: column;
		gap: 1px;
	}

	.item-name {
		font-weight: 500;
	}

	.item-meta {
		font-size: 0.75rem;
		color: var(--text-muted);
	}

	.remove {
		font-size: 0.75rem;
		padding: 2px 8px;
		border: 1px solid #ddd;
		background: #f9f9f9;
		color: #999;
		cursor: pointer;
		border-radius: 4px;
	}

	.remove:hover {
		background: #fee;
		color: #c00;
		border-color: #c00;
	}

	.actions {
		display: flex;
		justify-content: center;
		margin: 20px 0;
	}

	.actions .primary {
		padding: 10px 32px;
		font-size: 1rem;
	}

	/* Results */
	.results {
		margin-top: 16px;
	}

	.results h3 {
		margin-bottom: 12px;
	}

	.combo-count {
		font-size: 0.75rem;
		color: var(--text-muted);
		font-weight: normal;
	}

	.recommendation {
		border: 1px solid #e8e4de;
		border-radius: 8px;
		padding: 12px;
		margin-bottom: 10px;
		background: #fefcf9;
	}

	.recommendation.top-pick {
		border-color: var(--accent);
		background: #fdf8ef;
		box-shadow: 0 0 0 1px var(--accent);
	}

	.rec-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 8px;
	}

	.rec-rank {
		font-weight: 700;
		font-size: 1.1rem;
		color: var(--accent);
	}

	.top-pick .rec-rank {
		font-size: 1.2rem;
	}

	.rec-score {
		font-size: 0.8rem;
		color: var(--text-muted);
	}

	.rec-body {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.rec-section {
		display: flex;
		gap: 8px;
		font-size: 0.85rem;
	}

	.rec-label {
		color: var(--text-muted);
		min-width: 90px;
		flex-shrink: 0;
	}

	.rec-value {
		font-weight: 500;
	}

	.rec-value.none {
		color: var(--text-muted);
		font-style: italic;
		font-weight: normal;
	}

	.food-display {
		font-size: 0.95rem;
	}

	.rec-reasoning {
		margin-top: 6px;
		font-size: 0.8rem;
		color: var(--text-muted);
		font-style: italic;
		padding-top: 6px;
		border-top: 1px solid #eee;
	}
</style>
