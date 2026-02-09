<script lang="ts">
	import { searchBirds, getBonusCards, getGoals, analyzeSetup } from '$lib/api/client';
	import { FOOD_ICONS } from '$lib/api/types';
	import type { Bird, BonusCard, Goal, SetupRecommendation } from '$lib/api/types';
	import BirdSearch from './BirdSearch.svelte';
	import { createEventDispatcher } from 'svelte';

	const dispatch = createEventDispatcher();

	export let playerNames: string[] = [];

	// Input state
	let selectedBirds: Bird[] = [];
	let bonusCards: BonusCard[] = [];
	let allGoals: Goal[] = [];
	let selectedBonusCards: BonusCard[] = [];
	let goalSelections: string[] = ['No Goal', 'No Goal', 'No Goal', 'No Goal'];
	let playerCount = 2;
	let turnOrder = 1;
	let trayBirds: Bird[] = [];

	// Custom setup (manual selection)
	let customKeepBirds: Set<string> = new Set();
	let customKeepFood: Set<string> = new Set();
	let customBonusCard = '';

	// Results
	let recommendations: SetupRecommendation[] = [];
	let totalCombinations = 0;
	let loading = false;
	let error = '';
	let hasAnalyzed = false;

	// Search state for bonus cards
	let bonusSearchQuery = '';
	let showBonusDropdown = false;

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

	$: if (playerNames.length >= 2) {
		playerCount = playerNames.length;
	}
	$: if (turnOrder > playerCount) {
		turnOrder = playerCount;
	}

	function handleBirdSelect(e: CustomEvent<Bird>) {
		if (selectedBirds.length >= 5) return;
		if (selectedBirds.find(b => b.name === e.detail.name)) return;
		selectedBirds = [...selectedBirds, e.detail];
	}

	function removeBird(i: number) {
		const removed = selectedBirds[i];
		selectedBirds = selectedBirds.filter((_, idx) => idx !== i);
		hasAnalyzed = false;
		if (removed) {
			customKeepBirds.delete(removed.name);
			customKeepBirds = new Set(customKeepBirds);
		}
	}

	function addTrayBird(e: CustomEvent<Bird>) {
		if (trayBirds.length >= 3) return;
		if (trayBirds.find(b => b.name === e.detail.name)) return;
		trayBirds = [...trayBirds, e.detail];
	}

	function removeTrayBird(i: number) {
		trayBirds = trayBirds.filter((_, idx) => idx !== i);
	}

	$: filteredBonus = bonusSearchQuery.length > 0
		? bonusCards.filter(bc =>
			bc.name.toLowerCase().includes(bonusSearchQuery.toLowerCase()) ||
			bc.condition_text.toLowerCase().includes(bonusSearchQuery.toLowerCase())
		)
		: bonusCards;

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

	function setGoal(roundIdx: number, value: string) {
		goalSelections[roundIdx] = value;
		goalSelections = [...goalSelections];
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
			const roundGoals = goalSelections.map(g => g || 'No Goal');
			const data = await analyzeSetup(
				selectedBirds.map(b => b.name),
				selectedBonusCards.map(bc => bc.name),
				roundGoals,
				trayBirds.map(b => b.name),
				turnOrder,
				playerCount
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

	function toggleCustomBird(name: string) {
		if (customKeepBirds.has(name)) customKeepBirds.delete(name);
		else customKeepBirds.add(name);
		customKeepBirds = new Set(customKeepBirds);
	}

	function toggleCustomFood(type: string) {
		if (customKeepFood.has(type)) customKeepFood.delete(type);
		else customKeepFood.add(type);
		customKeepFood = new Set(customKeepFood);
	}

	function buildCustomFoodToKeep(): Record<string, number> {
		const food: Record<string, number> = {};
		for (const f of customKeepFood) food[f] = (food[f] || 0) + 1;
		return food;
	}

	function applyRecommendation(rec: SetupRecommendation) {
		dispatch('applySetup', {
			player_count: Number(playerCount),
			player_names: playerNames,
			turn_order: turnOrder,
			tray_cards: trayBirds.map(b => b.name),
			round_goals: goalSelections.map(g => g || 'No Goal'),
			birds_to_keep: rec.birds_to_keep,
			food_to_keep: rec.food_to_keep,
			bonus_card: rec.bonus_card
		});
	}

	function applyCustom() {
		const birds = Array.from(customKeepBirds);
		const food = buildCustomFoodToKeep();
		const requiredFood = Math.max(0, 5 - birds.length);
		const foodCount = Object.values(food).reduce((a, b) => a + b, 0);
		if (foodCount !== requiredFood) {
			error = `Custom setup needs ${requiredFood} food (you selected ${foodCount}).`;
			return;
		}
		const bonus = customBonusCard || selectedBonusCards[0]?.name || '';
		if (!bonus) {
			error = 'Select a bonus card to keep.';
			return;
		}
		dispatch('applySetup', {
			player_count: Number(playerCount),
			player_names: playerNames,
			turn_order: turnOrder,
			tray_cards: trayBirds.map(b => b.name),
			round_goals: goalSelections.map(g => g || 'No Goal'),
			birds_to_keep: birds,
			food_to_keep: food,
			bonus_card: bonus
		});
	}

	function continueToGame() {
		dispatch('continue', {
			player_count: Number(playerCount),
			player_names: playerNames,
			turn_order: turnOrder,
			tray_cards: trayBirds.map(b => b.name),
			round_goals: goalSelections.map(g => g || 'No Goal'),
		});
	}
</script>

<div class="setup-advisor card">
	<h2>Setup Draft Advisor</h2>
	<p class="subtitle">Enter your dealt cards to find the best starting combination.</p>

	{#if error}
		<div class="error">{error}</div>
	{/if}

	<!-- Player Count (from names) -->
	<div class="section">
		<h3>Players</h3>
		<div class="player-count">Players: {playerCount}</div>
	</div>

	<!-- Turn order -->
	<div class="section">
		<h3>Your Turn Order</h3>
		<select bind:value={turnOrder}>
			{#each Array(playerCount) as _, i}
				<option value={i + 1}>Turn {i + 1}</option>
			{/each}
		</select>
	</div>

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
						<li>
							<button
								class="dropdown-item"
								type="button"
								on:mousedown|preventDefault={() => selectBonusCard(bc)}
							>
								<span class="item-name">{bc.name}</span>
								<span class="item-desc">{bc.condition_text}</span>
							</button>
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
		<h3>Round Goals (4 rounds)</h3>
		<div class="goal-grid">
			{#each [0, 1, 2, 3] as idx}
				<div class="goal-row">
					<label for={`goal-round-${idx}`}>Round {idx + 1}</label>
					<select
						id={`goal-round-${idx}`}
						class="goal-select"
						value={goalSelections[idx]}
						on:change={(e) => setGoal(idx, e.currentTarget.value)}
					>
						<option value="No Goal">No Goal (+1 action later)</option>
						{#each allGoals as g}
							<option value={g.description}>{g.description}</option>
						{/each}
					</select>
				</div>
			{/each}
		</div>
	</div>

	<!-- Tray Cards -->
	<div class="section">
		<h3>Face-up Tray Cards ({trayBirds.length}/3)</h3>
		<BirdSearch placeholder="Add tray bird..." on:select={addTrayBird} disabled={trayBirds.length >= 3} />
		{#if trayBirds.length > 0}
			<div class="selected-list">
				{#each trayBirds as bird, i}
					<div class="selected-item">
						<div class="item-info">
							<span class="item-name">{bird.name}</span>
							<span class="item-meta">{bird.victory_points}VP</span>
						</div>
						<button class="remove" on:click={() => removeTrayBird(i)}>x</button>
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
		<button class="secondary" on:click={continueToGame}>
			Continue to Game
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
						<button class="apply-btn" on:click={() => applyRecommendation(rec)}>
							Use This Setup
						</button>
					</div>
				</div>
			{/each}
		</div>
	{/if}

	<!-- Manual selection -->
	{#if selectedBirds.length > 0 && selectedBonusCards.length > 0}
		<div class="section">
			<h3>Custom Setup</h3>
			<div class="custom-grid">
				<div>
					<div class="custom-label">Keep birds</div>
					{#each selectedBirds as bird}
						<label class="custom-option">
							<input type="checkbox" checked={customKeepBirds.has(bird.name)}
								on:change={() => toggleCustomBird(bird.name)} />
							{bird.name}
						</label>
					{/each}
				</div>
				<div>
					<div class="custom-label">Keep food</div>
					{#each ['invertebrate','seed','fish','fruit','rodent'] as ft}
						<label class="custom-option">
							<input type="checkbox" checked={customKeepFood.has(ft)}
								on:change={() => toggleCustomFood(ft)} />
							{FOOD_ICONS[ft] || ft} {ft}
						</label>
					{/each}
					<div class="custom-note">
						Keep {Math.max(0, 5 - customKeepBirds.size)} food total.
					</div>
				</div>
				<div>
					<div class="custom-label">Bonus card</div>
					<select bind:value={customBonusCard}>
						<option value="">Select bonus card</option>
						{#each selectedBonusCards as bc}
							<option value={bc.name}>{bc.name}</option>
						{/each}
					</select>
				</div>
			</div>
			<button class="primary apply-custom" on:click={applyCustom}>
				Apply Custom Setup
			</button>
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

	.goal-grid {
		display: grid;
		gap: 8px;
	}

	.goal-row {
		display: grid;
		grid-template-columns: 80px 1fr;
		gap: 8px;
		align-items: center;
	}

	.goal-select {
		width: 100%;
	}

	.apply-btn {
		margin-top: 8px;
	}

	.custom-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
		gap: 12px;
		margin-top: 8px;
	}

	.custom-label {
		font-weight: 600;
		margin-bottom: 6px;
	}

	.custom-option {
		display: block;
		font-size: 0.85rem;
		margin-bottom: 4px;
	}

	.custom-note {
		font-size: 0.75rem;
		color: var(--text-muted);
		margin-top: 6px;
	}

	.apply-custom {
		margin-top: 10px;
	}

	.player-count {
		font-size: 0.85rem;
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
