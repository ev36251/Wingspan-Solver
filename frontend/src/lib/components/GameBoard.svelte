<script lang="ts">
	import type { Player, Bird, BonusCard } from '$lib/api/types';
	import { FOOD_ICONS, HABITAT_LABELS } from '$lib/api/types';
	import { getBonusCards } from '$lib/api/client';
	import BirdSearch from './BirdSearch.svelte';
	import { createEventDispatcher } from 'svelte';

	export let player: Player;
	export let isCurrentPlayer = false;

	const dispatch = createEventDispatcher<{ changed: void }>();

	const CACHE_FOOD_TYPES = ['invertebrate', 'seed', 'fish', 'fruit', 'rodent'];
	const FOOD_TYPES = ['invertebrate', 'seed', 'fish', 'fruit', 'rodent', 'nectar'];

	// Board slot editing state
	let editingSlot: { habitat: number; slot: number } | null = null;

	// Bonus card search state
	let allBonusCards: BonusCard[] = [];
	let bonusSearchQuery = '';
	let showBonusDropdown = false;

	async function loadBonusCards() {
		try {
			const data = await getBonusCards();
			allBonusCards = data.bonus_cards;
		} catch { /* ignore */ }
	}
	loadBonusCards();

	$: filteredBonusCards = bonusSearchQuery.length > 0
		? allBonusCards.filter(bc =>
			bc.name.toLowerCase().includes(bonusSearchQuery.toLowerCase()) ||
			bc.condition_text.toLowerCase().includes(bonusSearchQuery.toLowerCase())
		)
		: allBonusCards;

	// --- Helpers ---

	function totalCachedFood(slot: { cached_food: Record<string, number> }): number {
		return Object.values(slot.cached_food).reduce((a, b) => a + b, 0);
	}

	function cachedFoodDisplay(cached: Record<string, number>): string {
		return Object.entries(cached)
			.filter(([, v]) => v > 0)
			.map(([k, v]) => `${v}${FOOD_ICONS[k] || k}`)
			.join(' ');
	}

	// --- Existing inline edit functions ---

	function toggleEgg(habIdx: number, slotIdx: number, eggIdx: number) {
		const slot = player.board[habIdx].slots[slotIdx];
		if (!slot.bird_name) return;
		if (eggIdx < slot.eggs) {
			slot.eggs -= 1;
		} else {
			slot.eggs += 1;
		}
		player = player;
		dispatch('changed');
	}

	function adjustTucked(habIdx: number, slotIdx: number, delta: number) {
		const slot = player.board[habIdx].slots[slotIdx];
		slot.tucked_cards = Math.max(0, slot.tucked_cards + delta);
		player = player;
		dispatch('changed');
	}

	function addCachedFood(habIdx: number, slotIdx: number, foodType: string) {
		const slot = player.board[habIdx].slots[slotIdx];
		slot.cached_food[foodType] = (slot.cached_food[foodType] || 0) + 1;
		player = player;
		dispatch('changed');
	}

	function removeCachedFood(habIdx: number, slotIdx: number, foodType: string) {
		const slot = player.board[habIdx].slots[slotIdx];
		if (slot.cached_food[foodType] > 0) {
			slot.cached_food[foodType] -= 1;
			if (slot.cached_food[foodType] === 0) {
				delete slot.cached_food[foodType];
			}
		}
		player = player;
		dispatch('changed');
	}

	// --- New edit functions (from StateEditor) ---

	function adjustActionCubes(delta: number) {
		player.action_cubes_remaining = Math.max(0, Math.min(8, player.action_cubes_remaining + delta));
		player = player;
		dispatch('changed');
	}

	function getFoodCount(ft: string): number {
		return (player.food_supply as Record<string, number>)[ft] || 0;
	}

	function adjustFood(foodType: string, delta: number) {
		const supply = player.food_supply as Record<string, number>;
		supply[foodType] = Math.max(0, (supply[foodType] || 0) + delta);
		player = player;
		dispatch('changed');
	}

	function addBirdToSlot(bird: Bird) {
		if (!editingSlot) return;
		const row = player.board[editingSlot.habitat];
		row.slots[editingSlot.slot].bird_name = bird.name;
		row.slots[editingSlot.slot].egg_limit = bird.egg_limit;
		player = player;
		dispatch('changed');
		editingSlot = null;
	}

	function playFromHand(handIndex: number) {
		if (!editingSlot) return;
		const birdName = player.hand[handIndex];
		const row = player.board[editingSlot.habitat];
		row.slots[editingSlot.slot].bird_name = birdName;
		// egg_limit won't be set here since we only have the name
		player.hand.splice(handIndex, 1);
		player = player;
		dispatch('changed');
		editingSlot = null;
	}

	function removeBird(habIdx: number, slotIdx: number) {
		const slot = player.board[habIdx].slots[slotIdx];
		slot.bird_name = null;
		slot.eggs = 0;
		slot.egg_limit = 0;
		slot.cached_food = {};
		slot.tucked_cards = 0;
		player = player;
		dispatch('changed');
	}

	function addBirdToHand(bird: Bird) {
		player.hand.push(bird.name);
		player = player;
		dispatch('changed');
	}

	function removeFromHand(index: number) {
		player.hand.splice(index, 1);
		player = player;
		dispatch('changed');
	}

	function selectBonusCard(bc: BonusCard) {
		if (!player.bonus_cards.includes(bc.name)) {
			player.bonus_cards = [...player.bonus_cards, bc.name];
			player = player;
			dispatch('changed');
		}
		bonusSearchQuery = '';
		showBonusDropdown = false;
	}

	function removeBonusCard(index: number) {
		player.bonus_cards = player.bonus_cards.filter((_, i) => i !== index);
		player = player;
		dispatch('changed');
	}
</script>

<div class="game-board card" class:current-player={isCurrentPlayer}>
	<!-- Header -->
	<div class="board-header">
		<h3>{player.name}{isCurrentPlayer ? ' (Current)' : ''}</h3>
		<div class="player-stats">
			<span class="stat-editable" title="Action cubes remaining">
				<button class="adj-btn" on:click={() => adjustActionCubes(-1)}>-</button>
				<span>{player.action_cubes_remaining}</span>
				<button class="adj-btn" on:click={() => adjustActionCubes(1)}>+</button>
				actions
			</span>
		</div>
	</div>

	<!-- Food Supply (editable +/-) -->
	<div class="food-supply">
		{#each FOOD_TYPES as ft}
			<div class="food-token editable">
				<button class="food-adj" on:click={() => adjustFood(ft, -1)}>-</button>
				<span>{getFoodCount(ft)}{FOOD_ICONS[ft]}</span>
				<button class="food-adj" on:click={() => adjustFood(ft, 1)}>+</button>
			</div>
		{/each}
	</div>

	<!-- Habitats -->
	<div class="habitats">
		{#each player.board as row, habIdx}
			<div class="habitat-row">
				<div class="habitat-label habitat-{row.habitat}">
					{HABITAT_LABELS[row.habitat] || row.habitat}
					{#if row.nectar_spent > 0}
						<span class="nectar-badge">{row.nectar_spent}{FOOD_ICONS.nectar}</span>
					{/if}
				</div>
				<div class="slots">
					{#each row.slots as slot, slotIdx}
						<div class="slot" class:occupied={slot.bird_name} class:editing={editingSlot?.habitat === habIdx && editingSlot?.slot === slotIdx}>
							{#if slot.bird_name}
								<div class="bird-name-row">
									<span class="bird-name" title={slot.bird_name}>{slot.bird_name}</span>
									<button class="remove-btn" on:click={() => removeBird(habIdx, slotIdx)} title="Remove bird">x</button>
								</div>
								<!-- Egg outlines: click to fill/unfill -->
								{#if slot.egg_limit > 0}
									<div class="egg-row">
										{#each Array(slot.egg_limit) as _, eggIdx}
											<button
												class="egg-circle"
												class:filled={eggIdx < slot.eggs}
												on:click={() => toggleEgg(habIdx, slotIdx, eggIdx)}
												title={eggIdx < slot.eggs ? 'Remove egg' : 'Add egg'}
											></button>
										{/each}
									</div>
								{/if}
								<div class="slot-controls">
									<!-- Tucked cards -->
									<div class="tucked-row">
										<span class="control-label">tucked</span>
										<button class="adj-btn" on:click={() => adjustTucked(habIdx, slotIdx, -1)}>-</button>
										<span class="control-val">{slot.tucked_cards}</span>
										<button class="adj-btn" on:click={() => adjustTucked(habIdx, slotIdx, 1)}>+</button>
									</div>
									<!-- Cached food -->
									<div class="cached-row">
										<span class="control-label">cached</span>
										{#if totalCachedFood(slot) > 0}
											<span class="cached-display">{cachedFoodDisplay(slot.cached_food)}</span>
										{/if}
										<div class="cache-btns">
											{#each CACHE_FOOD_TYPES as ft}
												<button
													class="cache-food-btn"
													on:click={() => addCachedFood(habIdx, slotIdx, ft)}
													title="Cache {ft}"
												>{FOOD_ICONS[ft]}</button>
											{/each}
										</div>
										{#if totalCachedFood(slot) > 0}
											<div class="cache-remove-btns">
												{#each Object.entries(slot.cached_food) as [ft, count]}
													{#if count > 0}
														<button
															class="cache-remove-btn"
															on:click={() => removeCachedFood(habIdx, slotIdx, ft)}
															title="Remove 1 {ft}"
														>-{FOOD_ICONS[ft]}</button>
													{/if}
												{/each}
											</div>
										{/if}
									</div>
								</div>
							{:else if editingSlot?.habitat === habIdx && editingSlot?.slot === slotIdx}
								<div class="add-bird-panel">
									{#if player.hand.length > 0}
										<div class="from-hand-picks">
											<span class="pick-label">From hand:</span>
											{#each player.hand as birdName, hi}
												<button class="pick-btn" on:click={() => playFromHand(hi)} title={birdName}>
													{birdName}
												</button>
											{/each}
										</div>
									{/if}
									<BirdSearch on:select={(e) => addBirdToSlot(e.detail)} placeholder="Or search..." />
									<button class="cancel-btn" on:click={() => editingSlot = null}>Cancel</button>
								</div>
							{:else}
								<button class="add-bird-btn" on:click={() => editingSlot = { habitat: habIdx, slot: slotIdx }}>
									+ Add Bird
								</button>
							{/if}
						</div>
					{/each}
				</div>
			</div>
		{/each}
	</div>

	<!-- Hand -->
	<div class="hand-section">
		<h4 class="section-title">
			Hand ({player.hand.length} known{#if player.unknown_hand_count > 0} + {player.unknown_hand_count} face-down{/if})
		</h4>
		<div class="hand-list">
			{#each player.hand as birdName, i}
				<span class="hand-card">
					{birdName}
					<button class="remove-btn" on:click={() => removeFromHand(i)}>x</button>
				</span>
			{:else}
				<span class="no-cards">No known cards</span>
			{/each}
		</div>
		<div class="hand-controls">
			<div class="add-hand">
				<BirdSearch on:select={(e) => addBirdToHand(e.detail)} placeholder="Add bird to hand..." />
			</div>
			<label class="face-down-input" title="Face-down cards (identity unknown)">
				Face-down:
				<input type="number" bind:value={player.unknown_hand_count} min="0" max="20"
					on:change={() => { player = player; dispatch('changed'); }} />
			</label>
		</div>
	</div>

	<!-- Bonus Cards -->
	<div class="bonus-section">
		<h4 class="section-title">
			Bonus Cards ({player.bonus_cards.length}{#if player.unknown_bonus_count > 0} + {player.unknown_bonus_count} unknown{/if})
		</h4>
		<div class="bonus-list">
			{#each player.bonus_cards as bcName, i}
				<span class="bonus-badge editable">
					{bcName}
					<button class="remove-btn" on:click={() => removeBonusCard(i)}>x</button>
				</span>
			{:else}
				<span class="no-cards">No bonus cards</span>
			{/each}
		</div>
		<div class="bonus-controls">
			<div class="search-container">
				<input
					type="text"
					bind:value={bonusSearchQuery}
					on:focus={() => showBonusDropdown = true}
					on:blur={() => setTimeout(() => showBonusDropdown = false, 200)}
					placeholder="Search bonus cards..."
				/>
				{#if showBonusDropdown && filteredBonusCards.length > 0}
					<ul class="bonus-dropdown">
						{#each filteredBonusCards.slice(0, 12) as bc}
							<li on:mousedown|preventDefault={() => selectBonusCard(bc)}>
								<span class="dropdown-name">{bc.name}</span>
								<span class="dropdown-desc">{bc.condition_text}</span>
							</li>
						{/each}
					</ul>
				{/if}
			</div>
			<label class="face-down-input" title="Unknown bonus cards (count only)">
				Unknown:
				<input type="number" bind:value={player.unknown_bonus_count} min="0" max="10"
					on:change={() => { player = player; dispatch('changed'); }} />
			</label>
		</div>
	</div>
</div>

<style>
	.game-board {
		margin-bottom: 16px;
	}

	.game-board.current-player {
		border-color: var(--accent);
		border-width: 2px;
	}

	.board-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 8px;
	}

	.player-stats {
		display: flex;
		gap: 12px;
		font-size: 0.8rem;
		color: var(--text-muted);
	}

	.stat-editable {
		display: flex;
		align-items: center;
		gap: 4px;
	}

	/* Food supply */
	.food-supply {
		display: flex;
		gap: 6px;
		margin-bottom: 12px;
		flex-wrap: wrap;
	}

	.food-token {
		font-size: 0.85rem;
		background: #f5f0e8;
		padding: 2px 6px;
		border-radius: 4px;
		display: flex;
		align-items: center;
		gap: 2px;
	}

	.food-adj {
		width: 18px;
		height: 18px;
		padding: 0;
		font-size: 0.7rem;
		font-weight: 700;
		border: 1px solid var(--border);
		background: #f5f5f5;
		border-radius: 3px;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.food-adj:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	/* Habitats */
	.habitats {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.habitat-row {
		display: flex;
		gap: 6px;
		align-items: stretch;
	}

	.habitat-label {
		writing-mode: horizontal-tb;
		padding: 6px 10px;
		border-radius: 6px;
		font-size: 0.75rem;
		font-weight: 600;
		min-width: 90px;
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		gap: 2px;
	}

	.nectar-badge {
		font-size: 0.7rem;
		opacity: 0.9;
	}

	.slots {
		display: flex;
		gap: 4px;
		flex: 1;
	}

	.slot {
		flex: 1;
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 6px;
		min-height: 60px;
		font-size: 0.75rem;
		background: white;
		display: flex;
		flex-direction: column;
		justify-content: flex-start;
	}

	.slot.occupied {
		background: #fefdf8;
	}

	.slot.editing {
		overflow: visible;
		z-index: 10;
	}

	/* Bird name with remove button */
	.bird-name-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 2px;
	}

	.bird-name {
		font-weight: 500;
		font-size: 0.7rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		flex: 1;
	}

	/* Egg circles */
	.egg-row {
		display: flex;
		gap: 3px;
		margin-top: 4px;
	}

	.egg-circle {
		width: 16px;
		height: 16px;
		border-radius: 50%;
		border: 2px solid #a5d6a7;
		background: transparent;
		cursor: pointer;
		padding: 0;
		transition: background 0.1s;
	}

	.egg-circle.filled {
		background: #4caf50;
		border-color: #388e3c;
	}

	.egg-circle:hover {
		border-color: #2e7d32;
		transform: scale(1.1);
	}

	/* Slot controls */
	.slot-controls {
		display: flex;
		flex-direction: column;
		gap: 2px;
		margin-top: 4px;
	}

	.tucked-row, .cached-row {
		display: flex;
		align-items: center;
		gap: 3px;
		flex-wrap: wrap;
	}

	.control-label {
		font-size: 0.6rem;
		color: var(--text-muted);
		min-width: 32px;
	}

	.control-val {
		font-size: 0.65rem;
		font-weight: 600;
		min-width: 12px;
		text-align: center;
	}

	.adj-btn {
		width: 16px;
		height: 16px;
		padding: 0;
		font-size: 0.6rem;
		font-weight: 700;
		border: 1px solid var(--border);
		background: #f5f5f5;
		border-radius: 3px;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.adj-btn:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	/* Cached food */
	.cached-display {
		font-size: 0.6rem;
		font-weight: 600;
		color: #e65100;
	}

	.cache-btns {
		display: flex;
		gap: 1px;
	}

	.cache-food-btn {
		width: 18px;
		height: 16px;
		padding: 0;
		font-size: 0.55rem;
		border: 1px solid var(--border);
		background: #fff3e0;
		border-radius: 2px;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.cache-food-btn:hover {
		border-color: #e65100;
		background: #ffe0b2;
	}

	.cache-remove-btns {
		display: flex;
		gap: 2px;
	}

	.cache-remove-btn {
		padding: 0 3px;
		font-size: 0.55rem;
		border: 1px solid #ffccbc;
		background: #fff3e0;
		border-radius: 2px;
		cursor: pointer;
		color: #bf360c;
	}

	.cache-remove-btn:hover {
		background: #ffccbc;
	}

	/* Add bird to slot */
	.add-bird-btn {
		font-size: 0.7rem;
		padding: 4px 8px;
		border: 1px dashed var(--border);
		background: transparent;
		color: var(--text-muted);
		border-radius: 4px;
		cursor: pointer;
		width: 100%;
		text-align: center;
	}

	.add-bird-btn:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	.add-bird-panel {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.from-hand-picks {
		display: flex;
		flex-wrap: wrap;
		gap: 3px;
		margin-bottom: 4px;
	}

	.pick-label {
		font-size: 0.6rem;
		color: var(--text-muted);
		width: 100%;
	}

	.pick-btn {
		font-size: 0.6rem;
		padding: 2px 5px;
		border: 1px solid var(--accent);
		background: #fdf8ef;
		color: var(--accent);
		border-radius: 3px;
		cursor: pointer;
		max-width: 100%;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.pick-btn:hover {
		background: var(--accent);
		color: white;
	}

	.cancel-btn {
		font-size: 0.65rem;
		padding: 2px 6px;
		margin-top: 2px;
		border: 1px solid var(--border);
		background: #f5f5f5;
		border-radius: 3px;
		cursor: pointer;
	}

	/* Remove button (shared) */
	.remove-btn {
		font-size: 0.65rem;
		padding: 1px 5px;
		border-radius: 3px;
		border: 1px solid #ddd;
		background: #f5f5f5;
		color: #999;
		line-height: 1;
		cursor: pointer;
		flex-shrink: 0;
	}

	.remove-btn:hover {
		background: #fee;
		color: #c00;
		border-color: #c00;
	}

	/* Hand section */
	.hand-section, .bonus-section {
		margin-top: 12px;
		padding-top: 10px;
		border-top: 1px solid var(--border);
	}

	.section-title {
		font-size: 0.8rem;
		color: var(--text-muted);
		margin-bottom: 6px;
		font-weight: 600;
	}

	.hand-list, .bonus-list {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin-bottom: 8px;
	}

	.hand-card {
		font-size: 0.8rem;
		background: #f5f0e8;
		padding: 3px 8px;
		border-radius: 4px;
		display: flex;
		align-items: center;
		gap: 4px;
	}

	.no-cards {
		font-size: 0.75rem;
		color: var(--text-muted);
		font-style: italic;
	}

	.hand-controls, .bonus-controls {
		display: flex;
		gap: 12px;
		align-items: flex-start;
	}

	.add-hand {
		flex: 1;
		max-width: 300px;
	}

	.face-down-input {
		font-size: 0.8rem;
		display: flex;
		align-items: center;
		gap: 4px;
		white-space: nowrap;
		color: var(--text-muted);
	}

	.face-down-input input {
		width: 50px;
		text-align: center;
	}

	/* Bonus cards */
	.bonus-badge {
		font-size: 0.7rem;
		background: #f3e5f5;
		color: #6a1b9a;
		padding: 2px 8px;
		border-radius: 4px;
		display: flex;
		align-items: center;
		gap: 4px;
	}

	/* Bonus card search dropdown */
	.search-container {
		position: relative;
		flex: 1;
		max-width: 300px;
	}

	.search-container input {
		width: 100%;
	}

	.bonus-dropdown {
		position: absolute;
		top: 100%;
		left: 0;
		right: 0;
		background: white;
		border: 1px solid var(--border);
		border-radius: 0 0 6px 6px;
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
		list-style: none;
		max-height: 200px;
		overflow-y: auto;
		z-index: 100;
	}

	.bonus-dropdown li {
		padding: 6px 10px;
		cursor: pointer;
		border-bottom: 1px solid #f0f0f0;
	}

	.bonus-dropdown li:hover {
		background: #f5f0e8;
	}

	.dropdown-name {
		display: block;
		font-weight: 500;
		font-size: 0.8rem;
	}

	.dropdown-desc {
		display: block;
		font-size: 0.7rem;
		color: var(--text-muted);
	}
</style>
