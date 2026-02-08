<script lang="ts">
	import type { Player, Bird, BonusCard } from '$lib/api/types';
	import { FOOD_ICONS, HABITAT_LABELS, NEST_ICONS } from '$lib/api/types';
	import { getBonusCards } from '$lib/api/client';
	import BirdSearch from './BirdSearch.svelte';
	import { createEventDispatcher } from 'svelte';

	export let player: Player;
	export let isCurrentPlayer = false;
	export let currentRound: number = 1;

	const dispatch = createEventDispatcher<{ changed: void }>();

	const CACHE_FOOD_TYPES = ['invertebrate', 'seed', 'fish', 'fruit', 'rodent'];
	const FOOD_TYPES = ['invertebrate', 'seed', 'fish', 'fruit', 'rodent', 'nectar'];
	// Vertical sidebar order: nectar, fruit, fish, invertebrate, rodent, seed
	const SIDEBAR_FOOD_ORDER = ['nectar', 'fruit', 'fish', 'invertebrate', 'rodent', 'seed'];

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

	// Reactive egg totals
	$: eggCurrent = player.board.reduce((sum, row) =>
		sum + row.slots.reduce((s, slot) => s + slot.eggs, 0), 0);
	$: eggCapacity = player.board.reduce((sum, row) =>
		sum + row.slots.reduce((s, slot) => s + (slot.bird_name ? slot.egg_limit : 0), 0), 0);

	// Reactive food counts — Svelte tracks direct property access on player
	$: foodCounts = {
		invertebrate: player.food_supply.invertebrate,
		seed: player.food_supply.seed,
		fish: player.food_supply.fish,
		fruit: player.food_supply.fruit,
		rodent: player.food_supply.rodent,
		nectar: player.food_supply.nectar,
	} as Record<string, number>;

	function adjustFood(foodType: string, delta: number) {
		const key = foodType as keyof typeof player.food_supply;
		const current = player.food_supply[key] || 0;
		(player.food_supply as Record<string, number>)[key] = Math.max(0, current + delta);
		player.food_supply = { ...player.food_supply };
		player = player;
		dispatch('changed');
	}

	function addBirdToSlot(bird: Bird) {
		if (!editingSlot) return;
		const row = player.board[editingSlot.habitat];
		row.slots[editingSlot.slot].bird_name = bird.name;
		row.slots[editingSlot.slot].egg_limit = bird.egg_limit;
		if (player.unknown_hand_count > 0) {
			player.unknown_hand_count -= 1;
		}
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

	// Hand drag-and-drop reordering
	let handDragIdx: number | null = null;
	let handDragOverIdx: number | null = null;

	function onHandDragStart(idx: number, e: DragEvent) {
		handDragIdx = idx;
		if (e.dataTransfer) e.dataTransfer.effectAllowed = 'move';
	}
	function onHandDragOver(idx: number, e: DragEvent) {
		e.preventDefault();
		handDragOverIdx = idx;
	}
	function onHandDrop(idx: number) {
		if (handDragIdx === null || handDragIdx === idx) {
			handDragIdx = null;
			handDragOverIdx = null;
			return;
		}
		const arr = [...player.hand];
		const [moved] = arr.splice(handDragIdx, 1);
		arr.splice(idx, 0, moved);
		player.hand = arr;
		player = player;
		dispatch('changed');
		handDragIdx = null;
		handDragOverIdx = null;
	}
	function onHandDragEnd() {
		handDragIdx = null;
		handDragOverIdx = null;
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

	<!-- Board area: sidebar + habitats -->
	<div class="board-area">
		<!-- Player Sidebar: round, eggs, food -->
		<div class="player-sidebar">
			<!-- Round counter -->
			<div class="sidebar-item round-counter" title="Current round">
				<span class="sidebar-icon">&#x1F4E6;</span>
				<span class="sidebar-value">{currentRound}</span>
			</div>
			<!-- Egg counter -->
			<div class="sidebar-item egg-counter" title="Eggs placed / total egg capacity">
				<span class="sidebar-icon">&#x1F95A;</span>
				<span class="sidebar-value">{eggCurrent}<span class="sidebar-denom">/{eggCapacity}</span></span>
			</div>
			<!-- Food supply (vertical) -->
			{#each SIDEBAR_FOOD_ORDER as ft}
				<div class="sidebar-food" title={ft}>
					<span class="sidebar-food-icon">{FOOD_ICONS[ft]}</span>
					<span class="sidebar-food-count">{foodCounts[ft]}</span>
					<div class="sidebar-food-btns">
						<button class="food-adj" on:click={() => adjustFood(ft, 1)}>+</button>
						<button class="food-adj" on:click={() => adjustFood(ft, -1)}>-</button>
					</div>
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
								<!-- VP badge + nest icon + bird name + remove -->
								<div class="bird-header">
									<span class="vp-badge" title="Victory points">{slot.victory_points}</span>
									{#if slot.nest_type}
										<span class="nest-icon nest-{slot.nest_type}" title="{slot.nest_type} nest">{NEST_ICONS[slot.nest_type] || '?'}</span>
									{/if}
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
										<span class="tucked-count" class:has-tucked={slot.tucked_cards > 0}>{slot.tucked_cards}</span>
										<button class="adj-btn" on:click={() => adjustTucked(habIdx, slotIdx, 1)}>+</button>
									</div>
									<!-- Cached food -->
									<div class="cached-row">
										<span class="control-label">cached</span>
										{#if totalCachedFood(slot) > 0}
											<span class="cached-display">{cachedFoodDisplay(slot.cached_food)}</span>
											{#if slot.cache_spendable}
												<span class="spendable-hint" title="Can be spent as food for bird costs (loses 1pt each)">spendable</span>
											{/if}
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
	</div> <!-- /board-area -->

	<!-- Hand -->
	<div class="hand-section">
		<h4 class="section-title">
			Hand ({player.hand.length} known{#if player.unknown_hand_count > 0} + {player.unknown_hand_count} face-down{/if})
		</h4>
		<div class="hand-list">
			{#each player.hand as birdName, i}
				<span
					class="hand-card"
					draggable="true"
					class:drag-over={handDragOverIdx === i && handDragIdx !== i}
					on:dragstart={(e) => onHandDragStart(i, e)}
					on:dragover={(e) => onHandDragOver(i, e)}
					on:drop={() => onHandDrop(i)}
					on:dragend={onHandDragEnd}
				>
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

	/* Board area: sidebar + habitats side-by-side */
	.board-area {
		display: flex;
		gap: 8px;
		margin-bottom: 12px;
	}

	/* Player sidebar */
	.player-sidebar {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 4px;
		background: #f9f6f0;
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 8px 6px;
		min-width: 64px;
	}

	.sidebar-item {
		display: flex;
		flex-direction: column;
		align-items: center;
		padding: 4px 0;
		border-bottom: 1px solid #e0dcd4;
		width: 100%;
	}

	.sidebar-icon {
		font-size: 1.2rem;
		line-height: 1;
	}

	.sidebar-value {
		font-size: 1.1rem;
		font-weight: 700;
		line-height: 1.2;
	}

	.sidebar-denom {
		font-size: 0.75rem;
		font-weight: 400;
		color: var(--text-muted);
	}

	.sidebar-food {
		display: flex;
		align-items: center;
		gap: 2px;
		width: 100%;
		justify-content: center;
	}

	.sidebar-food-icon {
		font-size: 1rem;
		width: 20px;
		text-align: center;
	}

	.sidebar-food-count {
		font-size: 1rem;
		font-weight: 700;
		min-width: 14px;
		text-align: center;
	}

	.sidebar-food-btns {
		display: flex;
		flex-direction: column;
		gap: 1px;
	}

	.food-adj {
		width: 16px;
		height: 14px;
		padding: 0;
		font-size: 0.6rem;
		font-weight: 700;
		border: 1px solid var(--border);
		background: #f5f5f5;
		border-radius: 2px;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		line-height: 1;
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
		flex: 1;
		min-width: 0;
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

	/* Bird header: VP badge + nest icon + name + remove */
	.bird-header {
		display: flex;
		align-items: center;
		gap: 3px;
	}

	.vp-badge {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 18px;
		height: 18px;
		border-radius: 50%;
		background: #1565c0;
		color: white;
		font-size: 0.6rem;
		font-weight: 700;
		flex-shrink: 0;
		line-height: 1;
	}

	.nest-icon {
		font-size: 0.85rem;
		flex-shrink: 0;
		line-height: 1;
	}

	.nest-platform { color: #8d6e63; }
	.nest-bowl { color: #43a047; }
	.nest-cavity { color: #5c6bc0; }
	.nest-ground { color: #ef6c00; }
	.nest-wild { color: #fdd835; }

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

	.tucked-count {
		font-size: 0.7rem;
		font-weight: 600;
		min-width: 14px;
		text-align: center;
		color: var(--text-muted);
	}

	.tucked-count.has-tucked {
		font-size: 0.85rem;
		font-weight: 800;
		color: #6a1b9a;
		background: #f3e5f5;
		border-radius: 4px;
		padding: 0 4px;
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
		font-size: 0.8rem;
		font-weight: 700;
		color: #e65100;
	}

	.spendable-hint {
		font-size: 0.65rem;
		font-weight: 700;
		font-style: italic;
		color: #e65100;
		cursor: help;
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
		cursor: grab;
		border: 1px solid transparent;
		transition: border-color 0.15s, box-shadow 0.15s;
	}

	.hand-card:active {
		cursor: grabbing;
	}

	.hand-card.drag-over {
		border-color: var(--accent);
		box-shadow: 0 0 0 2px rgba(196, 139, 47, 0.3);
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
