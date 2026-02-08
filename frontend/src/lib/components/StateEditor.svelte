<script lang="ts">
	import type { GameState, Bird, BonusCard } from '$lib/api/types';
	import { FOOD_ICONS, HABITAT_LABELS } from '$lib/api/types';
	import { updateGameState, getBonusCards } from '$lib/api/client';
	import BirdSearch from './BirdSearch.svelte';
	import { createEventDispatcher } from 'svelte';

	export let gameId: string;
	export let state: GameState;
	export let editingPlayer = 0;

	const dispatch = createEventDispatcher<{ updated: GameState }>();

	let saving = false;
	let error = '';
	$: player = state.players[editingPlayer];

	// Which habitat+slot are we editing
	let editingSlot: { habitat: number; slot: number } | null = null;

	const FOOD_TYPES = ['invertebrate', 'seed', 'fish', 'fruit', 'rodent', 'nectar'];

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

	function selectBonusCard(bc: BonusCard) {
		if (!player.bonus_cards.includes(bc.name)) {
			player.bonus_cards = [...player.bonus_cards, bc.name];
			state = state;
		}
		bonusSearchQuery = '';
		showBonusDropdown = false;
	}

	function removeBonusCard(index: number) {
		player.bonus_cards = player.bonus_cards.filter((_, i) => i !== index);
		state = state;
	}

	// Board slot editing
	function addBirdToSlot(bird: Bird) {
		if (!editingSlot) return;
		const p = state.players[editingPlayer];
		const row = p.board[editingSlot.habitat];
		row.slots[editingSlot.slot].bird_name = bird.name;
		state = state;
		editingSlot = null;
	}

	function playFromHand(handIndex: number) {
		if (!editingSlot) return;
		const p = state.players[editingPlayer];
		const birdName = p.hand[handIndex];
		const row = p.board[editingSlot.habitat];
		row.slots[editingSlot.slot].bird_name = birdName;
		p.hand.splice(handIndex, 1);
		state = state;
		editingSlot = null;
	}

	function removeBird(habIdx: number, slotIdx: number) {
		const slot = state.players[editingPlayer].board[habIdx].slots[slotIdx];
		slot.bird_name = null;
		slot.eggs = 0;
		slot.cached_food = {};
		slot.tucked_cards = 0;
		state = state;
	}

	// Hand editing
	function addBirdToHand(bird: Bird) {
		state.players[editingPlayer].hand.push(bird.name);
		state = state;
	}

	function removeFromHand(index: number) {
		state.players[editingPlayer].hand.splice(index, 1);
		state = state;
	}

	export async function saveState() {
		saving = true;
		error = '';
		try {
			const updated = await updateGameState(gameId, state);
			state = updated;
			dispatch('updated', updated);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to save state';
		} finally {
			saving = false;
		}
	}
</script>

<div class="state-editor card">
	<div class="editor-header">
		<h3>Edit Game State</h3>
		<button class="primary" on:click={saveState} disabled={saving}>
			{saving ? 'Saving...' : 'Save State'}
		</button>
	</div>

	{#if error}
		<div class="error">{error}</div>
	{/if}

	<!-- Game info -->
	<div class="game-info">
		<label>
			Round:
			<select bind:value={state.current_round}>
				{#each [1, 2, 3, 4] as r}
					<option value={r}>Round {r}</option>
				{/each}
			</select>
		</label>
		<label>
			Actions left:
			<input type="number" bind:value={player.action_cubes_remaining} min="0" max="8" style="width: 60px" />
		</label>
	</div>

	<!-- Food supply -->
	<div class="section">
		<h4>Food Supply</h4>
		<div class="food-grid">
			{#each FOOD_TYPES as ft}
				<label class="food-input">
					<span>{FOOD_ICONS[ft]}</span>
					<input
						type="number"
						bind:value={player.food_supply[ft]}
						min="0"
						max="20"
					/>
				</label>
			{/each}
		</div>
	</div>

	<!-- Board: habitat rows -->
	<div class="section">
		<h4>Board</h4>
		{#each player.board as row, habIdx}
			<div class="edit-habitat">
				<div class="hab-label habitat-{row.habitat}">
					{HABITAT_LABELS[row.habitat] || row.habitat}
				</div>
				<div class="edit-slots">
					{#each row.slots as slot, slotIdx}
						<div class="edit-slot">
							{#if slot.bird_name}
								<div class="slot-bird">
									<span class="slot-bird-name" title={slot.bird_name}>
										{slot.bird_name}
									</span>
									<button class="remove-btn" on:click={() => removeBird(habIdx, slotIdx)}>x</button>
								</div>
								<div class="slot-inputs">
									<label title="Eggs">
										🥚 <input type="number" bind:value={slot.eggs} min="0" max="10" />
									</label>
									<label title="Tucked cards">
										📄 <input type="number" bind:value={slot.tucked_cards} min="0" max="20" />
									</label>
								</div>
							{:else if editingSlot?.habitat === habIdx && editingSlot?.slot === slotIdx}
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
							{:else}
								<button class="add-btn" on:click={() => editingSlot = { habitat: habIdx, slot: slotIdx }}>
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
	<div class="section">
		<h4>Hand ({player.hand.length} known{#if player.unknown_hand_count > 0} + {player.unknown_hand_count} face-down{/if})</h4>
		<div class="hand-list">
			{#each player.hand as birdName, i}
				<span class="hand-card">
					{birdName}
					<button class="remove-btn" on:click={() => removeFromHand(i)}>x</button>
				</span>
			{/each}
		</div>
		<div class="hand-controls">
			<div class="add-hand">
				<BirdSearch on:select={(e) => addBirdToHand(e.detail)} placeholder="Add bird to hand..." />
			</div>
			<label class="face-down-input" title="Face-down cards (identity unknown)">
				Face-down:
				<input type="number" bind:value={player.unknown_hand_count} min="0" max="20" />
			</label>
		</div>
	</div>

	<!-- Bonus Cards -->
	<div class="section">
		<h4>Bonus Cards ({player.bonus_cards.length}{#if player.unknown_bonus_count > 0} + {player.unknown_bonus_count} unknown{/if})</h4>
		<div class="hand-list">
			{#each player.bonus_cards as bcName, i}
				<span class="hand-card bonus">
					{bcName}
					<button class="remove-btn" on:click={() => removeBonusCard(i)}>x</button>
				</span>
			{/each}
		</div>
		<div class="hand-controls">
			<div class="add-hand">
				<div class="search-container">
					<input
						type="text"
						bind:value={bonusSearchQuery}
						on:focus={() => showBonusDropdown = true}
						on:blur={() => setTimeout(() => showBonusDropdown = false, 200)}
						placeholder="Search bonus cards..."
					/>
					{#if showBonusDropdown && filteredBonusCards.length > 0}
						<ul class="dropdown">
							{#each filteredBonusCards.slice(0, 12) as bc}
								<li on:mousedown|preventDefault={() => selectBonusCard(bc)}>
									<span class="dropdown-name">{bc.name}</span>
									<span class="dropdown-desc">{bc.condition_text}</span>
								</li>
							{/each}
						</ul>
					{/if}
				</div>
			</div>
			<label class="face-down-input" title="Unknown bonus cards (count only)">
				Unknown:
				<input type="number" bind:value={player.unknown_bonus_count} min="0" max="10" />
			</label>
		</div>
	</div>
</div>

<style>
	.editor-header {
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

	.game-info {
		display: flex;
		gap: 16px;
		margin-bottom: 12px;
		align-items: center;
	}

	.game-info label {
		font-size: 0.85rem;
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.section {
		margin-bottom: 16px;
	}

	.section h4 {
		font-size: 0.85rem;
		color: var(--text-muted);
		margin-bottom: 6px;
	}

	.food-grid {
		display: flex;
		gap: 8px;
		flex-wrap: wrap;
	}

	.food-input {
		display: flex;
		align-items: center;
		gap: 4px;
		font-size: 0.85rem;
	}

	.food-input input {
		width: 50px;
		text-align: center;
	}

	/* Board editing */
	.edit-habitat {
		display: flex;
		gap: 6px;
		margin-bottom: 6px;
		align-items: stretch;
	}

	.hab-label {
		padding: 6px 10px;
		border-radius: 6px;
		font-size: 0.7rem;
		font-weight: 600;
		min-width: 80px;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.edit-slots {
		display: flex;
		gap: 4px;
		flex: 1;
	}

	.edit-slot {
		flex: 1;
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 6px;
		min-height: 50px;
		font-size: 0.75rem;
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.slot-bird {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.slot-bird-name {
		font-weight: 500;
		font-size: 0.7rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		flex: 1;
	}

	.slot-inputs {
		display: flex;
		gap: 6px;
	}

	.slot-inputs label {
		display: flex;
		align-items: center;
		gap: 2px;
		font-size: 0.7rem;
	}

	.slot-inputs input {
		width: 36px;
		padding: 2px 4px;
		font-size: 0.7rem;
		text-align: center;
	}

	.add-btn {
		font-size: 0.7rem;
		padding: 4px 8px;
		border: 1px dashed var(--border);
		background: transparent;
		color: var(--text-muted);
		border-radius: 4px;
	}

	.add-btn:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	.cancel-btn {
		font-size: 0.65rem;
		padding: 2px 6px;
		margin-top: 2px;
	}

	/* Play from hand quick-picks */
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

	.remove-btn {
		font-size: 0.65rem;
		padding: 1px 5px;
		border-radius: 3px;
		border: 1px solid #ddd;
		background: #f5f5f5;
		color: #999;
		line-height: 1;
	}

	.remove-btn:hover {
		background: #fee;
		color: #c00;
		border-color: #c00;
	}

	.hand-list {
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

	.hand-card.bonus {
		background: #f3e5f5;
		color: #6a1b9a;
	}

	.hand-controls {
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

	/* Bonus card search dropdown */
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
		max-height: 200px;
		overflow-y: auto;
		z-index: 100;
	}

	.dropdown li {
		padding: 6px 10px;
		cursor: pointer;
		border-bottom: 1px solid #f0f0f0;
	}

	.dropdown li:hover {
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
