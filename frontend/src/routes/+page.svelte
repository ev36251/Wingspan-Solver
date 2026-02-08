<script lang="ts">
	import type { GameState, Bird, Goal } from '$lib/api/types';
	import { FOOD_ICONS } from '$lib/api/types';
	import { createGame, updateGameState, getGoals } from '$lib/api/client';
	import GameBoard from '$lib/components/GameBoard.svelte';
	import StateEditor from '$lib/components/StateEditor.svelte';
	import SolverPanel from '$lib/components/SolverPanel.svelte';
	import ScoreSheet from '$lib/components/ScoreSheet.svelte';
	import SetupAdvisor from '$lib/components/SetupAdvisor.svelte';
	import BirdSearch from '$lib/components/BirdSearch.svelte';

	let gameId = '';
	let state: GameState | null = null;
	let error = '';
	let loading = false;
	let saving = false;

	// New game form
	let playerNames = ['Player 1', 'Player 2'];
	let showNewGame = true;
	let showEditor = false;
	let activeTab: 'setup' | 'newgame' = 'setup';
	let activePlayerIdx = 0;

	let scoreSheet: ScoreSheet;

	// Goal editing
	let allGoals: Goal[] = [];
	let goalSelections: string[] = ['', '', '', ''];

	async function loadGoals() {
		try {
			const data = await getGoals();
			allGoals = data.goals;
		} catch { /* ignore */ }
	}
	loadGoals();

	function syncGoalSelections() {
		if (!state) return;
		goalSelections = [
			state.round_goals[0] || '',
			state.round_goals[1] || '',
			state.round_goals[2] || '',
			state.round_goals[3] || '',
		];
	}

	function setGoal(roundIdx: number, value: string) {
		goalSelections[roundIdx] = value;
		goalSelections = [...goalSelections];
		if (!state) return;
		const goals = [...goalSelections];
		while (goals.length > 0 && goals[goals.length - 1] === '') {
			goals.pop();
		}
		state.round_goals = goals;
		state = state;
	}

	// Birdfeeder constants
	const FEEDER_SINGLE = ['invertebrate', 'seed', 'fish', 'fruit', 'rodent'];
	const FEEDER_CHOICE = [
		{ foods: ['invertebrate', 'seed'], label: '🪱/🌾' },
		{ foods: ['nectar', 'fruit'], label: '🌺/🍒' },
		{ foods: ['nectar', 'seed'], label: '🌺/🌾' },
	];

	function addPlayer() {
		if (playerNames.length < 5) {
			playerNames = [...playerNames, `Player ${playerNames.length + 1}`];
		}
	}

	function removePlayer(i: number) {
		if (playerNames.length > 1) {
			playerNames = playerNames.filter((_, idx) => idx !== i);
		}
	}

	async function startGame() {
		loading = true;
		error = '';
		try {
			const names = playerNames.filter(n => n.trim());
			if (names.length === 0) {
				error = 'Add at least one player';
				return;
			}
			const data = await createGame(names);
			gameId = data.game_id;
			state = data.state;
			showNewGame = false;
			activePlayerIdx = 0;
			syncGoalSelections();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to create game';
		} finally {
			loading = false;
		}
	}

	function handleStateUpdated(e: CustomEvent<GameState>) {
		state = e.detail;
		syncGoalSelections();
		scoreSheet?.refresh();
	}

	function resetGame() {
		gameId = '';
		state = null;
		showNewGame = true;
		showEditor = false;
		activePlayerIdx = 0;
	}

	// Save state from header (works even when editor is hidden)
	async function saveState() {
		if (!state) return;
		saving = true;
		try {
			state = await updateGameState(gameId, state);
			scoreSheet?.refresh();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to save';
		} finally {
			saving = false;
		}
	}

	// Birdfeeder functions
	function addDie(foodType: string) {
		if (!state) return;
		state.birdfeeder.dice = [...state.birdfeeder.dice, foodType];
		state = state;
	}

	function addChoiceDie(foods: string[]) {
		if (!state) return;
		state.birdfeeder.dice = [...state.birdfeeder.dice, foods];
		state = state;
	}

	function removeDie(index: number) {
		if (!state) return;
		state.birdfeeder.dice = state.birdfeeder.dice.filter((_, i) => i !== index);
		state = state;
	}

	function clearFeeder() {
		if (!state) return;
		state.birdfeeder.dice = [];
		state = state;
	}

	// Card tray functions
	function addBirdToTray(bird: Bird) {
		if (!state || state.card_tray.face_up.length >= 3) return;
		state.card_tray.face_up = [...state.card_tray.face_up, bird.name];
		state = state;
	}

	function removeTrayCard(index: number) {
		if (!state) return;
		state.card_tray.face_up = state.card_tray.face_up.filter((_, i) => i !== index);
		state = state;
	}

	function trayToHand(trayIndex: number, playerIndex: number) {
		if (!state) return;
		const birdName = state.card_tray.face_up[trayIndex];
		state.card_tray.face_up = state.card_tray.face_up.filter((_, i) => i !== trayIndex);
		state.players[playerIndex].hand.push(birdName);
		state = state;
	}
</script>

{#if showNewGame}
	<!-- Tab selector -->
	<div class="tab-bar">
		<button
			class="tab"
			class:active={activeTab === 'setup'}
			on:click={() => activeTab = 'setup'}
		>Draft Advisor</button>
		<button
			class="tab"
			class:active={activeTab === 'newgame'}
			on:click={() => activeTab = 'newgame'}
		>New Game</button>
	</div>

	{#if activeTab === 'setup'}
		<SetupAdvisor />
	{:else}
		<!-- Game creation screen -->
		<div class="new-game card">
			<h2>New Game</h2>
			<p class="subtitle">Enter player names to start a new Wingspan game session.</p>

			{#if error}
				<div class="error">{error}</div>
			{/if}

			<div class="player-list">
				{#each playerNames as name, i}
					<div class="player-input">
						<input
							type="text"
							bind:value={playerNames[i]}
							placeholder="Player name"
						/>
						{#if playerNames.length > 1}
							<button class="remove" on:click={() => removePlayer(i)}>x</button>
						{/if}
					</div>
				{/each}
			</div>

			<div class="actions">
				{#if playerNames.length < 5}
					<button on:click={addPlayer}>+ Add Player</button>
				{/if}
				<button class="primary" on:click={startGame} disabled={loading}>
					{loading ? 'Creating...' : 'Start Game'}
				</button>
			</div>
		</div>
	{/if}

{:else if state}
	<!-- Game view -->
	<div class="game-view">
		<div class="game-header">
			<div class="game-info">
				<span class="game-id">Game: {gameId}</span>
				<span class="round-info">Round {state.current_round}/4</span>
				<span class="turn-info">
					Current: {state.players[state.current_player_idx]?.name}
				</span>
			</div>
			<div class="game-actions">
				<button class="primary" on:click={saveState} disabled={saving}>
					{saving ? 'Saving...' : 'Save State'}
				</button>
				<button on:click={() => showEditor = !showEditor}>
					{showEditor ? 'Hide Editor' : 'Edit State'}
				</button>
				<button on:click={resetGame}>New Game</button>
			</div>
		</div>

		<!-- Player tabs -->
		<div class="player-tabs">
			{#each state.players as p, i}
				<button
					class="player-tab"
					class:active={activePlayerIdx === i}
					class:is-current={i === state.current_player_idx}
					on:click={() => activePlayerIdx = i}
				>
					{p.name}
					{#if i === state.current_player_idx}
						<span class="current-dot">&#9679;</span>
					{/if}
				</button>
			{/each}
		</div>

		<div class="game-layout">
			<div class="main-column">
				<!-- State editor (collapsible) -->
				{#if showEditor}
					<StateEditor {gameId} {state} editingPlayer={activePlayerIdx} on:updated={handleStateUpdated} />
				{/if}

				<!-- Active player board -->
				<GameBoard
					player={state.players[activePlayerIdx]}
					isCurrentPlayer={activePlayerIdx === state.current_player_idx}
				/>

				<!-- Card Tray + Score side by side -->
				<div class="bottom-row">
					<div class="tray-panel card">
						<h4 class="panel-title">Card Tray ({state.card_tray.face_up.length}/3)</h4>
						<div class="tray-cards">
							{#each state.card_tray.face_up as birdName, i}
								<div class="tray-card">
									<span class="tray-bird-name">{birdName}</span>
									<div class="tray-actions">
										{#each state.players as p, pi}
											<button
												class="tray-to-hand"
												on:click={() => trayToHand(i, pi)}
												title="Give to {p.name}"
											>
												&#8594; {p.name}
											</button>
										{/each}
										<button class="remove-btn" on:click={() => removeTrayCard(i)}>x</button>
									</div>
								</div>
							{:else}
								<span class="tray-empty">No face-up cards</span>
							{/each}
						</div>
						{#if state.card_tray.face_up.length < 3}
							<div style="margin-top: 6px;">
								<BirdSearch on:select={(e) => addBirdToTray(e.detail)} placeholder="Add bird to tray..." />
							</div>
						{/if}
					</div>

					<!-- Score Sheet -->
					<ScoreSheet {gameId} bind:this={scoreSheet} />
				</div>
			</div>

			<div class="side-column">
				<!-- Round Goals (editable) -->
				<div class="sidebar-panel card">
					<h4 class="panel-title">Round Goals</h4>
					<div class="goals-list">
						{#each [0, 1, 2, 3] as idx}
							<div class="goal-item" class:active-round={state.current_round === idx + 1}>
								<span class="goal-round">R{idx + 1}</span>
								<select
									class="goal-select"
									value={goalSelections[idx]}
									on:change={(e) => setGoal(idx, e.currentTarget.value)}
								>
									<option value="">-- Not set --</option>
									<option value="No Goal">No Goal (+1 action later)</option>
									{#each allGoals as g}
										<option value={g.description}>{g.description}</option>
									{/each}
								</select>
							</div>
						{/each}
					</div>
				</div>

				<!-- Birdfeeder -->
				<div class="sidebar-panel card">
					<div class="panel-header-row">
						<h4 class="panel-title">Birdfeeder ({state.birdfeeder.dice.length})</h4>
						{#if state.birdfeeder.dice.length > 0}
							<button class="clear-btn" on:click={clearFeeder}>Clear</button>
						{/if}
					</div>
					<div class="feeder-dice">
						{#each state.birdfeeder.dice as die, i}
							<button class="die-token" on:click={() => removeDie(i)} title="Click to remove">
								{#if Array.isArray(die)}
									{die.map(d => FOOD_ICONS[d] || d).join('/')}
								{:else}
									{FOOD_ICONS[die] || die}
								{/if}
							</button>
						{:else}
							<span class="feeder-empty">Empty feeder</span>
						{/each}
					</div>
					<div class="feeder-add">
						{#each FEEDER_SINGLE as ft}
							<button class="feeder-add-btn" on:click={() => addDie(ft)} title="Add {ft}">
								{FOOD_ICONS[ft]}
							</button>
						{/each}
					</div>
					<div class="feeder-add">
						{#each FEEDER_CHOICE as choice}
							<button class="feeder-add-btn choice" on:click={() => addChoiceDie(choice.foods)} title="Add choice die">
								{choice.label}
							</button>
						{/each}
					</div>
				</div>

				<!-- Solver -->
				<SolverPanel {gameId} disabled={state.current_round > 4} />
			</div>
		</div>
	</div>
{/if}

<style>
	/* Setup tab bar */
	.tab-bar {
		display: flex;
		gap: 0;
		max-width: 700px;
		margin: 20px auto 0;
		border-bottom: 2px solid var(--border);
	}

	.tab {
		padding: 10px 24px;
		background: none;
		border: none;
		border-bottom: 2px solid transparent;
		margin-bottom: -2px;
		font-size: 0.95rem;
		color: var(--text-muted);
		cursor: pointer;
		font-weight: 500;
	}

	.tab:hover {
		color: var(--text);
	}

	.tab.active {
		color: var(--accent);
		border-bottom-color: var(--accent);
	}

	.new-game {
		max-width: 500px;
		margin: 20px auto;
	}

	.new-game h2 {
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

	.player-list {
		display: flex;
		flex-direction: column;
		gap: 8px;
		margin-bottom: 16px;
	}

	.player-input {
		display: flex;
		gap: 8px;
	}

	.player-input input {
		flex: 1;
	}

	.remove {
		font-size: 0.8rem;
		padding: 4px 10px;
		border: 1px solid #ddd;
		background: #f9f9f9;
		color: #999;
	}

	.remove:hover {
		background: #fee;
		color: #c00;
		border-color: #c00;
	}

	.actions {
		display: flex;
		gap: 8px;
		justify-content: flex-end;
	}

	/* Game view */
	.game-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 12px;
		padding-bottom: 12px;
		border-bottom: 1px solid var(--border);
	}

	.game-info {
		display: flex;
		gap: 16px;
		font-size: 0.85rem;
	}

	.game-id {
		color: var(--text-muted);
		font-family: monospace;
	}

	.round-info {
		font-weight: 600;
		color: var(--accent);
	}

	.turn-info {
		color: var(--text);
	}

	.game-actions {
		display: flex;
		gap: 8px;
	}

	/* Player tabs */
	.player-tabs {
		display: flex;
		gap: 0;
		margin-bottom: 16px;
		border-bottom: 2px solid var(--border);
	}

	.player-tab {
		padding: 8px 20px;
		background: none;
		border: none;
		border-bottom: 2px solid transparent;
		margin-bottom: -2px;
		font-size: 0.9rem;
		color: var(--text-muted);
		cursor: pointer;
		font-weight: 500;
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.player-tab:hover {
		color: var(--text);
		background: none;
	}

	.player-tab.active {
		color: var(--accent);
		border-bottom-color: var(--accent);
	}

	.player-tab.is-current .current-dot {
		color: #4caf50;
		font-size: 0.7rem;
	}

	/* Game layout */
	.game-layout {
		display: grid;
		grid-template-columns: 1fr 380px;
		gap: 16px;
		align-items: start;
	}

	.main-column {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.bottom-row {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 12px;
	}

	.tray-panel {
		padding: 12px 16px;
	}

	.side-column {
		display: flex;
		flex-direction: column;
		gap: 12px;
		position: sticky;
		top: 16px;
		max-height: calc(100vh - 32px);
		overflow-y: auto;
	}

	/* Sidebar panels */
	.sidebar-panel {
		padding: 12px;
	}

	.panel-title {
		font-size: 0.85rem;
		color: var(--text);
		margin-bottom: 6px;
	}

	.panel-header-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 6px;
	}

	.panel-header-row .panel-title {
		margin-bottom: 0;
	}

	/* Round Goals */
	.goals-list {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.goal-item {
		display: flex;
		gap: 6px;
		align-items: center;
		padding: 4px 6px;
		border-radius: 4px;
		font-size: 0.8rem;
	}

	.goal-item.active-round {
		background: #fef9f0;
		border: 1px solid var(--accent);
	}

	.goal-round {
		font-weight: 700;
		color: var(--accent);
		min-width: 24px;
		flex-shrink: 0;
	}

	.goal-select {
		flex: 1;
		font-size: 0.75rem;
		padding: 3px 6px;
		min-width: 0;
	}

	/* Birdfeeder */
	.feeder-dice {
		display: flex;
		gap: 6px;
		flex-wrap: wrap;
		margin-bottom: 8px;
		min-height: 32px;
	}

	.die-token {
		font-size: 1.1rem;
		padding: 4px 10px;
		border: 1px solid #d4c9b8;
		background: #faf6ef;
		border-radius: 6px;
		cursor: pointer;
	}

	.die-token:hover {
		background: #fee;
		border-color: #c00;
	}

	.feeder-empty {
		font-size: 0.8rem;
		color: var(--text-muted);
		font-style: italic;
	}

	.feeder-add {
		display: flex;
		gap: 4px;
		margin-bottom: 4px;
	}

	.feeder-add-btn {
		font-size: 1rem;
		padding: 3px 10px;
		border: 1px dashed var(--border);
		background: transparent;
		border-radius: 4px;
		cursor: pointer;
	}

	.feeder-add-btn:hover {
		border-color: var(--accent);
		background: #fdf8ef;
	}

	.feeder-add-btn.choice {
		font-size: 0.85rem;
		padding: 3px 8px;
		border-style: dashed;
	}

	.clear-btn {
		font-size: 0.65rem;
		padding: 1px 6px;
		border: 1px solid #ddd;
		background: #f5f5f5;
		color: #999;
		border-radius: 3px;
		cursor: pointer;
	}

	.clear-btn:hover {
		background: #fee;
		color: #c00;
		border-color: #c00;
	}

	/* Card Tray */
	.tray-cards {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.tray-card {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 6px 10px;
		background: #eef7ff;
		border: 1px solid #c5ddf5;
		border-radius: 6px;
		font-size: 0.85rem;
	}

	.tray-bird-name {
		font-weight: 500;
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.tray-actions {
		display: flex;
		gap: 4px;
		flex-shrink: 0;
	}

	.tray-to-hand {
		font-size: 0.65rem;
		padding: 2px 6px;
		border: 1px solid #b0c4de;
		background: white;
		border-radius: 3px;
		cursor: pointer;
		color: #1565c0;
	}

	.tray-to-hand:hover {
		background: #e3f2fd;
		border-color: #1565c0;
	}

	.tray-empty {
		font-size: 0.8rem;
		color: var(--text-muted);
		font-style: italic;
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

	@media (max-width: 1000px) {
		.game-layout {
			grid-template-columns: 1fr;
		}

		.side-column {
			position: static;
			max-height: none;
		}
	}
</style>
