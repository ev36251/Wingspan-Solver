// API client for the Wingspan Solver backend

const BASE_URL = '/api';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
	const resp = await fetch(`${BASE_URL}${path}`, {
		headers: { 'Content-Type': 'application/json' },
		...options
	});
	if (!resp.ok) {
		const error = await resp.json().catch(() => ({ detail: resp.statusText }));
		throw new Error(error.detail || `API error: ${resp.status}`);
	}
	return resp.json();
}

// --- Data endpoints ---

export async function searchBirds(query: string, limit = 20) {
	return request<{ birds: import('./types').Bird[]; total: number }>(
		`/birds?search=${encodeURIComponent(query)}&limit=${limit}`
	);
}

export async function getAllBirds(limit = 50, offset = 0) {
	return request<{ birds: import('./types').Bird[]; total: number }>(
		`/birds?limit=${limit}&offset=${offset}`
	);
}

export async function getBird(name: string) {
	return request<import('./types').Bird>(`/birds/${encodeURIComponent(name)}`);
}

export async function getBonusCards() {
	return request<{ bonus_cards: import('./types').BonusCard[]; total: number }>('/bonus-cards');
}

export async function getGoals() {
	return request<{ goals: import('./types').Goal[]; total: number }>('/goals');
}

// --- Game endpoints ---

export async function createGame(
	playerNames: string[],
	roundGoals?: string[],
	boardType: 'base' | 'oceania' = 'oceania'
) {
	return request<{ game_id: string; state: import('./types').GameState }>('/games', {
		method: 'POST',
		body: JSON.stringify({ player_names: playerNames, board_type: boardType, round_goals: roundGoals })
	});
}

export async function getGame(gameId: string) {
	return request<import('./types').GameState>(`/games/${gameId}`);
}

export async function updateGameState(gameId: string, state: import('./types').GameState) {
	return request<import('./types').GameState>(`/games/${gameId}/state`, {
		method: 'PUT',
		body: JSON.stringify(state)
	});
}

export async function getLegalMoves(gameId: string) {
	return request<{ moves: import('./types').LegalMove[]; total: number }>(
		`/games/${gameId}/legal-moves`
	);
}

export async function getScores(gameId: string) {
	return request<{ scores: Record<string, import('./types').ScoreBreakdown> }>(
		`/games/${gameId}/score`
	);
}

// --- Action endpoints ---
// All action requests support companion mode (companion: true): the engine
// applies the full move (powers, turn advancement) but never invents hidden
// card identities — deck draws become face-down counts and the tray is left
// short for the user to enter the real revealed cards.

export interface PlayBirdParams {
	bird_name: string;
	habitat: string;
	food_payment?: Record<string, number>;
	egg_payment_slots?: [string, number][] | null;
	target_slot?: number | null;
	play_on_top?: boolean;
	play_on_top_discard?: boolean;
	hand_tuck_payment?: number;
	companion?: boolean;
}

export async function playBird(gameId: string, params: PlayBirdParams) {
	return request<import('./types').ActionResult>(`/games/${gameId}/play-bird`, {
		method: 'POST',
		body: JSON.stringify(params)
	});
}

export interface GainFoodParams {
	food_choices: string[];
	bonus_count?: number;
	reset_bonus?: boolean;
	prefer_nectar?: boolean;
	companion?: boolean;
}

export async function gainFood(gameId: string, params: GainFoodParams) {
	return request<import('./types').ActionResult>(`/games/${gameId}/gain-food`, {
		method: 'POST',
		body: JSON.stringify(params)
	});
}

export interface LayEggsParams {
	egg_distribution: Record<string, number>; // "habitat:slot_idx" -> count
	bonus_count?: number;
	prefer_nectar?: boolean;
	companion?: boolean;
}

export async function layEggs(gameId: string, params: LayEggsParams) {
	return request<import('./types').ActionResult>(`/games/${gameId}/lay-eggs`, {
		method: 'POST',
		body: JSON.stringify(params)
	});
}

export interface DrawCardsParams {
	from_tray_indices?: number[];
	from_deck_count?: number;
	bonus_count?: number;
	reset_bonus?: boolean;
	prefer_nectar?: boolean;
	companion?: boolean;
}

export async function drawCards(gameId: string, params: DrawCardsParams) {
	return request<import('./types').ActionResult>(`/games/${gameId}/draw-cards`, {
		method: 'POST',
		body: JSON.stringify(params)
	});
}

// --- Setup endpoints ---

export async function analyzeSetup(
	birdNames: string[],
	bonusCardNames: string[],
	roundGoals: string[] = [],
	trayCards: string[] = [],
	turnOrder: number = 1,
	numPlayers: number = 2
) {
	return request<{
		recommendations: import('./types').SetupRecommendation[];
		total_combinations: number;
	}>('/setup/analyze', {
		method: 'POST',
		body: JSON.stringify({
			bird_names: birdNames,
			bonus_card_names: bonusCardNames,
			round_goals: roundGoals,
			tray_cards: trayCards,
			turn_order: turnOrder,
			num_players: numPlayers
		})
	});
}

// --- Solver endpoints ---

export async function solveHeuristic(gameId: string, playerIdx?: number) {
	const params = playerIdx !== undefined ? `?player_idx=${playerIdx}` : '';
	return request<{
		recommendations: import('./types').SolverRecommendation[];
		evaluation_time_ms: number;
		player_name: string;
		feeder_reroll_available?: boolean;
	}>(`/games/${gameId}/solve/heuristic${params}`, { method: 'POST' });
}

export async function solveAdvisor(
	gameId: string,
	playerIdx?: number,
	activeSets?: string[]
) {
	return request<import('./types').AdvisorResponse>(
		`/games/${gameId}/solve/advisor`,
		{
			method: 'POST',
			body: JSON.stringify({
				player_idx: playerIdx,
				active_sets: activeSets && activeSets.length ? activeSets : null
			})
		}
	);
}

export async function getMaxScore(gameId: string, playerName?: string) {
	const params = playerName ? `?player_name=${encodeURIComponent(playerName)}` : '';
	return request<import('./types').MaxScoreResponse>(
		`/games/${gameId}/solve/max-score${params}`,
		{ method: 'POST' }
	);
}

export async function solveAfterReset(
	gameId: string,
	resetType: 'feeder' | 'tray',
	newFeederDice?: (string | string[])[],
	newTrayCards?: string[],
	totalToGain?: number
) {
	return request<import('./types').AfterResetResponse>(
		`/games/${gameId}/solve/after-reset`,
		{
			method: 'POST',
			body: JSON.stringify({
				reset_type: resetType,
				new_feeder_dice: newFeederDice || [],
				new_tray_cards: newTrayCards || [],
				total_to_gain: totalToGain || 0
			})
		}
		);
}

// --- ML diagnostics endpoints ---

export async function getMLRunsDashboard(limit = 20, includeIterations = false) {
	return request<import('./types').MLRunsDashboardResponse>(
		`/ml/runs/dashboard?limit=${limit}&include_iterations=${includeIterations ? 'true' : 'false'}`
	);
}

export async function getMLRunDiagnostics(runName: string, includeIterations = true) {
	return request<import('./types').MLRunSummary & { iterations?: import('./types').MLLatestIterationSummary[] }>(
		`/ml/runs/${encodeURIComponent(runName)}/diagnostics?include_iterations=${includeIterations ? 'true' : 'false'}`
	);
}
