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

export async function createGame(playerNames: string[], roundGoals?: string[]) {
	return request<{ game_id: string; state: import('./types').GameState }>('/games', {
		method: 'POST',
		body: JSON.stringify({ player_names: playerNames, board_type: 'oceania', round_goals: roundGoals })
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

export async function playBird(
	gameId: string,
	birdName: string,
	habitat: string,
	foodPayment: Record<string, number>
) {
	return request<import('./types').ActionResult>(`/games/${gameId}/play-bird`, {
		method: 'POST',
		body: JSON.stringify({ bird_name: birdName, habitat, food_payment: foodPayment })
	});
}

export async function gainFood(gameId: string, foodChoices: string[]) {
	return request<import('./types').ActionResult>(`/games/${gameId}/gain-food`, {
		method: 'POST',
		body: JSON.stringify({ food_choices: foodChoices })
	});
}

export async function layEggs(gameId: string, distribution: Record<string, number>) {
	return request<import('./types').ActionResult>(`/games/${gameId}/lay-eggs`, {
		method: 'POST',
		body: JSON.stringify({ egg_distribution: distribution })
	});
}

export async function drawCards(gameId: string, trayIndices: number[], deckCount: number) {
	return request<import('./types').ActionResult>(`/games/${gameId}/draw-cards`, {
		method: 'POST',
		body: JSON.stringify({ from_tray_indices: trayIndices, from_deck_count: deckCount })
	});
}

// --- Setup endpoints ---

export async function analyzeSetup(
	birdNames: string[],
	bonusCardNames: string[],
	roundGoals: string[] = []
) {
	return request<{
		recommendations: import('./types').SetupRecommendation[];
		total_combinations: number;
	}>('/setup/analyze', {
		method: 'POST',
		body: JSON.stringify({
			bird_names: birdNames,
			bonus_card_names: bonusCardNames,
			round_goals: roundGoals
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
	}>(`/games/${gameId}/solve/heuristic${params}`, { method: 'POST' });
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
