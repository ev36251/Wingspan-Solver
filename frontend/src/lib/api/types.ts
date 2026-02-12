// API types matching backend schemas

export interface FoodCost {
	items: string[];
	is_or: boolean;
	total: number;
}

export interface Bird {
	name: string;
	scientific_name: string;
	game_set: string;
	color: string;
	power_text: string;
	victory_points: number;
	nest_type: string;
	egg_limit: number;
	wingspan_cm: number | null;
	habitats: string[];
	food_cost: FoodCost;
	beak_direction: string;
	is_predator: boolean;
	is_flocking: boolean;
}

export interface BonusCard {
	name: string;
	game_sets: string[];
	condition_text: string;
	explanation_text: string | null;
	scoring_tiers: { min_count: number; max_count: number | null; points: number }[];
	is_per_bird: boolean;
}

export interface Goal {
	description: string;
	game_set: string;
	scoring: number[];
	reverse_description: string;
}

export interface BirdSlot {
	bird_name: string | null;
	egg_limit: number;
	eggs: number;
	cached_food: Record<string, number>;
	tucked_cards: number;
	victory_points: number;
	nest_type: string | null;
	cache_spendable: boolean;
}

export const NEST_ICONS: Record<string, string> = {
	platform: '\u2A09',   // ⨉ cross/sticks
	bowl: '\u2B58',       // ⭘ circle/bowl
	cavity: '\u25D1',     // ◑ half-filled circle
	ground: '\u2B22',     // ⬢ hexagon/ground
	wild: '\u2605',       // ★ star
};

export interface HabitatRow {
	habitat: string;
	slots: BirdSlot[];
	nectar_spent: number;
}

export interface FoodSupply {
	invertebrate: number;
	seed: number;
	fish: number;
	fruit: number;
	rodent: number;
	nectar: number;
}

export interface Player {
	name: string;
	board: HabitatRow[];
	food_supply: FoodSupply;
	hand: string[];
	bonus_cards: string[];
	action_cubes_remaining: number;
	unknown_hand_count: number;
	unknown_bonus_count: number;
}

export interface GameState {
	players: Player[];
	board_type: string;
	current_player_idx: number;
	current_round: number;
	turn_in_round: number;
	birdfeeder: { dice: (string | string[])[] };
	card_tray: { face_up: string[] };
	round_goals: string[];
	round_goal_scores: Record<number, Record<string, number>>;
	strict_rules_mode: boolean;
	deck_remaining: number;
}

export interface ScoreBreakdown {
	bird_vp: number;
	eggs: number;
	cached_food: number;
	tucked_cards: number;
	bonus_cards: number;
	round_goals: number;
	nectar: number;
	total: number;
}

export interface LegalMove {
	action_type: string;
	description: string;
	details: Record<string, unknown>;
}

export interface SolverRecommendation {
	rank: number;
	action_type: string;
	description: string;
	score: number;
	reasoning: string;
	details: Record<string, unknown>;
}

export interface ActionResult {
	success: boolean;
	action_type: string;
	message: string;
	food_gained: Record<string, number>;
	eggs_laid: number;
	cards_drawn: number;
	bird_played: string | null;
}

export interface SetupRecommendation {
	rank: number;
	score: number;
	birds_to_keep: string[];
	food_to_keep: Record<string, number>;
	bonus_card: string;
	reasoning: string;
}

export interface AfterResetRecommendation {
	rank: number;
	description: string;
	score: number;
	reasoning: string;
	details: Record<string, unknown>;
}

export interface AfterResetResponse {
	recommendations: AfterResetRecommendation[];
	reset_type: string;
	total_to_gain: number;
}

export interface MaxScoreResponse {
	max_possible_score: number;
	current_score: number;
	efficiency_pct: number;
	breakdown: Record<string, number>;
}

export interface MLEvalSummary {
	nn_wins: number;
	games: number;
	nn_mean_margin: number;
	nn_mean_score: number;
}

export interface MLPromotionGateSummary {
	nn_wins: number;
	games: number;
	nn_mean_margin: number;
	primary_opponent: string;
}

export interface MLLatestIterationSummary {
	iteration: string;
	stage: string;
	samples: number;
	mean_player_score: number;
	strict_game_fraction: number;
	strict_games: number;
	relaxed_games: number;
	eval: MLEvalSummary;
	promotion_gate: MLPromotionGateSummary;
}

export interface MLRunSummary {
	run_name: string;
	run_path: string;
	status: 'in_progress' | 'completed' | 'empty';
	current_stage: string;
	iterations_detected: number;
	iterations_completed: number;
	best_exists: boolean;
	latest_iteration: MLLatestIterationSummary | null;
	updated_at_epoch: number;
	updated_at_iso: string;
}

export interface MLRunsDashboardResponse {
	generated_at_epoch: number;
	reports_ml_dir: string;
	runs: MLRunSummary[];
	active_runs: MLRunSummary[];
	latest_completed_run: MLRunSummary | null;
}

// UI display helpers
export const HABITAT_COLORS: Record<string, string> = {
	forest: '#2d5016',
	grassland: '#8b7355',
	wetland: '#1a5276'
};

export const HABITAT_LABELS: Record<string, string> = {
	forest: 'Forest',
	grassland: 'Grassland',
	wetland: 'Wetland'
};

export const FOOD_ICONS: Record<string, string> = {
	invertebrate: '🪱',
	seed: '🌾',
	fish: '🐟',
	fruit: '🍒',
	rodent: '🐀',
	nectar: '🌺',
	wild: '⭐'
};

export const POWER_COLOR_STYLES: Record<string, string> = {
	white: 'bg-white text-gray-800 border-gray-300',
	brown: 'bg-amber-700 text-white',
	pink: 'bg-pink-300 text-gray-800',
	teal: 'bg-teal-400 text-gray-800',
	yellow: 'bg-yellow-300 text-gray-800',
	none: 'bg-gray-200 text-gray-600'
};
