<script lang="ts">
	import type { Player } from '$lib/api/types';
	import { FOOD_ICONS, HABITAT_LABELS } from '$lib/api/types';

	export let player: Player;
	export let isCurrentPlayer = false;

	function totalCachedFood(slot: { cached_food: Record<string, number> }): number {
		return Object.values(slot.cached_food).reduce((a, b) => a + b, 0);
	}

	function cachedFoodDisplay(cached: Record<string, number>): string {
		return Object.entries(cached)
			.filter(([, v]) => v > 0)
			.map(([k, v]) => `${v}${FOOD_ICONS[k] || k}`)
			.join(' ');
	}

	function foodSupplyDisplay(supply: Player['food_supply']): string[] {
		const items: string[] = [];
		if (supply.invertebrate) items.push(`${supply.invertebrate}${FOOD_ICONS.invertebrate}`);
		if (supply.seed) items.push(`${supply.seed}${FOOD_ICONS.seed}`);
		if (supply.fish) items.push(`${supply.fish}${FOOD_ICONS.fish}`);
		if (supply.fruit) items.push(`${supply.fruit}${FOOD_ICONS.fruit}`);
		if (supply.rodent) items.push(`${supply.rodent}${FOOD_ICONS.rodent}`);
		if (supply.nectar) items.push(`${supply.nectar}${FOOD_ICONS.nectar}`);
		return items;
	}
</script>

<div class="game-board card" class:current-player={isCurrentPlayer}>
	<div class="board-header">
		<h3>{player.name}{isCurrentPlayer ? ' (Current)' : ''}</h3>
		<div class="player-stats">
			<span title="Action cubes remaining">
				{player.action_cubes_remaining} actions
			</span>
			<span title="Cards in hand">
				{player.hand.length + (player.unknown_hand_count || 0)} cards{#if player.unknown_hand_count > 0} ({player.hand.length} known){/if}
			</span>
		</div>
	</div>

	<div class="food-supply">
		{#each foodSupplyDisplay(player.food_supply) as item}
			<span class="food-token">{item}</span>
		{:else}
			<span class="no-food">No food</span>
		{/each}
	</div>

	<div class="habitats">
		{#each player.board as row}
			<div class="habitat-row">
				<div class="habitat-label habitat-{row.habitat}">
					{HABITAT_LABELS[row.habitat] || row.habitat}
					{#if row.nectar_spent > 0}
						<span class="nectar-badge">{row.nectar_spent}{FOOD_ICONS.nectar}</span>
					{/if}
				</div>
				<div class="slots">
					{#each row.slots as slot, i}
						<div class="slot" class:occupied={slot.bird_name}>
							{#if slot.bird_name}
								<div class="bird-name" title={slot.bird_name}>
									{slot.bird_name}
								</div>
								<div class="slot-tokens">
									{#if slot.eggs > 0}
										<span class="token eggs" title="Eggs">{slot.eggs} egg{slot.eggs !== 1 ? 's' : ''}</span>
									{/if}
									{#if totalCachedFood(slot) > 0}
										<span class="token cached" title="Cached food">{cachedFoodDisplay(slot.cached_food)}</span>
									{/if}
									{#if slot.tucked_cards > 0}
										<span class="token tucked" title="Tucked cards">{slot.tucked_cards} tucked</span>
									{/if}
								</div>
							{:else}
								<div class="empty-slot">Slot {i + 1}</div>
							{/if}
						</div>
					{/each}
				</div>
			</div>
		{/each}
	</div>

	{#if player.bonus_cards.length > 0 || (player.unknown_bonus_count || 0) > 0}
		<div class="bonus-cards">
			<span class="label">Bonus:</span>
			{#each player.bonus_cards as bc}
				<span class="bonus-badge">{bc}</span>
			{/each}
			{#if (player.unknown_bonus_count || 0) > 0}
				<span class="bonus-badge unknown">+{player.unknown_bonus_count} unknown</span>
			{/if}
		</div>
	{/if}
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

	.food-supply {
		display: flex;
		gap: 8px;
		margin-bottom: 12px;
		flex-wrap: wrap;
	}

	.food-token {
		font-size: 0.9rem;
		background: #f5f0e8;
		padding: 2px 8px;
		border-radius: 4px;
	}

	.no-food {
		font-size: 0.8rem;
		color: var(--text-muted);
		font-style: italic;
	}

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
		justify-content: center;
	}

	.slot.occupied {
		background: #fefdf8;
	}

	.bird-name {
		font-weight: 500;
		font-size: 0.7rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.slot-tokens {
		display: flex;
		flex-wrap: wrap;
		gap: 3px;
		margin-top: 4px;
	}

	.token {
		font-size: 0.65rem;
		padding: 1px 4px;
		border-radius: 3px;
	}

	.token.eggs {
		background: #e8f5e9;
		color: #2e7d32;
	}

	.token.cached {
		background: #fff3e0;
		color: #e65100;
	}

	.token.tucked {
		background: #e3f2fd;
		color: #1565c0;
	}

	.empty-slot {
		color: var(--text-muted);
		font-size: 0.7rem;
		text-align: center;
	}

	.bonus-cards {
		display: flex;
		gap: 6px;
		margin-top: 10px;
		align-items: center;
		flex-wrap: wrap;
	}

	.label {
		font-size: 0.8rem;
		color: var(--text-muted);
	}

	.bonus-badge {
		font-size: 0.7rem;
		background: #f3e5f5;
		color: #6a1b9a;
		padding: 2px 8px;
		border-radius: 4px;
	}

	.bonus-badge.unknown {
		background: #e8e8e8;
		color: #666;
		font-style: italic;
	}
</style>
