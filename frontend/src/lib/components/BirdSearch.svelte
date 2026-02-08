<script lang="ts">
	import { searchBirds } from '$lib/api/client';
	import { FOOD_ICONS } from '$lib/api/types';
	import type { Bird } from '$lib/api/types';
	import { createEventDispatcher } from 'svelte';

	export let placeholder = 'Search for a bird...';
	export let disabled = false;

	const dispatch = createEventDispatcher<{ select: Bird }>();

	let query = '';
	let results: Bird[] = [];
	let showDropdown = false;
	let selectedIndex = -1;
	let debounceTimer: ReturnType<typeof setTimeout>;

	function handleInput() {
		clearTimeout(debounceTimer);
		selectedIndex = -1;
		if (query.length < 2) {
			results = [];
			showDropdown = false;
			return;
		}
		debounceTimer = setTimeout(async () => {
			try {
				const data = await searchBirds(query, 10);
				results = data.birds;
				showDropdown = results.length > 0;
			} catch {
				results = [];
			}
		}, 200);
	}

	function selectBird(bird: Bird) {
		dispatch('select', bird);
		query = '';
		results = [];
		showDropdown = false;
	}

	function handleKeydown(e: KeyboardEvent) {
		if (!showDropdown) return;
		if (e.key === 'ArrowDown') {
			e.preventDefault();
			selectedIndex = Math.min(selectedIndex + 1, results.length - 1);
		} else if (e.key === 'ArrowUp') {
			e.preventDefault();
			selectedIndex = Math.max(selectedIndex - 1, 0);
		} else if (e.key === 'Enter' && selectedIndex >= 0) {
			e.preventDefault();
			selectBird(results[selectedIndex]);
		} else if (e.key === 'Escape') {
			showDropdown = false;
		}
	}

	function foodDisplay(bird: Bird): string {
		if (bird.food_cost.total === 0) return 'Free';
		return bird.food_cost.items.map(f => FOOD_ICONS[f] || f).join(
			bird.food_cost.is_or ? ' / ' : ' '
		);
	}
</script>

<div class="bird-search">
	<input
		type="text"
		bind:value={query}
		on:input={handleInput}
		on:keydown={handleKeydown}
		on:blur={() => setTimeout(() => showDropdown = false, 200)}
		{placeholder}
		{disabled}
	/>

	{#if showDropdown}
		<ul class="dropdown">
			{#each results as bird, i}
				<li
					class:selected={i === selectedIndex}
					on:mousedown|preventDefault={() => selectBird(bird)}
					on:mouseenter={() => selectedIndex = i}
					role="option"
					aria-selected={i === selectedIndex}
				>
					<div class="bird-row">
						<span class="bird-name">{bird.name}</span>
						<span class="bird-vp">{bird.victory_points}VP</span>
					</div>
					<div class="bird-meta">
						<span class="bird-cost">{foodDisplay(bird)}</span>
						<span class="bird-habitats">{bird.habitats.join(', ')}</span>
						<span class="bird-color color-{bird.color}">{bird.color}</span>
					</div>
				</li>
			{/each}
		</ul>
	{/if}
</div>

<style>
	.bird-search {
		position: relative;
		width: 100%;
	}

	.bird-search input {
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
		max-height: 300px;
		overflow-y: auto;
		z-index: 100;
	}

	.dropdown li {
		padding: 8px 12px;
		cursor: pointer;
		border-bottom: 1px solid #f0f0f0;
	}

	.dropdown li:last-child {
		border-bottom: none;
	}

	.dropdown li.selected,
	.dropdown li:hover {
		background: #f5f0e8;
	}

	.bird-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.bird-name {
		font-weight: 500;
	}

	.bird-vp {
		font-size: 0.8rem;
		color: var(--accent);
		font-weight: 600;
	}

	.bird-meta {
		display: flex;
		gap: 8px;
		font-size: 0.75rem;
		color: var(--text-muted);
		margin-top: 2px;
	}

	.color-brown { color: #8B4513; }
	.color-white { color: #888; }
	.color-pink { color: #D63384; }
	.color-teal { color: #0D9488; }
	.color-yellow { color: #CA8A04; }
	.color-none { color: #999; }
</style>
