<script lang="ts">
	import { FOOD_ICONS } from '$lib/api/types';
	import type { Bird } from '$lib/api/types';
	import { loadBirds, searchBirdsLocal, birdsLoaded } from '$lib/api/birdCache';
	import { createEventDispatcher, onMount } from 'svelte';

	export let placeholder = 'Search for a bird...';
	export let disabled = false;
	/** Keep focus after a pick so you can type the next bird immediately. */
	export let rapid = true;

	const dispatch = createEventDispatcher<{ select: Bird }>();

	let query = '';
	let results: Bird[] = [];
	let showDropdown = false;
	let selectedIndex = 0;
	let inputEl: HTMLInputElement;
	let ready = birdsLoaded();

	onMount(async () => {
		try {
			await loadBirds();
			ready = true;
			if (query) handleInput();
		} catch {
			ready = false;
		}
	});

	function handleInput() {
		// Instant, local — no network per keystroke.
		results = searchBirdsLocal(query, 12);
		selectedIndex = 0; // highlight the top match so Enter adds it immediately
		showDropdown = results.length > 0;
	}

	function selectBird(bird: Bird) {
		dispatch('select', bird);
		query = '';
		results = [];
		showDropdown = false;
		if (rapid) inputEl?.focus(); // ready for the next bird
	}

	function handleKeydown(e: KeyboardEvent) {
		if (!showDropdown) {
			if (e.key === 'ArrowDown' && query) handleInput();
			return;
		}
		if (e.key === 'ArrowDown') {
			e.preventDefault();
			selectedIndex = (selectedIndex + 1) % results.length;
		} else if (e.key === 'ArrowUp') {
			e.preventDefault();
			selectedIndex = (selectedIndex - 1 + results.length) % results.length;
		} else if (e.key === 'Enter' && selectedIndex >= 0) {
			e.preventDefault();
			selectBird(results[selectedIndex]);
		} else if (e.key === 'Escape') {
			showDropdown = false;
		}
	}

	function foodDisplay(bird: Bird): string[] {
		if (bird.food_cost.total === 0) return ['Free'];
		return bird.food_cost.items.map((f) => FOOD_ICONS[f] || f);
	}
</script>

<div class="bird-search">
	<input
		bind:this={inputEl}
		type="text"
		bind:value={query}
		on:input={handleInput}
		on:keydown={handleKeydown}
		on:focus={() => query && handleInput()}
		on:blur={() => setTimeout(() => (showDropdown = false), 150)}
		placeholder={ready ? placeholder : 'Loading birds…'}
		{disabled}
		autocomplete="off"
		spellcheck="false"
	/>

	{#if showDropdown}
		<ul class="dropdown">
			{#each results as bird, i (bird.name)}
				<li
					class:selected={i === selectedIndex}
					on:mousedown|preventDefault={() => selectBird(bird)}
					on:mouseenter={() => (selectedIndex = i)}
					role="option"
					aria-selected={i === selectedIndex}
				>
					<div class="bird-row">
						<span class="bird-name">{bird.name}</span>
						<span class="bird-vp">{bird.victory_points}</span>
					</div>
					<div class="bird-meta">
						<span class="bird-cost">
							{#if bird.food_cost.total === 0}
								<span class="free">Free</span>
							{:else}
								{#each foodDisplay(bird) as f, fi}
									{#if fi > 0}<span class="cost-sep">{bird.food_cost.is_or ? '/' : ''}</span>{/if}<span
										class="cost-icon">{f}</span
									>
								{/each}
							{/if}
						</span>
						<span class="bird-habitats">{bird.habitats.join(' · ')}</span>
						<span class="bird-dot color-{bird.color}" title={bird.color}></span>
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
		top: calc(100% + 4px);
		left: 0;
		right: 0;
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 10px;
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
		list-style: none;
		max-height: 320px;
		overflow-y: auto;
		z-index: 100;
		padding: 4px;
	}

	.dropdown li {
		padding: 7px 10px;
		cursor: pointer;
		border-radius: 7px;
	}

	.dropdown li.selected {
		background: color-mix(in srgb, var(--accent) 16%, transparent);
	}

	.bird-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 8px;
	}

	.bird-name {
		font-weight: 600;
		font-size: 0.9rem;
	}

	.bird-vp {
		font-size: 0.72rem;
		color: var(--accent);
		font-weight: 700;
		background: color-mix(in srgb, var(--accent) 14%, transparent);
		border-radius: 999px;
		padding: 1px 8px;
		flex-shrink: 0;
	}

	.bird-meta {
		display: flex;
		align-items: center;
		gap: 10px;
		font-size: 0.74rem;
		color: var(--text-muted);
		margin-top: 3px;
	}

	.cost-icon {
		font-size: 0.95rem;
	}
	.cost-sep {
		opacity: 0.6;
		margin: 0 1px;
	}
	.free {
		font-style: italic;
		opacity: 0.8;
	}
	.bird-habitats {
		text-transform: capitalize;
	}

	.bird-dot {
		width: 9px;
		height: 9px;
		border-radius: 50%;
		margin-left: auto;
		flex-shrink: 0;
		border: 1px solid rgba(0, 0, 0, 0.15);
	}
	.color-brown {
		background: #8b4513;
	}
	.color-white {
		background: #f3f3f3;
	}
	.color-pink {
		background: #d63384;
	}
	.color-teal {
		background: #0d9488;
	}
	.color-yellow {
		background: #ca8a04;
	}
	.color-none {
		background: #cfcfcf;
	}
</style>
