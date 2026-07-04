<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import type { GameState } from '$lib/api/types';
	import { importScreenshot } from '$lib/api/client';
	import type { ScreenshotImportResult } from '$lib/api/client';
	import { fileToImagePayload } from '$lib/imageUpload';

	export let state: GameState;
	export let activePlayerIdx = 0;
	export let disabled = false;

	const dispatch = createEventDispatcher<{ apply: GameState }>();

	let expanded = false;
	let files: File[] = [];
	let previews: string[] = [];
	let notes = '';
	let target: 'auto' | number = 'auto';
	let reading = false;
	let error = '';
	let result: ScreenshotImportResult | null = null;

	function onFilesChosen(e: Event) {
		const input = e.target as HTMLInputElement;
		if (!input.files) return;
		for (const f of Array.from(input.files)) {
			if (files.length >= 8) break;
			files = [...files, f];
			previews = [...previews, URL.createObjectURL(f)];
		}
		input.value = '';
		result = null;
	}

	function removeFile(i: number) {
		URL.revokeObjectURL(previews[i]);
		files = files.filter((_, j) => j !== i);
		previews = previews.filter((_, j) => j !== i);
	}

	async function read() {
		if (!files.length || reading) return;
		reading = true;
		error = '';
		result = null;
		try {
			const images = await Promise.all(files.map(fileToImagePayload));
			result = await importScreenshot({
				images,
				notes: notes.trim() || null,
				current_state: state,
				target_player_idx: target === 'auto' ? null : target
			});
		} catch (e) {
			error = e instanceof Error ? e.message : 'Screenshot import failed';
		} finally {
			reading = false;
		}
	}

	function apply() {
		if (!result?.proposed) return;
		dispatch('apply', result.proposed);
		clear();
		expanded = false;
	}

	function clear() {
		previews.forEach((u) => URL.revokeObjectURL(u));
		files = [];
		previews = [];
		result = null;
		error = '';
	}
</script>

<div class="sidebar-panel card import-panel">
	<div class="panel-header-row">
		<h4 class="panel-title">Import from Screenshot</h4>
		<button class="toggle-btn" on:click={() => (expanded = !expanded)}>
			{expanded ? 'Hide' : 'Open'}
		</button>
	</div>

	{#if expanded}
		<p class="hint">
			Screenshot the game (digital app or a photo of the table), and the board,
			hands, feeder, and tray are read into the tracker for you to review.
		</p>

		<label class="file-drop">
			<input type="file" accept="image/*" multiple on:change={onFilesChosen} disabled={reading} />
			{files.length ? `${files.length} screenshot${files.length > 1 ? 's' : ''} selected — add more?` : 'Choose screenshot(s)…'}
		</label>

		{#if previews.length}
			<div class="thumbs">
				{#each previews as url, i}
					<div class="thumb">
						<img src={url} alt="screenshot {i + 1}" />
						<button class="remove-btn" on:click={() => removeFile(i)} disabled={reading}>x</button>
					</div>
				{/each}
			</div>
		{/if}

		<label class="field">
			Whose board is shown?
			<select bind:value={target} disabled={reading}>
				<option value="auto">Everything visible (auto)</option>
				{#each state.players as p, i}
					<option value={i}>Only {p.name}'s board{i === activePlayerIdx ? ' (active)' : ''}</option>
				{/each}
			</select>
		</label>

		<label class="field">
			Notes for the reader (optional)
			<input
				type="text"
				bind:value={notes}
				placeholder="e.g. I'm the bottom player; feeder was just rerolled"
				disabled={reading}
			/>
		</label>

		<button class="primary read-btn" on:click={read} disabled={!files.length || reading || disabled}>
			{#if reading}
				<span class="btn-spinner" aria-hidden="true"></span> Reading… (~30–60 s)
			{:else}
				Read screenshots
			{/if}
		</button>

		{#if error}
			<div class="import-error">{error}</div>
		{/if}

		{#if result}
			<div class="result">
				<div class="result-head">Reading complete.</div>
				{#if result.warnings.length}
					<div class="msg-block warn">
						<strong>Check these:</strong>
						<ul>
							{#each result.warnings as w}<li>{w}</li>{/each}
						</ul>
					</div>
				{/if}
				{#if result.uncertainties.length}
					<div class="msg-block unsure">
						<strong>The reader wasn't sure about:</strong>
						<ul>
							{#each result.uncertainties as u}<li>{u}</li>{/each}
						</ul>
					</div>
				{/if}
				<div class="result-actions">
					<button class="primary" on:click={apply}>Apply to board</button>
					<button on:click={() => (result = null)}>Discard</button>
				</div>
				<p class="hint">
					Applying replaces what was read on this screen only — then review the
					board and press <strong>Save State</strong>.
				</p>
			</div>
		{/if}
	{/if}
</div>

<style>
	.import-panel {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.panel-header-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}
	.panel-title {
		margin: 0;
	}
	.toggle-btn {
		font-size: 0.8rem;
		padding: 0.2rem 0.6rem;
	}
	.hint {
		font-size: 0.78rem;
		color: var(--text-muted);
		margin: 0;
	}
	.file-drop {
		display: block;
		border: 1.5px dashed var(--border-strong);
		border-radius: var(--radius-sm);
		padding: 0.7rem;
		text-align: center;
		font-size: 0.85rem;
		cursor: pointer;
		color: var(--text-muted);
	}
	.file-drop:hover {
		border-color: var(--accent);
		color: var(--accent-strong);
	}
	.file-drop input {
		display: none;
	}
	.thumbs {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
	}
	.thumb {
		position: relative;
		width: 72px;
		height: 48px;
	}
	.thumb img {
		width: 100%;
		height: 100%;
		object-fit: cover;
		border-radius: 4px;
		border: 1px solid var(--border);
	}
	.thumb .remove-btn {
		position: absolute;
		top: -6px;
		right: -6px;
		width: 18px;
		height: 18px;
		line-height: 1;
		padding: 0;
		border-radius: 50%;
		font-size: 0.7rem;
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
		font-size: 0.8rem;
		color: var(--text-muted);
	}
	.field select,
	.field input {
		font-size: 0.85rem;
		padding: 0.3rem;
	}
	.read-btn {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: 0.4rem;
	}
	.btn-spinner {
		width: 12px;
		height: 12px;
		border: 2px solid rgba(255, 255, 255, 0.4);
		border-top-color: #fff;
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
	.import-error {
		background: var(--error-bg);
		color: var(--error-text);
		border-radius: 4px;
		padding: 0.5rem;
		font-size: 0.8rem;
		white-space: pre-wrap;
	}
	.result {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.result-head {
		font-weight: 600;
		font-size: 0.85rem;
	}
	.msg-block {
		border-radius: 4px;
		padding: 0.5rem;
		font-size: 0.78rem;
	}
	.msg-block ul {
		margin: 0.3rem 0 0;
		padding-left: 1.1rem;
	}
	.msg-block.warn {
		background: #fff8e1;
		color: #7a5c00;
	}
	.msg-block.unsure {
		background: #e8f0fe;
		color: #1a4b8f;
	}
	.result-actions {
		display: flex;
		gap: 0.5rem;
	}
</style>
