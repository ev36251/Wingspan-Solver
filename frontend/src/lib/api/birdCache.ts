// Client-side bird cache: fetch all birds once, then filter locally so the
// search box is instant (no network round-trip per keystroke).
import { getAllBirds } from './client';
import type { Bird } from './types';

let _all: Bird[] | null = null;
let _loading: Promise<Bird[]> | null = null;

function norm(s: string): string {
	return s.toLowerCase().replace(/[^a-z0-9]/g, '');
}

/** Load (once) and cache the full bird list. */
export async function loadBirds(): Promise<Bird[]> {
	if (_all) return _all;
	if (!_loading) {
		_loading = getAllBirds(500, 0) // API caps limit at 500; all sets total <500
			.then((d) => {
				_all = d.birds;
				return _all;
			})
			.catch((e) => {
				_loading = null; // allow retry
				throw e;
			});
	}
	return _loading;
}

/** Instant local search over the cached list. Prefix matches rank first. */
export function searchBirdsLocal(query: string, limit = 12): Bird[] {
	if (!_all) return [];
	const q = norm(query);
	if (!q) return [];
	const scored: { bird: Bird; rank: number }[] = [];
	for (const bird of _all) {
		const n = norm(bird.name);
		let rank: number;
		if (n.startsWith(q)) rank = 0; // best: name starts with query
		else if (n.includes(q)) rank = 1; // contains query
		else if (norm(bird.scientific_name || '').includes(q)) rank = 2; // sci name
		else continue;
		scored.push({ bird, rank });
	}
	scored.sort(
		(a, b) =>
			a.rank - b.rank ||
			a.bird.name.length - b.bird.name.length ||
			a.bird.name.localeCompare(b.bird.name)
	);
	return scored.slice(0, limit).map((s) => s.bird);
}

export function birdsLoaded(): boolean {
	return _all !== null;
}
