"""Screenshot → game state extraction using Claude vision.

Pipeline:
  1. `extract_from_images()` sends the screenshots to Claude (vision +
     structured outputs) and gets back an `ExtractedGameState` — a loose,
     "only what's visible" reading with raw card-name strings.
  2. `build_proposed_state()` is a pure function (no API access) that fuzzy-
     matches every name against the loaded registries and merges the reading
     onto the caller's current `GameStateSchema`, returning a complete
     proposed state plus human-readable warnings for anything that didn't
     match cleanly.

The `anthropic` package and an ANTHROPIC_API_KEY are only required for step 1;
step 2 is fully testable offline.
"""

from __future__ import annotations

import base64
import difflib
import os
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field

from backend.api.schemas import (
    BirdSlotSchema,
    FoodSupplySchema,
    GameStateSchema,
    HabitatRowSchema,
    PlayerSchema,
)
from backend.data.registries import (
    get_bird_registry,
    get_bonus_registry,
    get_goal_registry,
)

VISION_MODEL = "claude-opus-4-8"
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # decoded size per image
ALLOWED_MEDIA_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}

HABITATS = ("forest", "grassland", "wetland")
SLOTS_PER_ROW = 5
FOOD_TYPES = ("invertebrate", "seed", "fish", "fruit", "rodent", "nectar")

# Common misreadings / synonyms for food tokens as they appear on screen.
_FOOD_SYNONYMS = {
    "invertebrate": "invertebrate", "worm": "invertebrate", "grub": "invertebrate",
    "bug": "invertebrate", "caterpillar": "invertebrate", "insect": "invertebrate",
    "seed": "seed", "seeds": "seed", "wheat": "seed", "grain": "seed",
    "fish": "fish",
    "fruit": "fruit", "berry": "fruit", "berries": "fruit", "cherry": "fruit",
    "cherries": "fruit",
    "rodent": "rodent", "mouse": "rodent", "mice": "rodent", "rat": "rodent",
    "nectar": "nectar", "flower": "nectar",
}


class ScreenshotImportError(Exception):
    """User-facing error (missing dependency/key, bad image, API failure)."""


# --- Extraction models (what Claude returns; structured output) ---
# No dicts with free-form keys: strict JSON schemas need enumerable properties.

class XBoardBird(BaseModel):
    """One bird on the board, left-to-right within its habitat row."""
    bird_name: str
    eggs: int = 0
    cached_food: list[str] = Field(
        default_factory=list,
        description="One entry per cached food token on this bird, e.g. ['fish', 'fish']",
    )
    tucked_cards: int = 0


class XHabitatRow(BaseModel):
    habitat: Literal["forest", "grassland", "wetland"]
    birds: list[XBoardBird]
    nectar_spent: int | None = None


class XFoodSupply(BaseModel):
    invertebrate: int | None = None
    seed: int | None = None
    fish: int | None = None
    fruit: int | None = None
    rodent: int | None = None
    nectar: int | None = None


class XPlayer(BaseModel):
    name: str | None = None
    board: list[XHabitatRow] | None = None
    food_supply: XFoodSupply | None = None
    hand: list[str] | None = None
    bonus_cards: list[str] | None = None
    action_cubes_remaining: int | None = None


class ExtractedGameState(BaseModel):
    players: list[XPlayer] = Field(default_factory=list)
    birdfeeder_dice: list[str] | None = Field(
        default=None,
        description="Dice in the feeder; two-option faces as 'a/b', e.g. 'invertebrate/seed'",
    )
    card_tray: list[str] | None = None
    current_round: int | None = None
    turn_in_round: int | None = None
    deck_remaining: int | None = None
    round_goals: list[str] | None = None
    uncertainties: list[str] = Field(
        default_factory=list,
        description="Anything hard to read or ambiguous in the screenshots",
    )


class ExtractedDraft(BaseModel):
    """The start-of-game draft screen: dealt cards to feed the setup advisor."""

    dealt_birds: list[str] = Field(default_factory=list)
    bonus_cards: list[str] = Field(default_factory=list)
    round_goals: list[str] | None = None
    tray_birds: list[str] | None = None
    uncertainties: list[str] = Field(
        default_factory=list,
        description="Anything hard to read or ambiguous in the screenshots",
    )


# --- Prompt construction ---

def _card_name_lists() -> str:
    bird_names = ", ".join(b.name for b in get_bird_registry().all_birds)
    bonus_names = ", ".join(c.name for c in get_bonus_registry().all_cards)
    goal_descs = "; ".join(g.description for g in get_goal_registry().all_goals)
    return (
        f"VALID BIRD NAMES: {bird_names}\n\n"
        f"VALID BONUS CARDS: {bonus_names}\n\n"
        f"VALID ROUND GOALS: {goal_descs}"
    )


def build_system_prompt() -> str:
    return f"""You are reading screenshots of a Wingspan board game (the digital app or a photo of a physical table) to populate a game-state tracker.

Report ONLY what is clearly visible in the images. Anything not visible must be omitted (null). Never invent hidden information: face-down cards, the opponent's hand, or deck contents. If a specific reading is doubtful, still give your best reading and add a short note to `uncertainties`.

Extract, when visible:
- Each player's board: the birds in each habitat row (forest / grassland / wetland) in left-to-right order, and for each bird its egg count, tucked-card count, and cached food tokens (one list entry per token).
- Each player's personal food supply: invertebrate (worm/grub), seed, fish, fruit (berry), rodent (mouse), nectar (flower).
- The visible hand of bird cards and bonus cards (usually only "your" hand at the bottom of the screen).
- The birdfeeder dice currently available. Faces: invertebrate, seed, fish, fruit, rodent, nectar; write a two-option face as "a/b" (e.g. "invertebrate/seed").
- The face-up card tray (usually 3 bird cards).
- Round number (1-4), turn within the round, each player's remaining action cubes, and cards remaining in the deck.
- The round goals if the goal board is visible.
- Nectar spent per habitat row (Oceania boards track spent nectar beside each row).

Cards in a fanned hand often overlap: usually only each card's title bar is visible, and the title is all you need — never skip a card just because its body is hidden, and never guess a fully hidden card.

Use the exact official card names from the lists below whenever you can identify a card. If a name is partly obscured, give your best reading of the visible text — it will be fuzzy-matched.

{_card_name_lists()}"""


def build_draft_system_prompt() -> str:
    return f"""You are reading screenshots of the START-OF-GAME DRAFT screen in Wingspan (the "choose things to keep" step: dealt bird cards + bonus cards) to feed a draft advisor.

Report ONLY what is clearly visible. Extract:
- dealt_birds: the dealt bird cards (usually 5), left to right. Cards may overlap — the title bar is enough to identify a card.
- bonus_cards: the dealt bonus cards (usually 2). In the digital app these can appear as small thumbnails near the top of the screen; only report ones whose name you can actually read, and add a note to `uncertainties` if they are too small to read.
- round_goals: the four round-goal tiles if visible (often small icon tiles in a corner, in round order 1→4). Give each goal's text, or your best short description of the icon (e.g. "birds in the forest", "eggs in cup nests").
- tray_birds: the face-up card tray birds, if shown.

Use the exact official names from the lists below whenever you can identify a card. If a name is partly obscured, give your best reading — it will be fuzzy-matched.

{_card_name_lists()}"""


def _user_text(notes: str | None, current: GameStateSchema | None) -> str:
    parts = ["Extract the current Wingspan game state from these screenshots."]
    if current is not None:
        names = ", ".join(p.name for p in current.players)
        parts.append(
            f"The tracked game has {len(current.players)} players in this order: {names} "
            f"(board type: {current.board_type}). Match the extracted players to these "
            "names when possible; in the digital app 'your' board is at the bottom."
        )
    if notes:
        parts.append(f"User notes: {notes}")
    return "\n".join(parts)


# --- Step 1: the vision API call ---

def decode_and_check_image(media_type: str, data_b64: str) -> None:
    """Validate one uploaded image payload; raises ScreenshotImportError."""
    if media_type not in ALLOWED_MEDIA_TYPES:
        raise ScreenshotImportError(
            f"Unsupported image type '{media_type}' — use PNG, JPEG, WebP, or GIF."
        )
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except Exception:
        raise ScreenshotImportError("An image payload was not valid base64.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ScreenshotImportError(
            f"An image is {len(raw) / 1e6:.1f} MB after decoding — max is "
            f"{MAX_IMAGE_BYTES / 1e6:.0f} MB. Downscale the screenshot and retry."
        )


def _call_vision(
    images: list[tuple[str, str]],
    system_text: str,
    user_text: str,
    output_model: type[BaseModel],
):
    """Shared plumbing: validate images, call Claude, return the parsed model."""
    try:
        import anthropic
    except ImportError:
        raise ScreenshotImportError(
            "The 'anthropic' package is not installed on the server. "
            "Run: pip install anthropic — then restart the backend."
        )

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise ScreenshotImportError(
            "ANTHROPIC_API_KEY is not set. Get a key at console.anthropic.com, then "
            "start the backend with it, e.g.: ANTHROPIC_API_KEY=sk-... uvicorn backend.main:app"
        )

    if not images:
        raise ScreenshotImportError("No images were provided.")
    if len(images) > MAX_IMAGES:
        raise ScreenshotImportError(f"Too many images — send at most {MAX_IMAGES}.")
    for media_type, data_b64 in images:
        decode_and_check_image(media_type, data_b64)

    content: list[dict] = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data_b64},
        }
        for media_type, data_b64 in images
    ]
    content.append({"type": "text", "text": user_text})

    client = anthropic.Anthropic()
    try:
        response = client.messages.parse(
            model=VISION_MODEL,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": system_text,
                    # The card-name lists are identical across imports: cache them.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": content}],
            output_format=output_model,
        )
    except anthropic.AuthenticationError:
        raise ScreenshotImportError(
            "The Anthropic API rejected the key (401). Check ANTHROPIC_API_KEY."
        )
    except anthropic.RateLimitError:
        raise ScreenshotImportError(
            "Anthropic API rate limit hit (429) — wait a moment and retry."
        )
    except anthropic.APIStatusError as e:
        raise ScreenshotImportError(f"Anthropic API error ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        raise ScreenshotImportError(
            "Could not reach the Anthropic API — check the server's network connection."
        )

    parsed = response.parsed_output
    if parsed is None:
        raise ScreenshotImportError("The model returned no structured reading — retry.")
    return parsed


def extract_from_images(
    images: list[tuple[str, str]],
    notes: str | None = None,
    current: GameStateSchema | None = None,
) -> ExtractedGameState:
    """Send (media_type, base64_data) screenshots to Claude and parse the reading."""
    return _call_vision(
        images, build_system_prompt(), _user_text(notes, current), ExtractedGameState
    )


def extract_draft_from_images(
    images: list[tuple[str, str]],
    notes: str | None = None,
) -> ExtractedDraft:
    """Read the start-of-game draft screen (dealt birds + bonus cards)."""
    parts = ["Extract the dealt draft cards from these screenshots."]
    if notes:
        parts.append(f"User notes: {notes}")
    return _call_vision(
        images, build_draft_system_prompt(), "\n".join(parts), ExtractedDraft
    )


# --- Step 2: fuzzy matching + merge (pure, offline-testable) ---

def _norm(text: str) -> str:
    stripped = "".join(
        ch
        for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    return " ".join(stripped.lower().replace("-", " ").replace("'", "").split())


class _NameMatcher:
    """Fuzzy matcher over a fixed list of canonical names.

    `allow_substring` adds a containment fallback (used for round goals, which
    the screen shows as terse icon tiles like "eggs in [forest]"): a reading
    that is contained in exactly one canonical name — or vice versa — matches.
    """

    def __init__(self, kind: str, names: list[str], allow_substring: bool = False):
        self.kind = kind
        self.canonical = {_norm(n): n for n in names}
        self.keys = list(self.canonical.keys())
        self.allow_substring = allow_substring

    def match(self, raw: str, warnings: list[str]) -> str | None:
        key = _norm(raw)
        if key in self.canonical:
            return self.canonical[key]
        close = difflib.get_close_matches(key, self.keys, n=1, cutoff=0.75)
        if close:
            matched = self.canonical[close[0]]
            if _norm(matched) != key:
                warnings.append(f"Read {self.kind} '{raw}' as '{matched}'.")
            return matched
        if self.allow_substring and len(key) >= 5:
            contained = [k for k in self.keys if key in k or k in key]
            if len(contained) == 1:
                matched = self.canonical[contained[0]]
                warnings.append(f"Read {self.kind} '{raw}' as '{matched}'.")
                return matched
        warnings.append(f"Could not match {self.kind} '{raw}' — add it manually if real.")
        return None


def _match_food(raw: str, warnings: list[str], context: str) -> str | None:
    key = _norm(raw)
    if key in _FOOD_SYNONYMS:
        return _FOOD_SYNONYMS[key]
    close = difflib.get_close_matches(key, list(_FOOD_SYNONYMS.keys()), n=1, cutoff=0.75)
    if close:
        return _FOOD_SYNONYMS[close[0]]
    warnings.append(f"Unrecognized food token '{raw}' in {context} — skipped.")
    return None


def _empty_row(habitat: str) -> HabitatRowSchema:
    return HabitatRowSchema(
        habitat=habitat, slots=[BirdSlotSchema() for _ in range(SLOTS_PER_ROW)]
    )


def _empty_player(name: str) -> PlayerSchema:
    return PlayerSchema(
        name=name,
        board=[_empty_row(h) for h in HABITATS],
        food_supply=FoodSupplySchema(),
        action_cubes_remaining=8,
    )


def _map_players(
    extracted: list[XPlayer],
    base_players: list[PlayerSchema],
    target_player_idx: int | None,
    warnings: list[str],
) -> dict[int, int]:
    """Map extracted-player index → base-player index (by name, then order)."""
    if target_player_idx is not None and len(extracted) == 1:
        if 0 <= target_player_idx < len(base_players):
            return {0: target_player_idx}
        warnings.append(f"target_player_idx {target_player_idx} is out of range — ignored.")

    mapping: dict[int, int] = {}
    used: set[int] = set()
    base_norm = [_norm(p.name) for p in base_players]
    # Pass 1: name matches.
    for i, xp in enumerate(extracted):
        if not xp.name:
            continue
        key = _norm(xp.name)
        close = difflib.get_close_matches(key, base_norm, n=1, cutoff=0.6)
        if close:
            j = base_norm.index(close[0])
            if j not in used:
                mapping[i] = j
                used.add(j)
    # Pass 2: remaining in order.
    for i, xp in enumerate(extracted):
        if i in mapping:
            continue
        free = [j for j in range(len(base_players)) if j not in used]
        if not free:
            warnings.append(
                f"Screenshot shows more players than the tracked game — extra player "
                f"'{xp.name or i + 1}' ignored."
            )
            continue
        mapping[i] = free[0]
        used.add(free[0])
    return mapping


def _apply_board(
    player: PlayerSchema,
    rows: list[XHabitatRow],
    birds: _NameMatcher,
    warnings: list[str],
) -> None:
    reg = get_bird_registry()
    for xrow in rows:
        row = next((r for r in player.board if r.habitat == xrow.habitat), None)
        if row is None:
            continue
        row.slots = [BirdSlotSchema() for _ in range(SLOTS_PER_ROW)]
        if xrow.nectar_spent is not None:
            row.nectar_spent = max(0, xrow.nectar_spent)
        slot_idx = 0
        for xbird in xrow.birds:
            if slot_idx >= SLOTS_PER_ROW:
                warnings.append(
                    f"More than {SLOTS_PER_ROW} birds read in {player.name}'s "
                    f"{xrow.habitat} row — extras dropped."
                )
                break
            name = birds.match(xbird.bird_name, warnings)
            if name is None:
                continue
            bird = reg.get(name)
            slot = row.slots[slot_idx]
            slot.bird_name = bird.name
            slot.egg_limit = bird.egg_limit
            slot.victory_points = bird.victory_points
            slot.nest_type = bird.nest_type.value
            slot.eggs = max(0, min(xbird.eggs, bird.egg_limit))
            if xbird.eggs > bird.egg_limit:
                warnings.append(
                    f"{bird.name}: read {xbird.eggs} eggs but its limit is "
                    f"{bird.egg_limit} — clamped."
                )
            slot.tucked_cards = max(0, xbird.tucked_cards)
            cached: dict[str, int] = {}
            for token in xbird.cached_food:
                food = _match_food(token, warnings, f"{bird.name}'s cache")
                if food:
                    cached[food] = cached.get(food, 0) + 1
            slot.cached_food = cached
            slot_idx += 1


def _parse_die(raw: str, warnings: list[str]) -> str | list[str] | None:
    if "/" in raw:
        halves = [h.strip() for h in raw.split("/") if h.strip()]
        matched = [_match_food(h, warnings, "the birdfeeder") for h in halves]
        matched = [m for m in matched if m]
        if len(matched) >= 2:
            return matched[:2]
        return matched[0] if matched else None
    return _match_food(raw, warnings, "the birdfeeder")


def build_proposed_state(
    extracted: ExtractedGameState,
    current: GameStateSchema | None = None,
    target_player_idx: int | None = None,
) -> tuple[GameStateSchema, list[str]]:
    """Merge an extraction onto the current state (or a fresh default).

    Null/omitted extracted fields keep the current value; present fields
    replace it. Returns the complete proposed state plus warnings.
    """
    warnings: list[str] = []
    birds = _NameMatcher("bird", [b.name for b in get_bird_registry().all_birds])
    bonuses = _NameMatcher("bonus card", [c.name for c in get_bonus_registry().all_cards])
    goals = _NameMatcher(
        "round goal",
        [g.description for g in get_goal_registry().all_goals],
        allow_substring=True,
    )

    if current is not None:
        proposed = current.model_copy(deep=True)
    else:
        n = max(1, len(extracted.players))
        proposed = GameStateSchema(
            players=[
                _empty_player(extracted.players[i].name if i < len(extracted.players) and extracted.players[i].name else f"Player {i + 1}")
                for i in range(n)
            ],
            board_type="oceania",
            round_goals=["No goal"] * 4,
        )

    mapping = _map_players(extracted.players, proposed.players, target_player_idx, warnings)
    for i, xp in enumerate(extracted.players):
        if i not in mapping:
            continue
        player = proposed.players[mapping[i]]
        if xp.board is not None:
            _apply_board(player, xp.board, birds, warnings)
        if xp.food_supply is not None:
            for food in FOOD_TYPES:
                value = getattr(xp.food_supply, food)
                if value is not None:
                    setattr(player.food_supply, food, max(0, value))
        if xp.hand is not None:
            player.hand = [m for raw in xp.hand if (m := birds.match(raw, warnings))]
        if xp.bonus_cards is not None:
            player.bonus_cards = [
                m for raw in xp.bonus_cards if (m := bonuses.match(raw, warnings))
            ]
        if xp.action_cubes_remaining is not None:
            player.action_cubes_remaining = max(0, min(8, xp.action_cubes_remaining))

    if extracted.birdfeeder_dice is not None:
        dice = [d for raw in extracted.birdfeeder_dice if (d := _parse_die(raw, warnings))]
        proposed.birdfeeder.dice = dice[:5]
    if extracted.card_tray is not None:
        proposed.card_tray.face_up = [
            m for raw in extracted.card_tray if (m := birds.match(raw, warnings))
        ][:3]
    if extracted.current_round is not None:
        proposed.current_round = max(1, min(4, extracted.current_round))
    if extracted.turn_in_round is not None:
        proposed.turn_in_round = max(1, extracted.turn_in_round)
    if extracted.deck_remaining is not None:
        proposed.deck_remaining = max(0, extracted.deck_remaining)
    if extracted.round_goals is not None:
        matched_goals = []
        for raw in extracted.round_goals[:4]:
            m = goals.match(raw, warnings)
            matched_goals.append(m if m else "No goal")
        while len(matched_goals) < 4:
            matched_goals.append(
                proposed.round_goals[len(matched_goals)]
                if len(matched_goals) < len(proposed.round_goals)
                else "No goal"
            )
        proposed.round_goals = matched_goals

    return proposed, warnings


class DraftProposal(BaseModel):
    """Matched draft-screen reading, ready for the setup advisor."""

    dealt_birds: list[str] = Field(default_factory=list)
    bonus_cards: list[str] = Field(default_factory=list)
    round_goals: list[str] = Field(default_factory=lambda: ["No Goal"] * 4)
    tray_birds: list[str] = Field(default_factory=list)


def build_draft_proposal(extracted: ExtractedDraft) -> tuple[DraftProposal, list[str]]:
    """Fuzzy-match a draft-screen reading against the registries (pure)."""
    warnings: list[str] = []
    birds = _NameMatcher("bird", [b.name for b in get_bird_registry().all_birds])
    bonuses = _NameMatcher("bonus card", [c.name for c in get_bonus_registry().all_cards])
    goals = _NameMatcher(
        "round goal",
        [g.description for g in get_goal_registry().all_goals],
        allow_substring=True,
    )

    def _dedupe(names: list[str]) -> list[str]:
        seen: set[str] = set()
        return [n for n in names if not (n in seen or seen.add(n))]

    dealt = _dedupe(
        [m for raw in extracted.dealt_birds if (m := birds.match(raw, warnings))]
    )
    if len(dealt) > 5:
        warnings.append(f"Read {len(dealt)} dealt birds — keeping the first 5.")
        dealt = dealt[:5]

    bonus = _dedupe(
        [m for raw in extracted.bonus_cards if (m := bonuses.match(raw, warnings))]
    )[:2]

    round_goals = ["No Goal"] * 4
    if extracted.round_goals:
        for i, raw in enumerate(extracted.round_goals[:4]):
            m = goals.match(raw, warnings)
            if m:
                round_goals[i] = m

    tray = _dedupe(
        [m for raw in (extracted.tray_birds or []) if (m := birds.match(raw, warnings))]
    )[:3]

    return (
        DraftProposal(
            dealt_birds=dealt,
            bonus_cards=bonus,
            round_goals=round_goals,
            tray_birds=tray,
        ),
        warnings,
    )
