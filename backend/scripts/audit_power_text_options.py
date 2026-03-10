"""Audit option/range semantics against authoritative Excel power text.

Focuses on power-text patterns that imply explicit player choice or numeric ranges
and verifies mapped implementation kwargs preserve those options.

Outputs:
- reports/power_text_conformance_flags.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from backend.config import EXCEL_FILE
from backend.data.registries import load_all
from backend.models.enums import FoodType, NestType
from backend.powers.registry import clear_cache, get_power


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "power_text_conformance_flags.json"


RE_CHOOSE_TUCK = re.compile(
    r"^Choose 1-(?P<max>\d+) birds in (?P<where>this habitat|your \[[^\]]+\])\. "
    r"Tuck 1 \[card\] from your hand behind each\."
    r"(?: If you tuck at least 1 card, draw (?P<draw>\d+) \[card\]\.)?$"
)

RE_CHOOSE_CACHE_SUPPLY = re.compile(
    r"^Choose 1-(?P<max>\d+) birds in your \[[^\]]+\]\. "
    r"Cache 1 \[(?P<food>[a-z]+)\] from your supply on each\.$"
)

RE_TUCK_UP_TO = re.compile(
    r"Tuck up to (?P<max>\d+) \[card\] from your hand behind this bird\."
)

RE_BONUS_DRAW_KEEP = re.compile(
    r"^Draw (?P<draw>\d+) new bonus cards and keep (?P<keep>\d+)\.$"
)
RE_BONUS_DRAW_DISCARD_FROM_ALL = re.compile(
    r"^Draw (?P<draw>\d+) bonus cards, then discard (?P<discard>\d+)\. "
    r"You may discard bonus cards you did not draw this turn\.$"
)
RE_KEA = re.compile(
    r"^Draw 1 bonus card\. You may discard any number of \[wild\] to draw that many additional bonus cards\. "
    r"Keep 1 of the cards you drew and discard the rest\.$"
)
RE_SPOON_BILLED = re.compile(
    r"^Draw (?P<draw>\d+) new bonus cards and keep (?P<keep>\d+)\. "
    r"Other players may discard any 2 resources \(\[wild\], \[egg\], or \[card\]\) to do the same\.$"
)
RE_KAKAPO = re.compile(
    r"^Draw (?P<draw>\d+) bonus cards, keep (?P<keep>\d+), and discard the other (?P<discard>\d+)\.$"
)

RE_TRAY_RESET_OR_REFILL = re.compile(
    r"^Draw 1 face-up \[card\] from the tray with a \[(?P<nest>bowl|cavity|ground|platform)\] "
    r"or \[star\] nest\. You may reset or refill the tray before doing so\.$"
)
RE_TRAY_RESET_REFILL_THEN_DRAW = re.compile(
    r"^Discard all remaining face-up \[card\] and refill the tray\. "
    r"If you do, draw 1 of the new face-up \[card\]\.$"
)

RE_DRAW_THEN_END_TURN_DISCARD = re.compile(
    r"^Draw (?P<draw>\d+) \[card\]\. If you do, discard (?P<discard>\d+) \[card\] "
    r"from your hand at the end of your turn\.$"
)
RE_DISCARD_FOOD_TO_TUCK = re.compile(
    r"^Discard (?P<count>\d+) \[(?P<food>[a-z]+)\] to tuck (?P<tuck>\d+) \[card\] from the deck behind this bird\.$"
)
RE_DISCARD_EGG_OTHER_FOR_ONE_WILD = re.compile(
    r"^Discard 1 \[egg\] from any of your other birds to gain 1 \[wild\] from the supply\.$"
)
RE_DISCARD_EGG_OTHER_FOR_TWO_WILD = re.compile(
    r"^Discard 1 \[egg\] from any of your other birds to gain 2 \[wild\] from the supply\.$"
)
RE_DISCARD_EGG_DRAW_TWO = re.compile(
    r"^Discard 1 \[egg\] to draw 2 \[card\]\.$"
)
RE_PUSH_YOUR_LUCK_PREDATOR = re.compile(
    r"^Choose any (?P<dice>\d+) \[die\]\. Roll (?:it|them) up to (?P<rolls>\d+) times\..*"
    r"If not, stop and return all food cached here this turn\.$"
)
RE_PUSH_YOUR_LUCK_WINGSPAN_DRAW = re.compile(
    r"^Up to (?P<draws>\d+) times, draw 1 \[card\] from the deck\. "
    r"When you stop, if the birds' total wingspan is less than (?P<threshold>\d+) cm, "
    r"tuck them behind this bird\. If not, discard them\.$"
)


def _food_from_token(token: str) -> FoodType | None:
    token = token.strip().lower()
    for ft in (
        FoodType.WILD,
        FoodType.INVERTEBRATE,
        FoodType.SEED,
        FoodType.FISH,
        FoodType.FRUIT,
        FoodType.RODENT,
        FoodType.NECTAR,
    ):
        if token == ft.value:
            return ft
    return None


def _nest_from_token(token: str) -> NestType | None:
    token = token.strip().lower()
    mapping = {
        "bowl": NestType.BOWL,
        "cavity": NestType.CAVITY,
        "ground": NestType.GROUND,
        "platform": NestType.PLATFORM,
        "star": NestType.WILD,
        "wild": NestType.WILD,
    }
    return mapping.get(token)


def _issue(bird_name: str, power_text: str, power, message: str) -> dict:
    return {
        "bird": bird_name,
        "power_text": power_text,
        "mapped_class": f"{type(power).__module__}.{type(power).__name__}",
        "mapped_kwargs": dict(getattr(power, "__dict__", {})),
        "issue": message,
    }


def build_flags() -> dict:
    birds, _, _ = load_all(EXCEL_FILE)
    clear_cache()
    flagged: list[dict] = []
    checked = 0

    for bird in sorted(birds.all_birds, key=lambda b: b.name):
        text = (bird.power_text or "").strip()
        if not text:
            continue
        power = get_power(bird)
        cls = type(power).__name__
        kwargs = dict(getattr(power, "__dict__", {}))

        m_choose_tuck = RE_CHOOSE_TUCK.match(text)
        if m_choose_tuck:
            checked += 1
            expected_max = int(m_choose_tuck.group("max"))
            expected_draw = int(m_choose_tuck.group("draw") or 0)
            got_max = int(kwargs.get("max_targets", -1))
            got_draw = int(kwargs.get("draw_if_tucked", 0))
            if cls != "ChooseBirdsInHabitatTuckFromHandEach":
                flagged.append(_issue(bird.name, text, power, "Expected ChooseBirdsInHabitatTuckFromHandEach"))
            elif got_max != expected_max:
                flagged.append(_issue(bird.name, text, power, f"max_targets mismatch: expected {expected_max}, got {got_max}"))
            elif got_draw != expected_draw:
                flagged.append(_issue(bird.name, text, power, f"draw_if_tucked mismatch: expected {expected_draw}, got {got_draw}"))

        m_choose_cache = RE_CHOOSE_CACHE_SUPPLY.match(text)
        if m_choose_cache:
            checked += 1
            expected_max = int(m_choose_cache.group("max"))
            expected_food = _food_from_token(m_choose_cache.group("food"))
            got_max = int(kwargs.get("max_targets", -1))
            got_food = kwargs.get("food_type")
            if cls != "ChooseBirdsInHabitatCacheFoodEach":
                flagged.append(_issue(bird.name, text, power, "Expected ChooseBirdsInHabitatCacheFoodEach"))
            elif got_max != expected_max:
                flagged.append(_issue(bird.name, text, power, f"max_targets mismatch: expected {expected_max}, got {got_max}"))
            elif got_food != expected_food:
                flagged.append(_issue(bird.name, text, power, f"food_type mismatch: expected {expected_food}, got {got_food}"))

        m_tuck_up_to = RE_TUCK_UP_TO.search(text)
        if m_tuck_up_to:
            checked += 1
            expected_max = int(m_tuck_up_to.group("max"))
            got = None
            if cls == "TuckFromHand":
                got = int(kwargs.get("tuck_count", -1))
            elif cls in {"DrawThenTuckFromHand", "PerThreeEggsInHabitatDrawThenTuck", "TuckFromHandThenDrawEqual"}:
                got = int(kwargs.get("max_tuck", -1))
            else:
                flagged.append(_issue(bird.name, text, power, "Unrecognized class for 'Tuck up to N' semantics"))
                continue
            if got != expected_max:
                flagged.append(_issue(bird.name, text, power, f"tuck max mismatch: expected {expected_max}, got {got}"))

        if text.startswith("Draw [card] equal to the number of players +1. Starting with you and proceeding clockwise, each player selects 1 of those cards and places it in their hand. You keep the extra card."):
            checked += 1
            if cls != "AmericanOystercatcherDraftPlayersPlusOneClockwise":
                flagged.append(_issue(bird.name, text, power, "Expected AmericanOystercatcherDraftPlayersPlusOneClockwise"))

        m_bonus_draw_keep = RE_BONUS_DRAW_KEEP.match(text)
        if m_bonus_draw_keep:
            checked += 1
            expected_draw = int(m_bonus_draw_keep.group("draw"))
            expected_keep = int(m_bonus_draw_keep.group("keep"))
            if cls != "DrawBonusCards":
                flagged.append(_issue(bird.name, text, power, "Expected DrawBonusCards for simple bonus draw/keep"))
            elif int(kwargs.get("draw", -1)) != expected_draw or int(kwargs.get("keep", -1)) != expected_keep:
                flagged.append(
                    _issue(
                        bird.name,
                        text,
                        power,
                        f"DrawBonusCards kwargs mismatch: expected draw={expected_draw}, keep={expected_keep}",
                    )
                )

        m_bonus_discard_from_all = RE_BONUS_DRAW_DISCARD_FROM_ALL.match(text)
        if m_bonus_discard_from_all:
            checked += 1
            expected_draw = int(m_bonus_discard_from_all.group("draw"))
            expected_discard = int(m_bonus_discard_from_all.group("discard"))
            expected_keep = max(0, expected_draw - expected_discard)
            if cls != "DrawBonusCards":
                flagged.append(_issue(bird.name, text, power, "Expected DrawBonusCards for draw-then-discard bonus power"))
            else:
                got_draw = int(kwargs.get("draw", -1))
                got_keep = int(kwargs.get("keep", -1))
                got_discard_all = bool(kwargs.get("discard_from_all_bonus", False))
                if (got_draw, got_keep, got_discard_all) != (expected_draw, expected_keep, True):
                    flagged.append(
                        _issue(
                            bird.name,
                            text,
                            power,
                            "Expected DrawBonusCards(draw, keep=draw-discard, discard_from_all_bonus=True)",
                        )
                    )

        if RE_KEA.match(text):
            checked += 1
            if cls != "KeaDrawBonusDiscardFoodForMoreKeepOne":
                flagged.append(_issue(bird.name, text, power, "Expected KeaDrawBonusDiscardFoodForMoreKeepOne"))

        m_spoon = RE_SPOON_BILLED.match(text)
        if m_spoon:
            checked += 1
            expected_draw = int(m_spoon.group("draw"))
            expected_keep = int(m_spoon.group("keep"))
            if cls != "SpoonBilledSandpiperDrawBonusOthersMayDiscardTwo":
                flagged.append(_issue(bird.name, text, power, "Expected SpoonBilledSandpiperDrawBonusOthersMayDiscardTwo"))
            elif int(kwargs.get("draw", -1)) != expected_draw or int(kwargs.get("keep", -1)) != expected_keep:
                flagged.append(
                    _issue(
                        bird.name,
                        text,
                        power,
                        f"SpoonBilled kwargs mismatch: expected draw={expected_draw}, keep={expected_keep}",
                    )
                )

        m_kakapo = RE_KAKAPO.match(text)
        if m_kakapo:
            checked += 1
            expected_draw = int(m_kakapo.group("draw"))
            expected_keep = int(m_kakapo.group("keep"))
            if cls != "EndGameDrawBonusKeep":
                flagged.append(_issue(bird.name, text, power, "Expected EndGameDrawBonusKeep"))
            elif int(kwargs.get("draw", -1)) != expected_draw or int(kwargs.get("keep", -1)) != expected_keep:
                flagged.append(
                    _issue(
                        bird.name,
                        text,
                        power,
                        f"EndGameDrawBonusKeep kwargs mismatch: expected draw={expected_draw}, keep={expected_keep}",
                    )
                )

        m_reset_or_refill = RE_TRAY_RESET_OR_REFILL.match(text)
        if m_reset_or_refill:
            checked += 1
            expected_nest = _nest_from_token(m_reset_or_refill.group("nest"))
            expected_nests = {expected_nest, NestType.WILD}
            if cls != "WillieWagtailDrawTrayNest":
                flagged.append(_issue(bird.name, text, power, "Expected WillieWagtailDrawTrayNest"))
            else:
                got_allow_reset = bool(kwargs.get("allow_reset", False))
                got_allow_refill = bool(kwargs.get("allow_refill", False))
                got_nests = kwargs.get("allowed_nests")
                if got_allow_reset is not True or got_allow_refill is not True:
                    flagged.append(_issue(bird.name, text, power, "Expected allow_reset=True and allow_refill=True"))
                elif not isinstance(got_nests, set) or got_nests != expected_nests:
                    flagged.append(
                        _issue(
                            bird.name,
                            text,
                            power,
                            f"allowed_nests mismatch: expected {expected_nests}, got {got_nests}",
                        )
                    )

        if RE_TRAY_RESET_REFILL_THEN_DRAW.match(text):
            checked += 1
            if cls != "WhiteStorkResetTrayThenDraw":
                flagged.append(_issue(bird.name, text, power, "Expected WhiteStorkResetTrayThenDraw"))

        m_draw_end = RE_DRAW_THEN_END_TURN_DISCARD.match(text)
        if m_draw_end:
            checked += 1
            expected_draw = int(m_draw_end.group("draw"))
            expected_discard = int(m_draw_end.group("discard"))
            if cls != "DrawThenEndTurnDiscardFromHand":
                flagged.append(_issue(bird.name, text, power, "Expected DrawThenEndTurnDiscardFromHand"))
            elif int(kwargs.get("draw", -1)) != expected_draw or int(kwargs.get("discard", -1)) != expected_discard:
                flagged.append(
                    _issue(
                        bird.name,
                        text,
                        power,
                        f"DrawThenEndTurnDiscard kwargs mismatch: expected draw={expected_draw}, discard={expected_discard}",
                    )
                )

        m_discard_tuck = RE_DISCARD_FOOD_TO_TUCK.match(text)
        if m_discard_tuck:
            checked += 1
            expected_discard = int(m_discard_tuck.group("count"))
            expected_food = _food_from_token(m_discard_tuck.group("food"))
            expected_tuck_total = int(m_discard_tuck.group("tuck"))
            expected_tuck_per = expected_tuck_total // expected_discard if expected_discard > 0 else expected_tuck_total
            if cls == "FlockingPower":
                got_count = int(kwargs.get("count", -1))
                if not (
                    expected_food == FoodType.WILD
                    and expected_discard == 1
                    and expected_tuck_total == 1
                    and got_count == 1
                ):
                    flagged.append(
                        _issue(
                            bird.name,
                            text,
                            power,
                            "Expected FlockingPower(count=1) only for 'Discard 1 [wild] to tuck 1'",
                        )
                    )
            elif cls != "DiscardFoodToTuckFromDeck":
                flagged.append(_issue(bird.name, text, power, "Expected DiscardFoodToTuckFromDeck"))
            else:
                got_discard = int(kwargs.get("max_discard", -1))
                got_food = kwargs.get("food_type")
                got_tuck_per = int(kwargs.get("tuck_per_discard", -1))
                if got_discard != expected_discard or got_food != expected_food or got_tuck_per != expected_tuck_per:
                    flagged.append(
                        _issue(
                            bird.name,
                            text,
                            power,
                            (
                                "DiscardFoodToTuckFromDeck kwargs mismatch: "
                                f"expected max_discard={expected_discard}, food_type={expected_food}, tuck_per_discard={expected_tuck_per}"
                            ),
                        )
                    )

        if RE_DISCARD_EGG_OTHER_FOR_ONE_WILD.match(text):
            checked += 1
            if cls != "DiscardEggForBenefit":
                flagged.append(_issue(bird.name, text, power, "Expected DiscardEggForBenefit"))
            else:
                if int(kwargs.get("egg_cost", -1)) != 1 or int(kwargs.get("food_gain", -1)) != 1 or kwargs.get("food_type") != FoodType.WILD:
                    flagged.append(_issue(bird.name, text, power, "Expected egg_cost=1, food_gain=1, food_type=WILD"))
                elif int(kwargs.get("card_gain", -1)) != 0 or bool(kwargs.get("require_other_bird", False)) is not True:
                    flagged.append(_issue(bird.name, text, power, "Expected card_gain=0 and require_other_bird=True"))

        if RE_DISCARD_EGG_OTHER_FOR_TWO_WILD.match(text):
            checked += 1
            if cls != "DiscardEggForBenefit":
                flagged.append(_issue(bird.name, text, power, "Expected DiscardEggForBenefit"))
            else:
                if int(kwargs.get("egg_cost", -1)) != 1 or int(kwargs.get("food_gain", -1)) != 2 or kwargs.get("food_type") != FoodType.WILD:
                    flagged.append(_issue(bird.name, text, power, "Expected egg_cost=1, food_gain=2, food_type=WILD"))
                elif int(kwargs.get("card_gain", -1)) != 0 or bool(kwargs.get("require_other_bird", False)) is not True:
                    flagged.append(_issue(bird.name, text, power, "Expected card_gain=0 and require_other_bird=True"))

        if RE_DISCARD_EGG_DRAW_TWO.match(text):
            checked += 1
            if cls != "DiscardEggForBenefit":
                flagged.append(_issue(bird.name, text, power, "Expected DiscardEggForBenefit"))
            else:
                if int(kwargs.get("egg_cost", -1)) != 1 or int(kwargs.get("card_gain", -1)) != 2:
                    flagged.append(_issue(bird.name, text, power, "Expected egg_cost=1 and card_gain=2"))

        m_push = RE_PUSH_YOUR_LUCK_PREDATOR.match(text)
        if m_push:
            checked += 1
            expected_dice = int(m_push.group("dice"))
            expected_rolls = int(m_push.group("rolls"))
            foods = {_food_from_token(token) for token in re.findall(r"\[([a-z]+)\]", text.lower())}
            expected_foods = {ft for ft in foods if ft in {FoodType.FISH, FoodType.INVERTEBRATE, FoodType.RODENT}}
            if cls != "PushYourLuckPredatorDiceCache":
                flagged.append(_issue(bird.name, text, power, "Expected PushYourLuckPredatorDiceCache"))
            else:
                got_dice = int(kwargs.get("dice_count", -1))
                got_rolls = int(kwargs.get("max_rolls", -1))
                got_foods = kwargs.get("target_foods")
                got_cache_food = kwargs.get("cache_food")
                got_cache_per = int(kwargs.get("cache_per_hit", -1))
                if got_dice != expected_dice or got_rolls != expected_rolls:
                    flagged.append(
                        _issue(
                            bird.name,
                            text,
                            power,
                            f"push-your-luck dice/roll mismatch: expected dice_count={expected_dice}, max_rolls={expected_rolls}",
                        )
                    )
                elif not isinstance(got_foods, set) or got_foods != expected_foods:
                    flagged.append(
                        _issue(
                            bird.name,
                            text,
                            power,
                            f"target_foods mismatch: expected {expected_foods}, got {got_foods}",
                        )
                    )
                elif got_cache_food not in expected_foods or got_cache_per != 1:
                    flagged.append(
                        _issue(
                            bird.name,
                            text,
                            power,
                            "Expected cache_food to be one of target_foods and cache_per_hit=1",
                        )
                    )

        m_wingspan_push = RE_PUSH_YOUR_LUCK_WINGSPAN_DRAW.match(text)
        if m_wingspan_push:
            checked += 1
            expected_draws = int(m_wingspan_push.group("draws"))
            expected_threshold = int(m_wingspan_push.group("threshold"))
            if cls != "PushYourLuckDrawByWingspanTotalThenTuck":
                flagged.append(_issue(bird.name, text, power, "Expected PushYourLuckDrawByWingspanTotalThenTuck"))
            else:
                got_draws = int(kwargs.get("max_draws", -1))
                got_threshold = int(kwargs.get("wingspan_threshold", -1))
                if got_draws != expected_draws or got_threshold != expected_threshold:
                    flagged.append(
                        _issue(
                            bird.name,
                            text,
                            power,
                            (
                                "push-your-luck wingspan mismatch: "
                                f"expected max_draws={expected_draws}, wingspan_threshold={expected_threshold}"
                            ),
                        )
                    )

    return {
        "checked_patterns": checked,
        "flagged": len(flagged),
        "rows": flagged,
    }


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_flags()
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"checked_patterns": report["checked_patterns"], "flagged": report["flagged"]}, indent=2))


if __name__ == "__main__":
    main()
