"""Screenshot import: read a Wingspan game state from uploaded screenshots."""

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.schemas import GameStateSchema
from backend.vision.screenshot_import import (
    DraftProposal,
    ScreenshotImportError,
    build_draft_proposal,
    build_proposed_state,
    extract_draft_from_images,
    extract_from_images,
    MAX_IMAGES,
    VISION_MODEL,
)

router = APIRouter()


class ImagePayloadSchema(BaseModel):
    media_type: str  # image/png, image/jpeg, image/webp, image/gif
    data: str  # base64-encoded image bytes


class ScreenshotImportRequest(BaseModel):
    images: list[ImagePayloadSchema] = Field(..., min_length=1, max_length=MAX_IMAGES)
    mode: Literal["game", "draft"] = "game"  # draft = the "choose 5 to keep" screen
    notes: str | None = None  # user hints, e.g. "I'm the bottom player"
    current_state: GameStateSchema | None = None  # merge target (else fresh state)
    target_player_idx: int | None = None  # single-player screenshots go to this seat


class ScreenshotImportResponse(BaseModel):
    proposed: GameStateSchema | None = None  # mode == "game"
    draft: DraftProposal | None = None  # mode == "draft"
    warnings: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    model_used: str = VISION_MODEL


# Plain `def`: the vision call blocks for tens of seconds and must run in the
# threadpool, not on the event loop.
@router.post("/screenshot", response_model=ScreenshotImportResponse)
def import_screenshot(req: ScreenshotImportRequest) -> ScreenshotImportResponse:
    images = [(img.media_type, img.data) for img in req.images]
    try:
        if req.mode == "draft":
            extracted_draft = extract_draft_from_images(images, notes=req.notes)
            draft, warnings = build_draft_proposal(extracted_draft)
            return ScreenshotImportResponse(
                draft=draft,
                warnings=warnings,
                uncertainties=extracted_draft.uncertainties,
            )

        extracted = extract_from_images(
            images, notes=req.notes, current=req.current_state
        )
        proposed, warnings = build_proposed_state(
            extracted,
            current=req.current_state,
            target_player_idx=req.target_player_idx,
        )
    except ScreenshotImportError as e:
        raise HTTPException(400, str(e))

    return ScreenshotImportResponse(
        proposed=proposed,
        warnings=warnings,
        uncertainties=extracted.uncertainties,
    )
