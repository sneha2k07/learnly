from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..models.document import Document, StudyAsset
from ..services import ai

router = APIRouter(prefix="/api/study", tags=["study"])

ASSET_GENERATORS = {
    "notes": ai.generate_notes,
    "flashcards": ai.generate_flashcards,
    "quiz": ai.generate_quiz,
    "podcast_script": ai.generate_podcast_script,
    "mindmap": ai.generate_mindmap,
}


def _get_ready_doc(doc_id: int, user_id: int, db: Session) -> Document:
    doc = db.query(Document).filter(Document.id == doc_id, Document.user_id == user_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != "ready":
        raise HTTPException(status_code=202, detail=f"Document is still {doc.status}. Try again shortly.")
    if not doc.raw_text:
        raise HTTPException(status_code=422, detail="No text could be extracted from this document.")
    return doc


@router.post("/{doc_id}/{asset_type}", status_code=201)
def generate_asset(
    doc_id: int,
    asset_type: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if asset_type not in ASSET_GENERATORS:
        raise HTTPException(status_code=400, detail=f"Unknown asset type. Choose from: {list(ASSET_GENERATORS)}")

    doc = _get_ready_doc(doc_id, current_user.id, db)

    # Return cached asset if it exists
    existing = db.query(StudyAsset).filter(
        StudyAsset.document_id == doc_id,
        StudyAsset.user_id == current_user.id,
        StudyAsset.asset_type == asset_type,
    ).first()
    if existing:
        return {"asset_type": asset_type, "content": existing.content, "cached": True}

    # Generate via AI
    try:
        generator = ASSET_GENERATORS[asset_type]
        content = generator(doc.raw_text, doc.title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")

    asset = StudyAsset(
        document_id=doc_id,
        user_id=current_user.id,
        asset_type=asset_type,
        content=content,
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)

    return {"asset_type": asset_type, "content": content, "cached": False}


@router.get("/{doc_id}", response_model=list)
def list_assets(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = db.query(Document).filter(Document.id == doc_id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    assets = db.query(StudyAsset).filter(
        StudyAsset.document_id == doc_id,
        StudyAsset.user_id == current_user.id,
    ).all()
    return [{"asset_type": a.asset_type, "id": a.id, "created_at": str(a.created_at)} for a in assets]


@router.delete("/{doc_id}/{asset_type}", status_code=204)
def regenerate_asset(
    doc_id: int,
    asset_type: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete cached asset so it will be regenerated fresh on next request."""
    asset = db.query(StudyAsset).filter(
        StudyAsset.document_id == doc_id,
        StudyAsset.user_id == current_user.id,
        StudyAsset.asset_type == asset_type,
    ).first()
    if asset:
        db.delete(asset)
        db.commit()


# ── CHAT ─────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@router.post("/{doc_id}/chat")
def chat(
    doc_id: int,
    body: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = _get_ready_doc(doc_id, current_user.id, db)
    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    try:
        reply = ai.chat_with_document(doc.raw_text, doc.title, messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
    return {"reply": reply}
