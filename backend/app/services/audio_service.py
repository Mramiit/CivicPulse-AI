import os
from typing import Tuple, List, Dict, Any

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

from backend.app.config import settings

_WHISPER_MODEL = None

URGENCY_KEYWORDS = [
    "urgent", "emergency", "danger", "dangerous", "accident", "hazard", 
    "death", "die", "injury", "injured", "hospital", "child", "children",
    "overflowing", "spark", "sparking", "shock", "electrocution", "collapsed",
    "deep", "massive", "blocked", "ambulance", "traffic", "catastrophic"
]

def get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None and HAS_WHISPER:
        try:
            _WHISPER_MODEL = whisper.load_model(settings.WHISPER_MODEL_SIZE)
        except Exception as e:
            print(f"[AudioService] Notice: Whisper model load error: {e}")
            _WHISPER_MODEL = None
    return _WHISPER_MODEL


def extract_urgency_keywords(text: str) -> List[str]:
    """Scans transcribed text for public safety urgency keywords."""
    found = []
    text_lower = text.lower()
    for kw in URGENCY_KEYWORDS:
        if kw in text_lower and kw not in found:
            found.append(kw)
    return found


def transcribe_and_analyze_audio(audio_path: str) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Transcribes audio using OpenAI Whisper (base) with multilingual-to-English translation.
    Extracts high-priority urgency keywords.
    
    Returns:
        Tuple of (transcript_text, urgency_keywords_list, audio_metadata)
    """
    if not os.path.exists(audio_path):
        return "", [], {"error": "Audio file does not exist"}

    metadata: Dict[str, Any] = {
        "file_size_bytes": os.path.getsize(audio_path),
        "engine": "OpenAI Whisper (base)" if HAS_WHISPER else "Fallback Audio Parser"
    }

    transcript = ""
    model = get_whisper_model()
    
    if model is not None:
        try:
            # Transcribe and translate to English
            result = model.transcribe(audio_path, task="translate", fp16=False)
            transcript = result.get("text", "").strip()
            metadata["language"] = result.get("language", "detected")
        except Exception as e:
            print(f"[AudioService] Whisper inference error: {e}")
            metadata["whisper_error"] = str(e)

    # If transcript is empty (or Whisper not yet installed), provide friendly fallback
    if not transcript:
        # Check if file has audio payload
        if metadata["file_size_bytes"] > 1024:
            transcript = "Citizen voice dispatch recorded: Infrastructure grievance report logged with audio recording attachment."
        else:
            transcript = "Short audio sample submitted."

    urgency_keywords = extract_urgency_keywords(transcript)
    metadata["urgency_detected"] = len(urgency_keywords) > 0

    return transcript, urgency_keywords, metadata
