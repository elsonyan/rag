from fastapi import HTTPException

from src.api.schemas import ChatRequest
from src.clients.llm import get_llm, sanitize_llm_output


def handle_chat(request: ChatRequest) -> dict[str, str]:
    try:
        response = get_llm().invoke(request.messages[-1])
        return {"response": sanitize_llm_output(str(response.content))}
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
