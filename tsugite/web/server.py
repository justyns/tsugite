"""FastAPI web server for tsugite."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from smolagents.monitoring import LogLevel

from tsugite.agent_runner import run_agent
from tsugite.chat import ChatManager
from tsugite.custom_ui import CustomUILogger, UIEvent
from tsugite.web.ui_handler import SSEUIHandler

app = FastAPI(title="Tsugite Web UI")

# Track active executions (single-run agents)
executions: Dict[str, SSEUIHandler] = {}

# Track chat sessions
chat_sessions: Dict[str, ChatManager] = {}


def find_agents(base_path: Path) -> list[dict]:
    """Find all agent markdown files."""
    agents = []

    # Search directories
    search_dirs = [
        base_path / ".tsugite",
        base_path / "agents",
        base_path / "examples" / "agents",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for agent_file in search_dir.glob("**/*.md"):
            # Skip files that look like docs
            if any(part.startswith("README") for part in agent_file.parts):
                continue

            agents.append(
                {
                    "name": agent_file.stem,
                    "path": str(agent_file.relative_to(base_path)),
                    "full_path": str(agent_file),
                }
            )

    return agents


def _get_session_or_404(session_id: str) -> ChatManager:
    """Get chat session or raise 404."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return chat_sessions[session_id]


async def _send_done_event():
    """Generate SSE done event."""
    return "event: done\ndata: {}\n\n"


async def _cleanup_and_done(execution_id: str):
    """Send done event and schedule cleanup."""
    asyncio.create_task(cleanup_execution(execution_id))
    return await _send_done_event()


@app.get("/")
async def index():
    """Serve the main UI."""
    static_dir = Path(__file__).parent / "static"
    return FileResponse(static_dir / "index.html")


@app.get("/chat")
async def chat():
    """Serve the chat UI."""
    static_dir = Path(__file__).parent / "static"
    return FileResponse(static_dir / "chat.html")


@app.get("/api/agents")
async def list_agents():
    """List available agents."""
    base_path = Path.cwd()
    agents = find_agents(base_path)
    return {"agents": agents}


@app.post("/api/run")
async def run_agent_endpoint(
    agent_path: str = Form(...),
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
):
    """Execute an agent and return execution ID for streaming."""
    # Generate unique execution ID
    execution_id = str(uuid.uuid4())

    # Create SSE handler
    sse_handler = SSEUIHandler()
    logger = CustomUILogger(sse_handler, level=LogLevel.INFO)

    # Store execution
    executions[execution_id] = sse_handler

    print(f"[WEB] Starting execution {execution_id} for agent: {agent_path}")

    # Run agent in background
    asyncio.create_task(
        _run_in_executor(
            execution_id,
            lambda: run_agent(
                agent_path=Path(agent_path),
                prompt=prompt,
                model_override=model,
                custom_logger=logger,
            ),
        )
    )

    return {"execution_id": execution_id}


async def _run_in_executor(execution_id: str, func, *args, error_type: str = "Execution Error"):
    """Run function in executor and handle events."""
    try:
        print(f"[WEB] Running agent in executor for {execution_id}")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, func, *args)
        print(f"[WEB] Agent completed for {execution_id}, result: {str(result)[:100]}")

        handler = executions[execution_id]
        handler.handle_event(UIEvent.FINAL_ANSWER, {"answer": result})

    except Exception as e:
        print(f"[WEB] Error in execution {execution_id}: {e}")
        import traceback

        traceback.print_exc()
        handler = executions.get(execution_id)
        if handler:
            handler.handle_event(UIEvent.ERROR, {"error": str(e), "error_type": error_type})


@app.get("/api/stream/{execution_id}")
async def stream_events(execution_id: str):
    """Stream SSE events for an execution."""
    if execution_id not in executions:
        raise HTTPException(status_code=404, detail="Execution not found")

    handler = executions[execution_id]
    print(f"[WEB] SSE connection established for {execution_id}")

    async def event_generator():
        """Generate SSE events."""
        try:
            while True:
                # Get next event with timeout
                try:
                    event_data = await asyncio.wait_for(handler.get_event(), timeout=0.5)
                except asyncio.TimeoutError:
                    if handler.is_done:
                        print(f"[WEB] Execution {execution_id} complete, sending done event")
                        yield await _cleanup_and_done(execution_id)
                        break
                    yield ": keepalive\n\n"
                    continue

                event_name = event_data["event"]
                data_json = json.dumps(event_data["data"])

                print(f"[WEB] Sending event '{event_name}' for {execution_id}")
                yield f"event: {event_name}\ndata: {data_json}\n\n"

                if event_name in ("final_answer", "error"):
                    yield await _cleanup_and_done(execution_id)
                    break

        except asyncio.CancelledError:
            # Client disconnected
            asyncio.create_task(cleanup_execution(execution_id))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def cleanup_execution(execution_id: str, delay: int = 60):
    """Clean up execution after delay."""
    await asyncio.sleep(delay)
    if execution_id in executions:
        del executions[execution_id]


@app.post("/api/chat/sessions")
async def create_chat_session(
    agent_path: str = Form(...),
    model: Optional[str] = Form(None),
    max_history: int = Form(50),
):
    """Create a new chat session."""
    # Validate agent path
    agent_file = Path(agent_path)
    if not agent_file.exists():
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_path}")

    # Create session
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = ChatManager(
        agent_path=agent_file,
        model_override=model,
        max_history=max_history,
    )

    return {"session_id": session_id, "agent": str(agent_file), "model": model or "default"}


@app.post("/api/chat/sessions/{session_id}/messages")
async def send_chat_message(
    session_id: str,
    message: str = Form(...),
):
    """Send a message to a chat session."""
    manager = _get_session_or_404(session_id)

    sse_handler = SSEUIHandler()
    logger = CustomUILogger(sse_handler, level=LogLevel.INFO)

    execution_id = str(uuid.uuid4())
    executions[execution_id] = sse_handler

    manager.custom_logger = logger

    asyncio.create_task(_run_in_executor(execution_id, manager.run_turn, message, error_type="Chat Error"))

    return {"execution_id": execution_id, "session_id": session_id}


@app.get("/api/chat/sessions/{session_id}")
async def get_chat_session(session_id: str):
    """Get chat session info and history."""
    manager = _get_session_or_404(session_id)
    stats = manager.get_stats()

    return {
        "session_id": session_id,
        "history": [
            {
                "timestamp": turn.timestamp.isoformat(),
                "user_message": turn.user_message,
                "agent_response": turn.agent_response,
            }
            for turn in manager.conversation_history
        ],
        "stats": stats,
    }


@app.post("/api/chat/sessions/{session_id}/clear")
async def clear_chat_history(session_id: str):
    """Clear chat session history."""
    manager = _get_session_or_404(session_id)
    manager.clear_history()
    return {"status": "cleared"}


@app.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session."""
    _get_session_or_404(session_id)
    del chat_sessions[session_id]
    return {"status": "deleted"}


# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
