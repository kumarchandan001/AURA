"""
ollama_client.py — Asynchronous Local LLM Client with Dynamic System Prompting.

Communicates with a locally hosted Ollama server to provide privacy-preserving
AI assistance whose communication style *adapts in real-time* to the user's
physiological stress level.

Core Innovation — Dynamic Prompt Injection:
    The system prompt is regenerated before every LLM call based on the
    current ``PhysiologicalState``.  When the user is **Calm**, the AI
    operates as a precise, technical coding assistant.  When the user is
    **Stressed**, the prompt switches to concise, supportive language with
    breathing reminders.

Thread Safety:
    All inference runs on a dedicated background thread via
    ``concurrent.futures.ThreadPoolExecutor``.  The GUI and camera loops
    are never blocked by LLM generation latency.

Classes:
    OllamaClient — Threaded Ollama chat client with state-aware prompting.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ======================================================================
# Dynamic System Prompts
# ======================================================================

_SYSTEM_PROMPT_CALM = """You are SENTIO AI — a highly precise, technical coding assistant
embedded in a biometric-aware development environment.

Current Physiological State: CALM (parasympathetic dominant).

Communication Protocol:
• Be detailed, thorough, and technically rigorous.
• Use precise terminology, cite best practices, and explore edge cases.
• Provide comprehensive explanations with code examples when relevant.
• Assume the user has high cognitive bandwidth — present the full picture.
• Structure responses with headers, bullet points, and code blocks.
• Challenge assumptions constructively when you spot potential issues.

You are part of Project SENTIO, an rPPG-based physiological monitoring framework.
Respond helpfully to any technical question the user asks."""

_SYSTEM_PROMPT_STRESSED = """You are SENTIO AI — a supportive, calm coding assistant
embedded in a biometric-aware development environment.

Current Physiological State: ELEVATED STRESS (sympathetic activation detected).

Communication Protocol:
• Be concise and direct. Use short sentences.
• Simplify complex concepts — break them into small, actionable steps.
• Use encouraging, warm language. Avoid overwhelming detail.
• Lead with the solution, then offer optional deeper explanation.
• If the user is debugging, gently suggest: "Let's take one thing at a time."
• Occasionally remind: "Take a slow breath — you've got this. 🫁"
• Limit responses to the essential information needed right now.
• Use bullet points and keep lists short (3-5 items max).

You are part of Project SENTIO. Help the user feel supported while staying productive."""

_SYSTEM_PROMPT_UNKNOWN = """You are SENTIO AI — a helpful coding assistant
embedded in a biometric-aware development environment.

Current Physiological State: Calibrating (biometric data is still being collected).

Communication Protocol:
• Be helpful, clear, and moderately detailed.
• Use a balanced tone — professional yet approachable.
• Provide clear answers with relevant context.

You are part of Project SENTIO. Respond helpfully to the user's questions."""


def _get_system_prompt(state: str) -> str:
    """Return the system prompt corresponding to the affective state.

    Parameters
    ----------
    state : str
        One of ``"Calm"``, ``"Stressed"``, ``"Unknown"``.

    Returns
    -------
    str
        The full system prompt text.
    """
    prompts = {
        "Calm": _SYSTEM_PROMPT_CALM,
        "Stressed": _SYSTEM_PROMPT_STRESSED,
        "Unknown": _SYSTEM_PROMPT_UNKNOWN,
    }
    return prompts.get(state, _SYSTEM_PROMPT_UNKNOWN)


# ======================================================================
# Ollama Client
# ======================================================================

class OllamaClient:
    """Thread-safe, asynchronous Ollama chat client with affective prompting.

    Parameters
    ----------
    model : str
        Ollama model name (e.g., ``"qwen2:0.5b"``, ``"phi3"``, ``"llama3"``).
    base_url : str
        Ollama server URL (default: ``"http://localhost:11434"``).
    max_history : int
        Maximum conversation turns to retain for context.
    timeout : float
        Request timeout in seconds.

    Attributes
    ----------
    is_available : bool
        Whether the Ollama server is reachable.
    is_generating : bool
        Whether an inference request is currently in progress.

    Examples
    --------
    >>> client = OllamaClient(model="qwen2:0.5b")
    >>> client.chat_async("Explain rPPG", state="Calm", callback=print)
    """

    def __init__(
        self,
        model: str = "qwen2:0.5b",
        base_url: str = "http://localhost:11434",
        max_history: int = 10,
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._max_history = max_history
        self._timeout = timeout

        self._history: List[Dict[str, str]] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Ollama")

        self.is_available: bool = False
        self.is_generating: bool = False

        # Check server connectivity.
        self._check_server()

        logger.info(
            "OllamaClient initialised — model=%s  server=%s  available=%s",
            model,
            base_url,
            self.is_available,
        )

    def _check_server(self) -> None:
        """Ping the Ollama server to verify connectivity."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self._base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    self.is_available = True
                    logger.info("Ollama server is reachable.")
                else:
                    self.is_available = False
                    logger.warning("Ollama server returned status %d.", resp.status)
        except Exception as e:
            self.is_available = False
            logger.warning("Ollama server not reachable: %s", e)

    def chat_async(
        self,
        user_message: str,
        state: str = "Unknown",
        callback: Optional[Callable[[str], None]] = None,
    ) -> Optional[Future]:
        """Submit a chat request to Ollama on a background thread.

        The system prompt is dynamically injected based on the current
        physiological ``state``.  The result is delivered via ``callback``
        on the worker thread — callers should use thread-safe mechanisms
        (e.g., ``queue.Queue`` or ``tkinter.after``) to update the UI.

        Parameters
        ----------
        user_message : str
            The user's chat message.
        state : str
            Current affective state (``"Calm"``, ``"Stressed"``, ``"Unknown"``).
        callback : callable, optional
            Function called with the AI response string upon completion.
            Also called with an error message string on failure.

        Returns
        -------
        concurrent.futures.Future or None
            The submitted future, or ``None`` if the server is unavailable.
        """
        if not self.is_available:
            # Retry connectivity check.
            self._check_server()
            if not self.is_available:
                error_msg = (
                    "⚠️ SENTIO AI is offline.\n\n"
                    "The local Ollama server is not reachable. Please ensure:\n"
                    "1. Ollama is installed (https://ollama.com)\n"
                    "2. The server is running: `ollama serve`\n"
                    f"3. A model is pulled: `ollama pull {self._model}`"
                )
                if callback:
                    callback(error_msg)
                return None

        if self.is_generating:
            if callback:
                callback("⏳ Please wait — still generating the previous response...")
            return None

        future = self._executor.submit(
            self._do_chat, user_message, state, callback
        )
        return future

    def _do_chat(
        self,
        user_message: str,
        state: str,
        callback: Optional[Callable[[str], None]],
    ) -> str:
        """Execute the chat request synchronously (runs on worker thread).

        Parameters
        ----------
        user_message : str
            The user message.
        state : str
            Affective state for prompt selection.
        callback : callable or None
            Result callback.

        Returns
        -------
        str
            The AI response text.
        """
        self.is_generating = True
        try:
            system_prompt = _get_system_prompt(state)

            # Build messages array.
            messages = [{"role": "system", "content": system_prompt}]

            with self._lock:
                messages.extend(self._history[-self._max_history:])

            messages.append({"role": "user", "content": user_message})

            # Call Ollama API via HTTP.
            response_text = self._call_ollama_api(messages)

            # Update history.
            with self._lock:
                self._history.append({"role": "user", "content": user_message})
                self._history.append({"role": "assistant", "content": response_text})

                # Trim history.
                if len(self._history) > self._max_history * 2:
                    self._history = self._history[-self._max_history * 2:]

            if callback:
                callback(response_text)

            return response_text

        except Exception as e:
            error_msg = f"⚠️ LLM Error: {e}"
            logger.error("Ollama chat failed: %s", e)
            if callback:
                callback(error_msg)
            return error_msg
        finally:
            self.is_generating = False

    def _call_ollama_api(self, messages: List[Dict[str, str]]) -> str:
        """Make a synchronous HTTP POST to the Ollama chat endpoint.

        Parameters
        ----------
        messages : list[dict]
            The conversation messages in OpenAI-compatible format.

        Returns
        -------
        str
            The assistant's response text.

        Raises
        ------
        RuntimeError
            If Ollama returns an error (e.g., insufficient memory).
        """
        import json
        import urllib.error
        import urllib.request

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 512,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as http_err:
            # Read the error body from Ollama for a descriptive message.
            error_detail = "Unknown server error"
            try:
                err_body = json.loads(http_err.read().decode("utf-8"))
                error_detail = err_body.get("error", str(http_err))
            except Exception:
                error_detail = str(http_err)
            logger.error(
                "Ollama HTTP %d: %s", http_err.code, error_detail
            )
            raise RuntimeError(error_detail) from http_err

        return body.get("message", {}).get("content", "(empty response)")

    def clear_history(self) -> None:
        """Clear the conversation history."""
        with self._lock:
            self._history.clear()
        logger.info("Conversation history cleared.")

    def shutdown(self) -> None:
        """Shut down the thread pool executor."""
        self._executor.shutdown(wait=False)
        logger.info("OllamaClient executor shut down.")
