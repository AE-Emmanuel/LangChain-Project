# ollama_cli.py
from __future__ import annotations
import shlex
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OllamaCLI:
    """
    Minimal wrapper to call `ollama run <model> --prompt "<prompt>"` and return the text.
    Designed to be passed into QAAgent as llm_instance.
    """

    def __init__(self, model: str = "llama3", timeout: int = 60):
        self.model = model
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        """
        Call the Ollama CLI and return model output as a string.
        This uses subprocess.run and captures stdout.
        """
        # Build base command
        cmd = ["ollama", "run", self.model, prompt]

        logger.info("Running ollama CLI: %s", " ".join(cmd[:5]) + " ...")
        try:
            # Use subprocess to call CLI
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except FileNotFoundError:
            raise RuntimeError("`ollama` CLI not found in PATH. Install Ollama and ensure `ollama` is on PATH.")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ollama call timed out.")

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            raise RuntimeError(f"Ollama CLI failed (rc={proc.returncode}): {stderr}")

        out = proc.stdout.strip()
        # CLI output may include metadata or newlines; heuristics:
        # Some Ollama CLI versions print the generation directly. Return stdout.
        return out
