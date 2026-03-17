"""
Claude Code LLM Wrapper
Replaces local Ollama models with Claude Code CLI for faster, higher-quality generation.
"""
import subprocess
import logging
import json

logger = logging.getLogger(__name__)


class ClaudeCodeLLM:
    """LLM wrapper that calls the Claude Code CLI (`claude`) as its backend."""

    def __init__(self, model: str = "sonnet", max_tokens: int = 16000):
        self.model = model
        self.max_tokens = max_tokens
        self._verify_cli()

    def _verify_cli(self):
        """Check that the claude CLI is available."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                logger.info(f"Claude Code CLI found: {result.stdout.strip()}")
            else:
                logger.warning("Claude Code CLI returned non-zero, but may still work")
        except FileNotFoundError:
            raise RuntimeError(
                "Claude Code CLI ('claude') not found on PATH. "
                "Install it from https://claude.ai/claude-code"
            )
        except Exception as e:
            logger.warning(f"Could not verify Claude Code CLI: {e}")

    def invoke(self, prompt: str) -> str:
        """
        Send a prompt to Claude Code CLI and return the response text.
        Compatible with the LangChain LLM .invoke() interface used throughout CyberBron.
        """
        try:
            result = subprocess.run(
                [
                    "claude",
                    "-p", prompt,
                    "--model", self.model,
                    "--max-turns", "1",
                    "--output-format", "text",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.error(f"Claude Code CLI error: {error_msg}")
                raise RuntimeError(f"Claude Code CLI failed: {error_msg}")

            response = result.stdout.strip()
            logger.info(f"Claude Code response received ({len(response)} chars)")
            return response

        except subprocess.TimeoutExpired:
            logger.error("Claude Code CLI timed out after 120s")
            raise RuntimeError("Claude Code CLI timed out")
        except FileNotFoundError:
            raise RuntimeError("Claude Code CLI ('claude') not found on PATH")

    def __call__(self, prompt: str) -> str:
        """Allow calling the instance directly."""
        return self.invoke(prompt)
