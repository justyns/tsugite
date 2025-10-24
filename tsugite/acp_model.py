"""ACP (Agent Client Protocol) model adapter for tsugite.

This module provides integration with ACP-compatible services like Claude Code.
ACP is a protocol for communicating with AI agents over HTTP.

References:
- https://agentclientprotocol.com/
- https://github.com/agentclientprotocol/agent-client-protocol
- https://github.com/zed-industries/claude-code-acp
"""

import json
from typing import Any, Dict, List, Optional

import httpx
from smolagents.models import Model


class ACPModel(Model):
    """Model adapter for Agent Client Protocol (ACP) services.

    ACP is a protocol that allows communication with AI agents over HTTP.
    This adapter enables tsugite to connect to ACP servers like Claude Code.

    Args:
        server_url: Base URL of the ACP server (e.g., "http://localhost:8080")
        model_id: Optional model identifier to pass to the ACP server
        timeout: Request timeout in seconds
        **kwargs: Additional configuration options

    Examples:
        >>> model = ACPModel(server_url="http://localhost:8080")
        >>> model = ACPModel(server_url="http://localhost:8080", model_id="claude-3-5-sonnet-20241022")
    """

    def __init__(
        self,
        server_url: str,
        model_id: Optional[str] = None,
        timeout: float = 300.0,
        **kwargs,
    ):
        self.server_url = server_url.rstrip("/")
        self.model_id = model_id or "acp"
        self.timeout = timeout
        self.extra_params = kwargs

        # Initialize HTTP client
        self.client = httpx.Client(timeout=timeout)

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Execute ACP request and return response.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            stop_sequences: Optional stop sequences
            grammar: Optional grammar specification
            **kwargs: Additional parameters

        Returns:
            Response text from the ACP server

        Raises:
            RuntimeError: If the ACP request fails
        """
        # Convert messages to ACP format
        # ACP typically expects a prompt or conversation history
        acp_request = self._build_acp_request(messages, stop_sequences, grammar, **kwargs)

        try:
            # Send request to ACP server
            response = self.client.post(
                f"{self.server_url}/acp/v1/messages",
                json=acp_request,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            # Parse response
            result = response.json()
            return self._extract_response_text(result)

        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"ACP request failed with status {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            raise RuntimeError(f"ACP request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"ACP request failed: {e}")

    def _build_acp_request(
        self,
        messages: List[Dict[str, Any]],
        stop_sequences: Optional[List[str]],
        grammar: Optional[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """Build ACP request payload from smolagents format.

        Args:
            messages: Smolagents message list
            stop_sequences: Stop sequences
            grammar: Grammar specification
            **kwargs: Additional parameters

        Returns:
            ACP-formatted request dictionary
        """
        # Build base request
        request = {
            "messages": messages,
        }

        # Add model if specified
        if self.model_id and self.model_id != "acp":
            request["model"] = self.model_id

        # Add optional parameters
        if stop_sequences:
            request["stop_sequences"] = stop_sequences

        if grammar:
            request["grammar"] = grammar

        # Add max_tokens if specified
        max_tokens = kwargs.get("max_tokens") or self.extra_params.get("max_tokens")
        if max_tokens:
            request["max_tokens"] = max_tokens

        # Add temperature if specified
        temperature = kwargs.get("temperature") or self.extra_params.get("temperature")
        if temperature is not None:
            request["temperature"] = temperature

        # Merge any additional parameters
        request.update({k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature"]})

        return request

    def _extract_response_text(self, response: Dict[str, Any]) -> str:
        """Extract response text from ACP response.

        ACP responses can have various formats depending on the server.
        This method handles common response formats.

        Args:
            response: ACP response dictionary

        Returns:
            Extracted response text

        Raises:
            RuntimeError: If response format is unexpected
        """
        # Try different common response formats
        # Format 1: Anthropic-style response
        if "content" in response:
            content = response["content"]
            if isinstance(content, list):
                # Extract text from content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                return "".join(text_parts)
            elif isinstance(content, str):
                return content

        # Format 2: OpenAI-style response
        if "choices" in response:
            choices = response["choices"]
            if choices and len(choices) > 0:
                choice = choices[0]
                if "message" in choice:
                    return choice["message"].get("content", "")
                elif "text" in choice:
                    return choice["text"]

        # Format 3: Simple text response
        if "text" in response:
            return response["text"]

        # Format 4: Message response
        if "message" in response:
            message = response["message"]
            if isinstance(message, str):
                return message
            elif isinstance(message, dict):
                return message.get("content", "")

        # If we can't find a standard format, return the entire response as JSON
        # This allows debugging and may work for simple cases
        return json.dumps(response)

    def __del__(self):
        """Clean up HTTP client on deletion."""
        if hasattr(self, "client"):
            self.client.close()
