"""Generic HTTP proxy for external backend services.

This module provides a simple HTTP client for forwarding requests
to configured backend services. No service-specific logic here.
"""

import os
from typing import Dict, Any
import requests
import logging

logger = logging.getLogger("mcp-obsidian.backend")


class BackendProxy:
    """Generic HTTP proxy for backend services."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 120.0,
    ):
        """Initialize the proxy.

        Args:
            base_url: Backend base URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Forward a POST request to the backend.

        Args:
            endpoint: API endpoint path (e.g., "/api/smart-search-vault")
            payload: JSON payload to send

        Returns:
            Response JSON as dict

        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get(self, endpoint: str) -> Dict[str, Any]:
        """Forward a GET request to the backend.

        Args:
            endpoint: API endpoint path

        Returns:
            Response JSON as dict
        """
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def is_healthy(self) -> bool:
        """Check if backend is responding.

        Returns:
            True if backend responds to health check
        """
        try:
            health = self.get("/health")
            return health.get("status") == "healthy"
        except Exception:
            return False


# Global singleton
_proxy: BackendProxy | None = None


def get_backend_proxy() -> BackendProxy:
    """Get or create the backend proxy singleton.

    Configurable via RAG_BACKEND_URL environment variable.

    Returns:
        BackendProxy instance
    """
    global _proxy
    if _proxy is None:
        base_url = os.getenv("RAG_BACKEND_URL", "http://localhost:8000")
        _proxy = BackendProxy(base_url=base_url)
    return _proxy
