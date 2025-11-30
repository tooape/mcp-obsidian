"""EmbeddingGemma backend client and manager.

This module provides:
- EmbeddingBackend: HTTP client for the FastAPI embedding service
- BackendManager: Auto-starts backend with zombie process detection
"""

import os
import signal
import subprocess
import time
from typing import List, Optional, Dict, Any
import requests
import logging

logger = logging.getLogger("mcp-obsidian.backend")


class EmbeddingBackend:
    """HTTP client for the EmbeddingGemma FastAPI backend."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ):
        """Initialize the backend client.

        Args:
            base_url: Backend base URL (default: http://localhost:8000)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> Dict[str, Any]:
        """Check if backend is running and healthy.

        Returns:
            Health status dict with model info

        Raises:
            Exception if backend is not reachable
        """
        response = requests.get(
            f"{self.base_url}/health",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def is_healthy(self) -> bool:
        """Quick check if backend is responding.

        Returns:
            True if backend responds to health check
        """
        try:
            health = self.health_check()
            return health.get("status") == "healthy"
        except Exception:
            return False

    def embed_texts(
        self,
        texts: List[str],
        note_ids: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            note_ids: Optional note IDs for caching
            use_cache: Whether to use cached embeddings

        Returns:
            Dict with embeddings, model info, and latency
        """
        payload = {
            "texts": texts,
            "use_cache": use_cache,
        }
        if note_ids:
            payload["note_ids"] = note_ids

        response = requests.post(
            f"{self.base_url}/api/embed",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def search(
        self,
        query: str,
        note_ids: List[str],
        top_k: int = 100,
    ) -> Dict[str, Any]:
        """Perform semantic search.

        Args:
            query: Search query text
            note_ids: Note IDs to search within
            top_k: Number of top results to return

        Returns:
            Dict with ranked results and latency
        """
        response = requests.post(
            f"{self.base_url}/api/search",
            json={
                "query": query,
                "note_ids": note_ids,
                "top_k": top_k,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """Rerank documents using cross-encoder.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top results to return

        Returns:
            Dict with reranked results and latency
        """
        response = requests.post(
            f"{self.base_url}/api/rerank",
            json={
                "query": query,
                "documents": documents,
                "top_k": top_k,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


class BackendManager:
    """Manages the Python embedding backend lifecycle.

    Handles:
    - Auto-starting the backend
    - Detecting and killing zombie processes
    - Health monitoring
    """

    def __init__(
        self,
        backend_dir: str = os.path.expanduser("~/obsidianrag/python-backend"),
        python_path: str = "python3",
        port: int = 8000,
        host: str = "127.0.0.1",
    ):
        """Initialize the backend manager.

        Args:
            backend_dir: Directory containing the FastAPI backend
            python_path: Path to Python interpreter
            port: Port to run backend on
            host: Host to bind to
        """
        self.backend_dir = backend_dir
        self.python_path = python_path
        self.port = port
        self.host = host
        self.process: Optional[subprocess.Popen] = None
        self.client = EmbeddingBackend(base_url=f"http://{host}:{port}")

    def _find_process_on_port(self) -> Optional[int]:
        """Find process ID using the configured port.

        Returns:
            PID if found, None otherwise
        """
        try:
            # Use lsof to find process on port (macOS/Linux)
            result = subprocess.run(
                ["lsof", "-ti", f":{self.port}"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                # May return multiple PIDs, take first
                pid_str = result.stdout.strip().split("\n")[0]
                return int(pid_str)
        except Exception as e:
            logger.debug(f"Could not find process on port {self.port}: {e}")
        return None

    def _kill_zombie(self) -> bool:
        """Kill any zombie process on the backend port.

        Returns:
            True if a process was killed
        """
        pid = self._find_process_on_port()
        if pid:
            logger.info(f"Found zombie process {pid} on port {self.port}, killing...")
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                # Check if still running, force kill if needed
                try:
                    os.kill(pid, 0)  # Check if process exists
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(0.5)
                except ProcessLookupError:
                    pass  # Process already dead
                logger.info(f"Killed zombie process {pid}")
                return True
            except Exception as e:
                logger.warning(f"Failed to kill zombie process {pid}: {e}")
        return False

    def _spawn_backend(self) -> bool:
        """Spawn a new backend process.

        Returns:
            True if process started successfully
        """
        try:
            # Check if venv exists
            venv_python = os.path.join(self.backend_dir, "venv", "bin", "python")
            if os.path.exists(venv_python):
                python_cmd = venv_python
            else:
                python_cmd = self.python_path

            # Start the backend
            self.process = subprocess.Popen(
                [
                    python_cmd,
                    "-m", "uvicorn",
                    "app.main:app",
                    "--host", self.host,
                    "--port", str(self.port),
                ],
                cwd=self.backend_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # Detach from parent
            )
            logger.info(f"Started backend process (PID: {self.process.pid})")

            # Wait for backend to be ready
            for _ in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                if self.client.is_healthy():
                    logger.info("Backend is healthy and ready")
                    return True

            logger.warning("Backend started but not responding to health checks")
            return False

        except Exception as e:
            logger.error(f"Failed to spawn backend: {e}")
            return False

    def ensure_running(self) -> bool:
        """Ensure the backend is running, starting it if necessary.

        Handles zombie detection and cleanup.

        Returns:
            True if backend is running and healthy
        """
        # 1. Check if backend responds on health endpoint
        if self.client.is_healthy():
            logger.debug("Backend already running and healthy")
            return True

        # 2. Check for and kill zombie process
        self._kill_zombie()
        time.sleep(0.5)

        # 3. Start fresh backend process
        return self._spawn_backend()

    def stop(self) -> None:
        """Stop the backend process if we started it."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
            logger.info("Backend stopped")

    def get_client(self) -> EmbeddingBackend:
        """Get the backend client, ensuring backend is running.

        Returns:
            EmbeddingBackend client instance

        Raises:
            RuntimeError if backend cannot be started
        """
        if not self.ensure_running():
            raise RuntimeError("Failed to start embedding backend")
        return self.client


# Global singleton for backend manager
_backend_manager: Optional[BackendManager] = None


def get_backend_manager() -> BackendManager:
    """Get or create the global backend manager.

    Returns:
        BackendManager singleton instance
    """
    global _backend_manager
    if _backend_manager is None:
        backend_dir = os.getenv(
            "EMBEDDING_BACKEND_DIR",
            os.path.expanduser("~/obsidianrag/python-backend"),
        )
        _backend_manager = BackendManager(backend_dir=backend_dir)
    return _backend_manager


def get_embedding_client() -> EmbeddingBackend:
    """Get the embedding backend client, ensuring backend is running.

    This is the main entry point for other modules.

    Returns:
        EmbeddingBackend client instance

    Raises:
        RuntimeError if backend cannot be started
    """
    return get_backend_manager().get_client()
