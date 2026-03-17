"""Health check server for agents.

Each agent exposes a FastAPI endpoint for monitoring.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI
from uvicorn import Config, Server

from forge.core.events import AgentState, AgentStatus


class HealthServer:
    """FastAPI-based health check server for an agent.
    
    Usage:
        server = HealthServer(agent_id="benchmark", port=8081)
        server.set_status_provider(agent.get_status)
        await server.start()
        # ... agent runs ...
        await server.stop()
    """
    
    def __init__(
        self,
        agent_id: str,
        port: int,
        host: str = "0.0.0.0",
        log_level: str = "warning"
    ):
        self.agent_id = agent_id
        self.port = port
        self.host = host
        self.log_level = log_level
        
        self._status_provider: Optional[Callable[[], AgentStatus]] = None
        self._app: Optional[FastAPI] = None
        self._server: Optional[Server] = None
        self._server_task: Optional[asyncio.Task] = None
    
    def set_status_provider(self, provider: Callable[[], AgentStatus]) -> None:
        """Set the function that provides current agent status."""
        self._status_provider = provider
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            yield
            # Shutdown
        
        app = FastAPI(
            title=f"Inference Forge - {self.agent_id} Health",
            lifespan=lifespan,
            docs_url=None,  # Disable docs in production
            redoc_url=None,
        )
        
        @app.get("/health")
        async def health() -> Dict[str, Any]:
            """Get agent health status."""
            if self._status_provider:
                status = self._status_provider()
                return {
                    "agent_id": status.agent_id,
                    "state": status.state.value,
                    "current_task": status.current_task,
                    "progress_percent": status.progress_percent,
                    "uptime_seconds": status.uptime_seconds,
                    "last_checkpoint": status.last_checkpoint.isoformat() if status.last_checkpoint else None,
                    "error_message": status.error_message,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return {
                    "agent_id": self.agent_id,
                    "state": "unknown",
                    "error": "Status provider not set",
                }
        
        @app.get("/ready")
        async def ready() -> Dict[str, str]:
            """Readiness probe - returns 200 if agent can accept tasks."""
            if self._status_provider:
                status = self._status_provider()
                if status.state == AgentState.IDLE:
                    return {"status": "ready"}
                else:
                    return {"status": "busy"}
            return {"status": "unknown"}
        
        @app.get("/live")
        async def live() -> Dict[str, str]:
            """Liveness probe - returns 200 if agent is alive."""
            return {"status": "alive"}
        
        return app
    
    async def start(self) -> None:
        """Start the health server."""
        if self._server_task:
            return  # Already running
        
        self._app = self._create_app()
        
        config = Config(
            app=self._app,
            host=self.host,
            port=self.port,
            log_level=self.log_level,
            access_log=False,
        )
        
        self._server = Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
    
    async def stop(self) -> None:
        """Stop the health server."""
        if self._server:
            self._server.should_exit = True
            
        if self._server_task and not self._server_task.done():
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        
        self._server = None
        self._server_task = None
    
    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._server_task is not None and not self._server_task.done()
    
    @property
    def url(self) -> str:
        """Get health endpoint URL."""
        return f"http://{self.host}:{self.port}/health"


class HealthClient:
    """Client for checking agent health."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
    
    async def check(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Check agent health."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e), "status": "unreachable"}
    
    async def is_ready(self, timeout: float = 5.0) -> bool:
        """Check if agent is ready to accept tasks."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/ready",
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("status") == "ready"
                    return False
        except Exception:
            return False
    
    async def is_alive(self, timeout: float = 5.0) -> bool:
        """Check if agent is alive."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/live",
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    return response.status == 200
        except Exception:
            return False
