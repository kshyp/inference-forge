"""Base agent class for Inference Forge.

Provides common functionality: checkpointing, health monitoring, event loop.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from uuid import UUID

from forge.core.checkpoint import Checkpointable
from forge.core.events import AgentState, AgentStatus, Task
from forge.core.health import HealthServer
from forge.core.state import StateStore


class BaseAgent(ABC, Checkpointable):
    """Abstract base class for all agents.
    
    Agents must implement:
    - execute(task): Main execution logic
    - get_checkpoint_data(): Serialize state for recovery
    - restore_from_checkpoint(data): Deserialize state
    
    Example:
        class MyAgent(BaseAgent):
            async def execute(self, task: Task) -> Dict[str, Any]:
                # Do work
                return {"result": "success"}
            
            def get_checkpoint_data(self) -> Dict[str, Any]:
                return {"progress": self.progress}
            
            def restore_from_checkpoint(self, data: Dict[str, Any]) -> None:
                self.progress = data.get("progress", 0)
    """
    
    def __init__(
        self,
        agent_id: str,
        state_store: StateStore,
        health_port: int,
        health_host: str = "0.0.0.0"
    ):
        # Initialize checkpointable
        super().__init__(agent_id, state_store)
        
        self.agent_id = agent_id
        self.state_store = state_store
        
        # Health server
        self.health_server = HealthServer(
            agent_id=agent_id,
            port=health_port,
            host=health_host
        )
        self.health_server.set_status_provider(self.get_status)
        
        # State
        self.state = AgentState.IDLE
        self.current_task: Optional[Task] = None
        self.current_progress: float = 0.0
        self.error_message: Optional[str] = None
        
        # Timing
        self.start_time: float = time.time()
        self.task_start_time: Optional[float] = None
        
        # Task queue / callback
        self._task_callback: Optional[Callable[[Task], None]] = None
        self._stop_event = asyncio.Event()
    
    @abstractmethod
    async def execute(self, task: Task) -> Dict[str, Any]:
        """Execute a task.
        
        Must be implemented by subclass.
        
        Args:
            task: The task to execute
            
        Returns:
            Result dictionary
        """
        pass
    
    @abstractmethod
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Return serializable state for checkpointing."""
        pass
    
    @abstractmethod
    def restore_from_checkpoint(self, data: Dict[str, Any]) -> None:
        """Restore state from checkpoint data."""
        pass
    
    def get_status(self) -> AgentStatus:
        """Get current agent status for health checks."""
        return AgentStatus(
            agent_id=self.agent_id,
            state=self.state,
            current_task=self.current_task.summary() if self.current_task else None,
            progress_percent=self.current_progress,
            uptime_seconds=time.time() - self.start_time,
            last_checkpoint=self.last_checkpoint_time,
            error_message=self.error_message
        )
    
    def set_task_callback(self, callback: Callable[[Task], None]) -> None:
        """Set callback for when a task is completed.
        
        The callback receives the task with result attached.
        """
        self._task_callback = callback
    
    async def run(self) -> None:
        """Main agent event loop.
        
        Override this if you need custom task acquisition logic.
        Default implementation polls get_next_task().
        """
        # Start health server
        await self.health_server.start()
        self.start_time = time.time()
        
        try:
            while not self._stop_event.is_set():
                # Get next task
                task = await self.get_next_task()
                if task is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute task with checkpointing
                await self.execute_with_checkpoints(task)
                
        except asyncio.CancelledError:
            pass
        finally:
            await self.health_server.stop()
    
    async def run_single_task(self, task: Task) -> Dict[str, Any]:
        """Run a single task and return result (for testing/debugging).
        
        Doesn't start the full event loop, just executes one task.
        Health server is managed externally (started/stopped by caller).
        """
        result = await self.execute_with_checkpoints(task)
        return result
    
    async def execute_with_checkpoints(self, task: Task) -> Dict[str, Any]:
        """Execute a task with checkpointing support.
        
        Handles:
        - State transitions
        - Checkpoint before/during/after
        - Error handling
        - Result callback
        """
        self.current_task = task
        self.state = AgentState.RUNNING
        self.current_progress = 0.0
        self.error_message = None
        self.task_start_time = time.time()
        
        result: Dict[str, Any] = {}
        
        try:
            # Check for existing checkpoint
            if await self.recover_if_needed(task.id):
                await self.log_event("task_recovered", task)
            else:
                await self.log_event("task_started", task)
            
            # Start auto-checkpointing
            self.start_auto_checkpoint(task.id)
            
            # Execute
            result = await self.execute(task)
            result["_task_id"] = str(task.id)
            result["_agent_id"] = self.agent_id
            result["_success"] = True
            
            self.state = AgentState.COMPLETED
            await self.log_event("task_completed", task, {"result": result})
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.error_message = str(e)
            result = {
                "_task_id": str(task.id),
                "_agent_id": self.agent_id,
                "_success": False,
                "_error": str(e),
            }
            await self.log_event("task_failed", task, {"error": str(e)})
            
        finally:
            # Stop auto-checkpointing
            self.stop_auto_checkpoint()
            
            # Final checkpoint (or clear if completed)
            if self.state == AgentState.COMPLETED:
                await self.clear_checkpoint(task.id)
            else:
                await self.checkpoint(task.id)
            
            # Call callback if set
            if self._task_callback:
                try:
                    self._task_callback(task)
                except Exception as e:
                    # Log but don't fail the task
                    await self.log_event("callback_failed", task, {"error": str(e)})
            
            # Reset state
            self.current_task = None
            self.current_progress = 0.0
            self.task_start_time = None
            if self.state != AgentState.ERROR:
                self.state = AgentState.IDLE
        
        return result
    
    async def get_next_task(self) -> Optional[Task]:
        """Get next task to execute.
        
        Override this for custom task acquisition (queue, pub/sub, etc.)
        Default returns None - agent idles.
        """
        return None
    
    def submit_task(self, task: Task) -> None:
        """Submit a task for execution.
        
        Override this if implementing in-memory task queue.
        """
        raise NotImplementedError("submit_task not implemented - override get_next_task instead")
    
    async def log_event(
        self,
        event_type: str,
        task: Optional[Task] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an event to the state store."""
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.state_store.log_event,
            event_type,
            self.agent_id,
            task.id if task else None,
            payload
        )
    
    async def stop(self) -> None:
        """Signal the agent to stop."""
        self._stop_event.set()
        await self.health_server.stop()
    
    async def start_health(self) -> None:
        """Start the health server."""
        if not self.health_server.is_running:
            await self.health_server.start()
    
    async def stop_health(self) -> None:
        """Stop the health server."""
        await self.health_server.stop()
    
    def update_progress(self, percent: float) -> None:
        """Update progress (0-100)."""
        self.current_progress = max(0.0, min(100.0, percent))
    
    @property
    def health_url(self) -> str:
        """Get health endpoint URL."""
        return self.health_server.url
