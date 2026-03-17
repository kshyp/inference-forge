"""Checkpoint mixin for agent crash recovery.

Provides save/load checkpoint functionality that integrates with StateStore.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from forge.core.state import StateStore


class Checkpointable:
    """Mixin for agents that support checkpointing.
    
    Agents must implement:
    - get_checkpoint_data(): Return serializable state
    - restore_from_checkpoint(data): Restore state from checkpoint
    
    Note: Not an ABC because it's used as a mixin with other ABCs.
    Subclasses should implement the abstract methods.
    """
    
    def __init__(self, agent_id: str, state_store: StateStore):
        self.agent_id = agent_id
        self.state_store = state_store
        self._last_checkpoint_time: Optional[datetime] = None
        self._checkpoint_task: Optional[asyncio.Task] = None
        self._auto_checkpoint_interval: float = 30.0  # seconds
    
    @abstractmethod
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Return serializable state for checkpointing.
        
        Must be implemented by agent.
        """
        pass
    
    @abstractmethod
    def restore_from_checkpoint(self, data: Dict[str, Any]) -> None:
        """Restore agent state from checkpoint data.
        
        Must be implemented by agent.
        """
        pass
    
    async def checkpoint(self, task_id: UUID) -> None:
        """Save checkpoint for current task."""
        data = self.get_checkpoint_data()
        data["_checkpoint_meta"] = {
            "agent_id": self.agent_id,
            "task_id": str(task_id),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Run in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.state_store.save_checkpoint,
            self.agent_id,
            task_id,
            data
        )
        
        self._last_checkpoint_time = datetime.now()
    
    async def load_checkpoint(self, task_id: UUID) -> Optional[Dict[str, Any]]:
        """Load checkpoint for task if exists."""
        data = await asyncio.get_event_loop().run_in_executor(
            None,
            self.state_store.load_checkpoint,
            self.agent_id,
            task_id
        )
        
        if data and "_checkpoint_meta" in data:
            # Remove metadata before returning
            data = {k: v for k, v in data.items() if not k.startswith("_")}
        
        return data
    
    async def clear_checkpoint(self, task_id: UUID) -> None:
        """Clear checkpoint after task completion."""
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.state_store.delete_checkpoint,
            self.agent_id,
            task_id
        )
    
    def start_auto_checkpoint(self, task_id: UUID) -> None:
        """Start periodic auto-checkpointing for a task."""
        if self._checkpoint_task and not self._checkpoint_task.done():
            self._checkpoint_task.cancel()
        
        self._checkpoint_task = asyncio.create_task(
            self._auto_checkpoint_loop(task_id)
        )
    
    def stop_auto_checkpoint(self) -> None:
        """Stop auto-checkpointing."""
        if self._checkpoint_task and not self._checkpoint_task.done():
            self._checkpoint_task.cancel()
            self._checkpoint_task = None
    
    async def _auto_checkpoint_loop(self, task_id: UUID) -> None:
        """Background task for periodic checkpoints."""
        try:
            while True:
                await asyncio.sleep(self._auto_checkpoint_interval)
                await self.checkpoint(task_id)
        except asyncio.CancelledError:
            pass  # Normal shutdown
    
    @property
    def last_checkpoint_time(self) -> Optional[datetime]:
        """Get timestamp of last checkpoint."""
        return self._last_checkpoint_time
    
    async def recover_if_needed(self, task_id: UUID) -> bool:
        """Attempt to recover from checkpoint.
        
        Returns True if recovery was successful, False if no checkpoint exists.
        """
        checkpoint = await self.load_checkpoint(task_id)
        if checkpoint:
            self.restore_from_checkpoint(checkpoint)
            return True
        return False
