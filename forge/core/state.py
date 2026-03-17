"""SQLite persistence layer for Inference Forge.

Handles checkpoints, experiment history, and event logging.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from forge.core.events import Checkpoint


class StateStore:
    """SQLite-backed state management.
    
    All methods are async-compatible (run in thread pool if needed).
    For MVP, we use blocking sqlite3 with potential for async wrapper later.
    """
    
    def __init__(self, db_path: str = "./data/state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript("""
                -- Agent checkpoints for crash recovery
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    checkpoint_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(agent_id, task_id)
                );
                
                -- Experiment history
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    iteration INTEGER NOT NULL,
                    parent_experiment_id TEXT,
                    config_flags TEXT NOT NULL,
                    benchmark_results TEXT,
                    profiling_reports TEXT,
                    expert_opinions TEXT,
                    final_recommendations TEXT,
                    status TEXT DEFAULT 'running',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                );
                
                -- Event log for debugging/replay
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    agent_id TEXT,
                    task_id TEXT,
                    payload TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_checkpoints_agent_task 
                    ON checkpoints(agent_id, task_id);
                CREATE INDEX IF NOT EXISTS idx_experiments_status 
                    ON experiments(status);
                CREATE INDEX IF NOT EXISTS idx_events_agent 
                    ON events(agent_id, timestamp);
            """)
            conn.commit()
    
    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # Checkpoint methods
    
    def save_checkpoint(self, agent_id: str, task_id: UUID, data: Dict[str, Any]) -> None:
        """Save or update checkpoint for an agent/task.
        
        Uses INSERT OR REPLACE for upsert behavior.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints 
                (agent_id, task_id, checkpoint_data, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    agent_id,
                    str(task_id),
                    json.dumps(data),
                    datetime.now().isoformat()
                )
            )
            conn.commit()
    
    def load_checkpoint(self, agent_id: str, task_id: UUID) -> Optional[Dict[str, Any]]:
        """Load checkpoint for an agent/task."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT checkpoint_data FROM checkpoints
                WHERE agent_id = ? AND task_id = ?
                """,
                (agent_id, str(task_id))
            ).fetchone()
            
            if row:
                return json.loads(row["checkpoint_data"])
            return None
    
    def delete_checkpoint(self, agent_id: str, task_id: UUID) -> None:
        """Delete checkpoint after task completion."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM checkpoints WHERE agent_id = ? AND task_id = ?",
                (agent_id, str(task_id))
            )
            conn.commit()
    
    def list_checkpoints(self, agent_id: Optional[str] = None) -> List[Checkpoint]:
        """List all checkpoints, optionally filtered by agent."""
        with self._connect() as conn:
            if agent_id:
                rows = conn.execute(
                    """
                    SELECT agent_id, task_id, checkpoint_data, created_at
                    FROM checkpoints WHERE agent_id = ?
                    ORDER BY created_at DESC
                    """,
                    (agent_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT agent_id, task_id, checkpoint_data, created_at
                    FROM checkpoints ORDER BY created_at DESC
                    """
                ).fetchall()
            
            return [
                Checkpoint(
                    agent_id=row["agent_id"],
                    task_id=UUID(row["task_id"]),
                    data=json.loads(row["checkpoint_data"]),
                    created_at=datetime.fromisoformat(row["created_at"])
                )
                for row in rows
            ]
    
    # Experiment tracking
    
    def create_experiment(
        self, 
        experiment_id: UUID, 
        iteration: int,
        parent_id: Optional[UUID] = None,
        config_flags: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create a new experiment record."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments 
                (id, iteration, parent_experiment_id, config_flags, status)
                VALUES (?, ?, ?, ?, 'running')
                """,
                (
                    str(experiment_id),
                    iteration,
                    str(parent_id) if parent_id else None,
                    json.dumps(config_flags or {})
                )
            )
            conn.commit()
    
    def update_experiment(
        self,
        experiment_id: UUID,
        benchmark_results: Optional[Dict[str, Any]] = None,
        profiling_reports: Optional[List[str]] = None,
        expert_opinions: Optional[List[Dict]] = None,
        final_recommendations: Optional[Dict] = None,
        status: Optional[str] = None
    ) -> None:
        """Update experiment with results."""
        with self._connect() as conn:
            # Build dynamic update
            updates = []
            params = []
            
            if benchmark_results is not None:
                updates.append("benchmark_results = ?")
                params.append(json.dumps(benchmark_results))
            if profiling_reports is not None:
                updates.append("profiling_reports = ?")
                params.append(json.dumps(profiling_reports))
            if expert_opinions is not None:
                updates.append("expert_opinions = ?")
                params.append(json.dumps(expert_opinions))
            if final_recommendations is not None:
                updates.append("final_recommendations = ?")
                params.append(json.dumps(final_recommendations))
            if status is not None:
                updates.append("status = ?")
                params.append(status)
                if status in ("completed", "failed", "converged"):
                    updates.append("completed_at = ?")
                    params.append(datetime.now().isoformat())
            
            if updates:
                params.append(str(experiment_id))
                query = f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?"
                conn.execute(query, params)
                conn.commit()
    
    def get_experiment(self, experiment_id: UUID) -> Optional[Dict[str, Any]]:
        """Get full experiment record."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (str(experiment_id),)
            ).fetchone()
            
            if not row:
                return None
            
            return {
                "id": row["id"],
                "iteration": row["iteration"],
                "parent_experiment_id": row["parent_experiment_id"],
                "config_flags": json.loads(row["config_flags"]) if row["config_flags"] else {},
                "benchmark_results": json.loads(row["benchmark_results"]) if row["benchmark_results"] else None,
                "profiling_reports": json.loads(row["profiling_reports"]) if row["profiling_reports"] else None,
                "expert_opinions": json.loads(row["expert_opinions"]) if row["expert_opinions"] else None,
                "final_recommendations": json.loads(row["final_recommendations"]) if row["final_recommendations"] else None,
                "status": row["status"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
            }
    
    def list_experiments(
        self, 
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List experiments, optionally filtered by status."""
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM experiments 
                    WHERE status = ? 
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (status, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM experiments 
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,)
                ).fetchall()
            
            return [
                {
                    "id": row["id"],
                    "iteration": row["iteration"],
                    "status": row["status"],
                    "config_flags": json.loads(row["config_flags"]) if row["config_flags"] else {},
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
    
    def get_experiment_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get experiment history for pattern detection."""
        return self.list_experiments(limit=limit)
    
    # Event logging
    
    def log_event(
        self,
        event_type: str,
        agent_id: Optional[str] = None,
        task_id: Optional[UUID] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an event for debugging/replay."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (event_type, agent_id, task_id, payload)
                VALUES (?, ?, ?, ?)
                """,
                (
                    event_type,
                    agent_id,
                    str(task_id) if task_id else None,
                    json.dumps(payload) if payload else None
                )
            )
            conn.commit()
    
    def get_events(
        self,
        agent_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get events for debugging."""
        with self._connect() as conn:
            conditions = []
            params = []
            
            if agent_id:
                conditions.append("agent_id = ?")
                params.append(agent_id)
            if event_type:
                conditions.append("event_type = ?")
                params.append(event_type)
            
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            
            rows = conn.execute(
                f"""
                SELECT * FROM events
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (*params, limit)
            ).fetchall()
            
            return [
                {
                    "id": row["id"],
                    "event_type": row["event_type"],
                    "agent_id": row["agent_id"],
                    "task_id": row["task_id"],
                    "payload": json.loads(row["payload"]) if row["payload"] else None,
                    "timestamp": row["timestamp"],
                }
                for row in rows
            ]
    
    # Utility
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._connect() as conn:
            checkpoint_count = conn.execute(
                "SELECT COUNT(*) as count FROM checkpoints"
            ).fetchone()["count"]
            
            experiment_count = conn.execute(
                "SELECT COUNT(*) as count FROM experiments"
            ).fetchone()["count"]
            
            event_count = conn.execute(
                "SELECT COUNT(*) as count FROM events"
            ).fetchone()["count"]
            
            running_experiments = conn.execute(
                "SELECT COUNT(*) as count FROM experiments WHERE status = 'running'"
            ).fetchone()["count"]
            
            return {
                "checkpoints": checkpoint_count,
                "experiments": experiment_count,
                "events": event_count,
                "running_experiments": running_experiments,
            }
