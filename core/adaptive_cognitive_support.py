# Adaptive Cognitive Support Module
# This module provides dynamic load management, executive function support,
# and integration with neural pattern recognition to assist users in managing
# cognitive tasks effectively.

from typing import List, Dict
from datetime import datetime
import numpy as np

class CognitiveTask:
    """Represents a cognitive task with priority and estimated load."""
    def __init__(self, task_id: str, description: str, priority: int, estimated_load: float):
        self.task_id = task_id
        self.description = description
        self.priority = priority
        self.estimated_load = estimated_load
        self.created_at = datetime.now()

class CognitiveSupportManager:
    """Manages cognitive tasks and provides dynamic load management."""
    def __init__(self):
        self.tasks: List[CognitiveTask] = []

    def add_task(self, task: CognitiveTask):
        """Add a new cognitive task to the manager."""
        self.tasks.append(task)
        self.tasks.sort(key=lambda t: (t.priority, t.estimated_load))  # Sort by priority and load.

    def get_task_plan(self) -> List[Dict[str, str]]:
        """Generate a task plan based on current tasks."""
        return [
            {
                "task_id": task.task_id,
                "description": task.description,
                "priority": str(task.priority),
                "estimated_load": f"{task.estimated_load:.2f}"
            }
            for task in self.tasks
        ]

    def manage_load(self, max_load: float) -> List[CognitiveTask]:
        """Filter tasks to fit within the maximum cognitive load."""
        current_load = 0.0
        selected_tasks = []
        for task in self.tasks:
            if current_load + task.estimated_load <= max_load:
                selected_tasks.append(task)
                current_load += task.estimated_load
        return selected_tasks

# Example usage of the CognitiveSupportManager.
if __name__ == "__main__":
    manager = CognitiveSupportManager()
    manager.add_task(CognitiveTask("1", "Write project report", 1, 2.5))
    manager.add_task(CognitiveTask("2", "Prepare presentation", 2, 1.5))
    manager.add_task(CognitiveTask("3", "Email responses", 3, 0.5))

    print("Task Plan:", manager.get_task_plan())
    print("Managed Load:", [task.description for task in manager.manage_load(max_load=3.0)])
