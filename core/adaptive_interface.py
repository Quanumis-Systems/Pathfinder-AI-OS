# Adaptive Interface Module
# This module provides a unified, user-friendly interface for interacting with
# the Pathfinder AI OS. It integrates core functionalities and adapts to user
# interactions to provide a personalized experience.

from core.system_core import PathfinderAgent, SystemConfig
from typing import Dict, Any
import logging

# Importing additional core modules for integration.
from core.adaptive_cognitive_support import CognitiveSupportManager, CognitiveTask
from core.community_connection import CommunityManager, UserProfile
from core.integrated_project_management import DAOManager, Project

class AdaptiveInterface:
    """Unified interface for interacting with Pathfinder AI OS."""
    def __init__(self):
        self.agent = PathfinderAgent(SystemConfig())  # Initialize the Pathfinder agent.
        self.user_preferences: Dict[str, Any] = {}  # Store user preferences for adaptation.
        self.cognitive_support_manager = CognitiveSupportManager()  # Initialize cognitive support manager.
        self.community_manager = CommunityManager()  # Initialize community manager.
        self.dao_manager = DAOManager()  # Initialize DAO manager.

    def process_command(self, command: str, data: Dict[str, Any] = None) -> str:
        """Process user commands and route them to the appropriate functionality."""
        data = data or {}
        if command == "add_cognitive_task":
            task = CognitiveTask(
                task_id=data.get("task_id"),
                description=data.get("description"),
                priority=data.get("priority"),
                estimated_load=data.get("estimated_load")
            )
            self.cognitive_support_manager.add_task(task)
            return f"Cognitive task '{data.get('description')}' added successfully."
        elif command == "get_cognitive_task_plan":
            task_plan = self.cognitive_support_manager.get_task_plan()
            return f"Cognitive Task Plan: {task_plan}"
        elif command == "manage_cognitive_load":
            managed_tasks = self.cognitive_support_manager.manage_load(max_load=data.get("max_load"))
            return f"Managed Cognitive Tasks: {[task.description for task in managed_tasks]}"
        elif command == "add_user_profile":
            profile = UserProfile(
                user_id=data.get("user_id"),
                interests=data.get("interests"),
                preferences=data.get("preferences")
            )
            self.community_manager.add_user_profile(profile)
            return f"User profile for '{data.get('user_id')}' added successfully."
        elif command == "find_user_matches":
            matches = self.community_manager.find_matches(user_id=data.get("user_id"))
            return f"User Matches: {[profile.user_id for profile in matches]}"
        elif command == "create_safe_space":
            safe_space = self.community_manager.create_safe_space(topic=data.get("topic"))
            return safe_space
        elif command == "create_project":
            self.dao_manager.create_project(
                project_id=data.get("project_id"),
                name=data.get("name"),
                resources=data.get("resources")
            )
            return f"Project '{data.get('name')}' created successfully."
        elif command == "list_projects":
            projects = self.dao_manager.list_projects()
            return f"Projects: {projects}"
        elif command == "allocate_project_resources":
            result = self.dao_manager.allocate_resources(
                project_id=data.get("project_id"),
                resource_name=data.get("resource_name"),
                amount=data.get("amount")
            )
            return result
        elif command == "adapt_learning":
            result = self.agent.adapt_learning(data.get("user_data"))
            return f"Learning system adapted: {result}"
        elif command == "evolve_interface":
            interface_state = self.agent.evolve_interface(data.get("user_id"))
            return f"Interface evolved for user {data.get('user_id')}: {interface_state}"
        else:
            return "Unknown command. Please try again."

    def adapt_to_user(self, interaction_data: Dict[str, Any]):
        """Adapt the interface based on user interactions."""
        # Update user preferences based on interaction data.
        self.user_preferences.update(interaction_data)
        logging.info(f"User preferences updated: {self.user_preferences}")

# Example usage of the AdaptiveInterface.
if __name__ == "__main__":
    interface = AdaptiveInterface()

    # Simulate user commands.
    print(interface.process_command("add_task", {
        "task_id": "1",
        "description": "Write project report",
        "priority": 1,
        "estimated_load": 2.5
    }))

    print(interface.process_command("get_task_plan"))
    print(interface.process_command("manage_load", {"max_load": 3.0}))

    print(interface.process_command("create_project", {
        "project_id": "1",
        "name": "AI Research",
        "resources": {"budget": 10000}
    }))

    print(interface.process_command("list_projects"))
