# Integrated Project Management Module
# This module provides tools for managing decentralized organizations (DAOs),
# fostering project synergy, and efficiently allocating resources.

from typing import List, Dict
from datetime import datetime

class Project:
    """Represents a project with details and resources."""
    def __init__(self, project_id: str, name: str, resources: Dict[str, float]):
        self.project_id = project_id
        self.name = name
        self.resources = resources
        self.created_at = datetime.now()

class DAOManager:
    """Manages decentralized organizations and their projects."""
    def __init__(self):
        self.projects: List[Project] = []

    def create_project(self, project_id: str, name: str, resources: Dict[str, float]):
        """Create a new project and add it to the DAO."""
        project = Project(project_id, name, resources)
        self.projects.append(project)

    def allocate_resources(self, project_id: str, resource_name: str, amount: float) -> str:
        """Allocate resources to a specific project."""
        project = next((p for p in self.projects if p.project_id == project_id), None)
        if not project:
            return f"Project {project_id} not found."
        if resource_name in project.resources:
            project.resources[resource_name] += amount
        else:
            project.resources[resource_name] = amount
        return f"Allocated {amount} of {resource_name} to project {project_id}."

    def list_projects(self) -> List[Dict[str, str]]:
        """List all projects in the DAO."""
        return [
            {
                "project_id": project.project_id,
                "name": project.name,
                "resources": str(project.resources)
            }
            for project in self.projects
        ]

# Example usage of the DAOManager.
if __name__ == "__main__":
    manager = DAOManager()
    manager.create_project("1", "AI Research", {"budget": 10000})
    manager.create_project("2", "Community Building", {"budget": 5000})

    print("Projects:", manager.list_projects())
    print(manager.allocate_resources("1", "budget", 2000))
    print("Projects after allocation:", manager.list_projects())
