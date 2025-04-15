import os
import sys

# Add the workspace directory to the Python path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import unittest
from core.testing_framework import IntegratedTestFramework
from core.system_core import PathfinderAgent, SystemConfig


class TestPathfinderSystem(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        self.agent = PathfinderAgent(SystemConfig())
        self.test_framework = IntegratedTestFramework(self.agent)

    def test_comprehensive_system(self):
        """Run comprehensive system tests."""
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self.test_framework.run_comprehensive_tests())
        self.assertIn("results", results)
        self.assertIn("analysis", results)
        self.assertIn("recommendations", results)
        print("Test Results:", results)


if __name__ == "__main__":
    unittest.main()
