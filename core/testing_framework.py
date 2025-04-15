from typing import Dict


class IntegratedTestFramework:
    def __init__(self, core_system):
        self.core_system = core_system
        self.test_generator = TestGenerator()
        self.scenario_runner = ScenarioRunner()
        self.result_analyzer = ResultAnalyzer()
        self.regression_tester = RegressionTester()
        self.neurodiversity_validator = NeurodiversityValidator()
        self.accessibility_tester = AccessibilityTester()
        self.performance_tester = PerformanceTester()

    async def _run_adaptive_learning_tests(self) -> Dict:
        """Test the adaptive learning system."""
        return await self.core_system.learning_system.test_adaptive_capabilities()

    async def _run_interface_evolution_tests(self) -> Dict:
        """Test the interface evolution system."""
        return await self.core_system.interface_evolution.test_interface_generation()

    async def _run_multimodal_input_tests(self) -> Dict:
        """Test the multimodal input functionalities."""
        return {
            "voice_input": await self._test_voice_input(),
            "gesture_recognition": await self._test_gesture_recognition()
        }

    async def _test_voice_input(self) -> bool:
        """Test voice input functionality."""
        # Placeholder for voice input test logic.
        return True

    async def _test_gesture_recognition(self) -> bool:
        """Test gesture recognition functionality."""
        # Placeholder for gesture recognition test logic.
        return True

    async def _run_neurodiversity_tests(self) -> Dict:
        """Test system's neurodiversity support capabilities."""
        return await self.neurodiversity_validator.validate({
            "cognitive_load": await self._test_cognitive_load_management(),
            "sensory_processing": await self._test_sensory_adaptations(),
            "communication": await self._test_communication_support(),
            "executive_function": await self._test_executive_support()
        })

    async def run_comprehensive_tests(self) -> Dict:
        """Run comprehensive system tests with focus on neurodiversity support."""
        test_results = {
            "functional": await self._run_functional_tests(),
            "accessibility": await self._run_accessibility_tests(),
            "performance": await self._run_performance_tests(),
            "integration": await self._run_integration_tests(),
            "neurodiversity": await self._run_neurodiversity_tests(),
            "adaptive_learning": await self._run_adaptive_learning_tests(),
            "interface_evolution": await self._run_interface_evolution_tests(),
            "multimodal_input": await self._run_multimodal_input_tests()
        }

        analysis = await self.result_analyzer.analyze_results(test_results)
        recommendations = await self._generate_improvements(analysis)

        return {
            "results": test_results,
            "analysis": analysis,
            "recommendations": recommendations
        }