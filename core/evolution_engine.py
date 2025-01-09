# pathfinder_os/core/evolution_engine.py

from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
import asyncio
from dataclasses import dataclass
from enum import Enum

class EvolutionStage(Enum):
    SEED = "seed"              # Initial blank state
    SPROUTING = "sprouting"    # Basic personalization beginning
    GROWING = "growing"        # Active learning and adaptation
    FLOWERING = "flowering"    # Advanced personalization
    FRUITING = "fruiting"     # Generating new capabilities

@dataclass
class SystemCapability:
    name: str
    status: str
    confidence: float
    usage_frequency: float
    last_used: datetime
    dependencies: List[str]
    evolution_data: Dict[str, Any]

class EvolutionEngine:
    def __init__(self, core_system):
        self.core = core_system
        self.current_stage = EvolutionStage.SEED
        self.capabilities: Dict[str, SystemCapability] = {}
        self.evolution_history = []
        self.learning_rate = 0.01
        self.adaptation_threshold = 0.75
        self.last_evolution = datetime.now()

    async def initialize(self):
        """Initialize the evolution engine with basic seed capabilities."""
        seed_capabilities = {
            "basic_interaction": SystemCapability(
                name="basic_interaction",
                status="active",
                confidence=1.0,
                usage_frequency=1.0,
                last_used=datetime.now(),
                dependencies=[],
                evolution_data={"interaction_count": 0}
            ),
            "user_profiling": SystemCapability(
                name="user_profiling",
                status="learning",
                confidence=0.1,
                usage_frequency=0.0,
                last_used=datetime.now(),
                dependencies=["basic_interaction"],
                evolution_data={"profile_depth": 0}
            )
        }
        self.capabilities.update(seed_capabilities)

    async def process_interaction(self, interaction_data: Dict) -> None:
        """Process user interaction and evolve system capabilities."""
        # Update interaction statistics
        await self._update_usage_metrics(interaction_data)
        
        # Analyze interaction for potential evolution
        evolution_opportunities = await self._analyze_evolution_opportunities(interaction_data)
        
        # Evolve system based on opportunities
        if evolution_opportunities:
            await self._evolve_capabilities(evolution_opportunities)

    async def _update_usage_metrics(self, interaction_data: Dict) -> None:
        """Update usage metrics for involved capabilities."""
        capability = interaction_data.get("capability")
        if capability in self.capabilities:
            cap = self.capabilities[capability]
            cap.usage_frequency = (cap.usage_frequency * 0.9) + 0.1
            cap.last_used = datetime.now()
            cap.evolution_data["interaction_count"] += 1

    async def _analyze_evolution_opportunities(self, interaction_data: Dict) -> List[Dict]:
        """Analyze interaction data for potential system evolution."""
        opportunities = []
        
        # Analyze user proficiency
        proficiency = await self._calculate_user_proficiency(interaction_data)
        
        # Analyze system capability gaps
        capability_gaps = await self._identify_capability_gaps(interaction_data)
        
        # Analyze interaction patterns
        pattern_insights = await self._analyze_interaction_patterns(interaction_data)
        
        # Combine analyses to identify evolution opportunities
        if proficiency > self.adaptation_threshold:
            opportunities.extend(await self._generate_advancement_opportunities(proficiency))
            
        if capability_gaps:
            opportunities.extend(await self._generate_gap_filling_opportunities(capability_gaps))
            
        return opportunities

    async def _evolve_capabilities(self, opportunities: List[Dict]) -> None:
        """Evolve system capabilities based on identified opportunities."""
        for opportunity in opportunities:
            capability_name = opportunity["capability"]
            evolution_type = opportunity["type"]
            
            if evolution_type == "new":
                await self._create_new_capability(opportunity)
            elif evolution_type == "enhance":
                await self._enhance_existing_capability(opportunity)
            elif evolution_type == "merge":
                await self._merge_capabilities(opportunity)

        # Update evolution stage if necessary
        await self._update_evolution_stage()

    async def _create_new_capability(self, opportunity: Dict) -> None:
        """Create a new system capability."""
        capability = SystemCapability(
            name=opportunity["capability"],
            status="learning",
            confidence=0.1,
            usage_frequency=0.0,
            last_used=datetime.now(),
            dependencies=opportunity.get("dependencies", []),
            evolution_data={"creation_context": opportunity}
        )
        self.capabilities[capability.name] = capability
        
        await self.core.event_bus.publish(SystemEvent(
            "capability_created",
            {"capability": capability.name, "context": opportunity}
        ))

    async def _enhance_existing_capability(self, opportunity: Dict) -> None:
        """Enhance an existing capability based on evolution opportunity."""
        capability = self.capabilities[opportunity["capability"]]
        enhancement = opportunity["enhancement"]
        
        # Apply enhancement
        capability.evolution_data.update(enhancement)
        capability.confidence = min(1.0, capability.confidence + self.learning_rate)
        
        await self.core.event_bus.publish(SystemEvent(
            "capability_enhanced",
            {"capability": capability.name, "enhancement": enhancement}
        ))

    async def _update_evolution_stage(self) -> None:
        """Update the system's evolution stage based on capabilities and usage."""
        capability_metrics = await self._calculate_capability_metrics()
        
        if capability_metrics["advanced_capabilities_ratio"] > 0.7:
            self.current_stage = EvolutionStage.FRUITING
        elif capability_metrics["active_capabilities_ratio"] > 0.5:
            self.current_stage = EvolutionStage.FLOWERING
        elif capability_metrics["learning_capabilities_ratio"] > 0.3:
            self.current_stage = EvolutionStage.GROWING
        elif capability_metrics["basic_capabilities_ratio"] > 0.1:
            self.current_stage = EvolutionStage.SPROUTING

    async def get_evolution_status(self) -> Dict:
        """Get current evolution status and metrics."""
        return {
            "stage": self.current_stage.value,
            "capabilities_count": len(self.capabilities),
            "active_capabilities": len([c for c in self.capabilities.values() if c.status == "active"]),
            "learning_capabilities": len([c for c in self.capabilities.values() if c.status == "learning"]),
            "average_confidence": np.mean([c.confidence for c in self.capabilities.values()]),
            "last_evolution": self.last_evolution.isoformat(),
            "evolution_history_length": len(self.evolution_history)
        }

    async def save_state(self) -> Dict:
        """Save current evolution state for persistence."""
        return {
            "stage": self.current_stage.value,
            "capabilities": {
                name: {
                    "status": cap.status,
                    "confidence": cap.confidence,
                    "usage_frequency": cap.usage_frequency,
                    "last_used": cap.last_used.isoformat(),
                    "dependencies": cap.dependencies,
                    "evolution_data": cap.evolution_data
                }
                for name, cap in self.capabilities.items()
            },
            "evolution_history": self.evolution_history,
            "learning_rate": self.learning_rate,
            "adaptation_threshold": self.adaptation_threshold,
            "last_evolution": self.last_evolution.isoformat()
        }

    @classmethod
    async def load_state(cls, core_system, state_data: Dict) -> 'EvolutionEngine':
        """Load evolution engine from saved state."""
        engine = cls(core_system)
        engine.current_stage = EvolutionStage(state_data["stage"])
        engine.learning_rate = state_data["learning_rate"]
        engine.adaptation_threshold = state_data["adaptation_threshold"]
        engine.last_evolution = datetime.fromisoformat(state_data["last_evolution"])
        
        # Restore capabilities
        for name, cap_data in state_data["capabilities"].items():
            engine.capabilities[name] = SystemCapability(
                name=name,
                status=cap_data["status"],
                confidence=cap_data["confidence"],
                usage_frequency=cap_data["usage_frequency"],
                last_used=datetime.fromisoformat(cap_data["last_used"]),
                dependencies=cap_data["dependencies"],
                evolution_data=cap_data["evolution_data"]
            )
        
        engine.evolution_history = state_data["evolution_history"]
        return engine