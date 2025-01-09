# pathfinder_os/core/adaptation_manager.py

from typing import Dict, List, Set, Optional, Union
import numpy as np
from datetime import datetime
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import json

class AdaptationType(Enum):
    INTERFACE = "interface"
    COGNITIVE = "cognitive"
    BEHAVIORAL = "behavioral"
    ENVIRONMENTAL = "environmental"
    PERFORMANCE = "performance"
    EMOTIONAL = "emotional"

@dataclass
class AdaptationRule:
    rule_id: str
    type: AdaptationType
    conditions: Dict[str, Any]
    actions: Dict[str, Any]
    priority: float
    confidence: float
    success_rate: float
    usage_count: int
    last_triggered: Optional[datetime] = None

class AdaptationManager:
    def __init__(self, evolution_engine, event_bus):
        self.evolution_engine = evolution_engine
        self.event_bus = event_bus
        self.rules: Dict[str, AdaptationRule] = {}
        self.active_adaptations: Dict[str, Dict] = {}
        self.adaptation_history: List[Dict] = []
        self.learning_rate = 0.01
        self.context_window = []
        self.last_adaptation = datetime.now()

    async def initialize(self):
        """Initialize the adaptation system with base rules."""
        await self._load_base_rules()
        await self._initialize_context_tracking()
        await self.event_bus.subscribe("user_interaction", self.process_interaction)
        await self.event_bus.subscribe("system_metric_update", self.process_metrics)

    async def process_interaction(self, interaction_data: Dict):
        """Process user interaction for potential adaptations."""
        context = await self._build_context(interaction_data)
        self.context_window.append(context)
        
        # Maintain context window size
        if len(self.context_window) > 1000:
            self.context_window = self.context_window[-1000:]

        # Identify applicable adaptations
        applicable_rules = await self._identify_applicable_rules(context)
        
        # Execute adaptations
        for rule in applicable_rules:
            await self._execute_adaptation(rule, context)

    async def _build_context(self, interaction_data: Dict) -> Dict:
        """Build comprehensive context from interaction data."""
        return {
            "timestamp": datetime.now(),
            "interaction": interaction_data,
            "system_state": await self.evolution_engine.get_evolution_status(),
            "user_state": await self._get_user_state(),
            "environmental_factors": await self._get_environmental_factors(),
            "performance_metrics": await self._get_performance_metrics()
        }

    async def _execute_adaptation(self, rule: AdaptationRule, context: Dict):
        """Execute an adaptation rule and track its effects."""
        adaptation_id = f"adapt_{datetime.now().timestamp()}"
        
        try:
            # Prepare adaptation
            adaptation_plan = await self._prepare_adaptation(rule, context)
            
            # Execute adaptation actions
            success = await self._apply_adaptation_actions(adaptation_plan)
            
            if success:
                # Track successful adaptation
                self.active_adaptations[adaptation_id] = {
                    "rule": rule.rule_id,
                    "context": context,
                    "plan": adaptation_plan,
                    "start_time": datetime.now()
                }
                
                # Update rule metrics
                await self._update_rule_metrics(rule, True)
                
                # Notify system of adaptation
                await self.event_bus.publish(SystemEvent(
                    "adaptation_applied",
                    {"adaptation_id": adaptation_id, "rule_id": rule.rule_id}
                ))

        except Exception as e:
            await self._handle_adaptation_failure(rule, e)

    async def _prepare_adaptation(self, rule: AdaptationRule, context: Dict) -> Dict:
        """Prepare detailed adaptation plan based on rule and context."""
        return {
            "rule_id": rule.rule_id,
            "type": rule.type.value,
            "actions": await self._contextualize_actions(rule.actions, context),
            "rollback_plan": await self._create_rollback_plan(rule, context),
            "validation_criteria": await self._create_validation_criteria(rule),
            "expected_outcomes": await self._predict_outcomes(rule, context)
        }

    async def learn_new_rule(self, pattern: Dict):
        """Learn new adaptation rule from observed pattern."""
        rule_id = f"rule_{len(self.rules)}"
        
        new_rule = AdaptationRule(
            rule_id=rule_id,
            type=self._determine_rule_type(pattern),
            conditions=await self._extract_conditions(pattern),
            actions=await self._extract_actions(pattern),
            priority=0.5,
            confidence=0.1,
            success_rate=0.0,
            usage_count=0
        )
        
        self.rules[rule_id] = new_rule
        
        await self.event_bus.publish(SystemEvent(
            "adaptation_rule_created",
            {"rule_id": rule_id, "pattern": pattern}
        ))

    async def validate_adaptation_effects(self, adaptation_id: str):
        """Validate the effects of an applied adaptation."""
        if adaptation_id not in self.active_adaptations:
            return
        
        adaptation = self.active_adaptations[adaptation_id]
        rule = self.rules[adaptation["rule"]["rule_id"]]
        
        # Collect validation metrics
        current_state = await self._build_context({})
        validation_results = await self._validate_adaptation_outcomes(
            adaptation["plan"]["expected_outcomes"],
            current_state
        )
        
        if validation_results["success"]:
            # Update rule confidence and success rate
            rule.confidence = min(1.0, rule.confidence + self.learning_rate)
            rule.success_rate = (rule.success_rate * rule.usage_count + 1) / (rule.usage_count + 1)
        else:
            # Rollback if necessary
            await self._execute_rollback(adaptation)
            rule.confidence = max(0.0, rule.confidence - self.learning_rate)
            rule.success_rate = (rule.success_rate * rule.usage_count) / (rule.usage_count + 1)

        # Archive adaptation history
        self.adaptation_history.append({
            "adaptation_id": adaptation_id,
            "rule_id": rule.rule_id,
            "context": adaptation["context"],
            "results": validation_results,
            "timestamp": datetime.now()
        })
        
        del self.active_adaptations[adaptation_id]

    async def _validate_adaptation_outcomes(self, expected_outcomes: Dict, current_state: Dict) -> Dict:
        """Validate adaptation outcomes against expectations."""
        validation_results = {
            "success": True,
            "metrics": {},
            "discrepancies": []
        }
        
        for metric, expected_value in expected_outcomes.items():
            actual_value = await self._extract_metric_value(current_state, metric)
            difference = abs(expected_value - actual_value)
            
            validation_results["metrics"][metric] = {
                "expected": expected_value,
                "actual": actual_value,
                "difference": difference
            }
            
            if difference > self._get_threshold_for_metric(metric):
                validation_results["success"] = False
                validation_results["discrepancies"].append(metric)

        return validation_results

    async def optimize_rules(self):
        """Optimize adaptation rules based on historical performance."""
        for rule_id, rule in self.rules.items():
            if rule.usage_count > 10:  # Minimum sample size for optimization
                # Analyze rule performance
                performance_metrics = await self._analyze_rule_performance(rule)
                
                # Optimize conditions
                rule.conditions = await self._optimize_conditions(
                    rule.conditions,
                    performance_metrics
                )
                
                # Optimize actions
                rule.actions = await self._optimize_actions(
                    rule.actions,
                    performance_metrics
                )
                
                # Update priority based on success rate and impact
                rule.priority = await self._calculate_rule_priority(
                    rule,
                    performance_metrics
                )

    async def save_state(self) -> Dict:
        """Save current adaptation state."""
        return {
            "rules": {
                rule_id: {
                    "type": rule.type.value,
                    "conditions": rule.conditions,
                    "actions": rule.actions,
                    "priority": rule.priority,
                    "confidence": rule.confidence,
                    "success_rate": rule.success_rate,
                    "usage_count": rule.usage_count,
                    "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None
                }
                for rule_id, rule in self.rules.items()
            },
            "active_adaptations": self.active_adaptations,
            "adaptation_history": self.adaptation_history,
            "learning_rate": self.learning_rate
        }

    @classmethod
    async def load_state(cls, evolution_engine, event_bus, state_data: Dict) -> 'AdaptationManager':
        """Load adaptation manager from saved state."""
        manager = cls(evolution_engine, event_bus)
        
        # Restore rules
        for rule_id, rule_data in state_data["rules"].items():
            manager.rules[rule_id] = AdaptationRule(
                rule_id=rule_id,
                type=AdaptationType(rule_data["type"]),
                conditions=rule_data["conditions"],
                actions=rule_data["actions"],
                priority=rule_data["priority"],
                confidence=rule_data["confidence"],
                success_rate=rule_data["success_rate"],
                usage_count=rule_data["usage_count"],
                last_triggered=datetime.fromisoformat(rule_data["last_triggered"]) 
                    if rule_data["last_triggered"] else None
            )
        
        manager.active_adaptations = state_data["active_adaptations"]
        manager.adaptation_history = state_data["adaptation_history"]
        manager.learning_rate = state_data["learning_rate"]
        
        return manager