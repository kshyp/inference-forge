"""Synthesis engine for ranking experiments from SME responses.

Takes multiple SME opinions and produces a unified, ranked experiment plan.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from difflib import SequenceMatcher

from forge.smes.base import SMEResponse, ExperimentSuggestion
from forge.core.events import RankedExperiment, ExperimentPlan

logger = logging.getLogger(__name__)


@dataclass
class SynthesisConfig:
    """Configuration for experiment synthesis."""
    # Minimum confidence to include a suggestion
    min_confidence: float = 0.5
    
    # Maximum number of experiments to recommend
    max_experiments: int = 5
    
    # Weight for SME confidence vs consensus agreement
    confidence_weight: float = 0.6
    agreement_weight: float = 0.4
    
    # Whether to merge similar suggestions
    merge_similar: bool = True
    
    # Similarity threshold for merging (0.0-1.0)
    merge_threshold: float = 0.8


@dataclass
class MergedExperiment:
    """An experiment candidate merged from multiple suggestions."""
    config_changes: Dict[str, Any]
    sources: List[Tuple[str, float]]  # [(sme_id, confidence), ...]
    expected_improvements: List[str]
    rationales: List[str]
    
    @property
    def primary_sme(self) -> str:
        """Get the SME with highest confidence."""
        return max(self.sources, key=lambda x: x[1])[0]
    
    @property
    def avg_confidence(self) -> float:
        """Average confidence across all sources."""
        if not self.sources:
            return 0.0
        return sum(s[1] for s in self.sources) / len(self.sources)
    
    @property
    def consensus_score(self) -> float:
        """Score based on how many SMEs agree (0.0-1.0)."""
        # More unique SMEs = higher consensus
        unique_smes = len(set(s[0] for s in self.sources))
        # Normalize: assume 3+ SMEs is full consensus
        return min(unique_smes / 3.0, 1.0)


class ExperimentSynthesizer:
    """
    Synthesizes experiment recommendations from multiple SME responses.
    
    Algorithm:
    1. Collect all suggestions from all SMEs
    2. Group/merge similar suggestions by config_changes similarity
    3. Score each merged experiment:
       - SME confidence (weighted by model quality if available)
       - Consensus (how many SMEs agree)
       - Historical success (if available)
    4. Rank and select top N experiments
    5. Detect conflicts between experiments
    6. Generate ExperimentPlan
    """
    
    def __init__(self, config: Optional[SynthesisConfig] = None):
        self.config = config or SynthesisConfig()
    
    def synthesize(
        self,
        sme_responses: List[SMEResponse],
        sme_ids: List[str],
        current_config: Dict[str, Any],
        iteration: int = 1,
        parent_experiment_id: Optional[str] = None
    ) -> ExperimentPlan:
        """
        Synthesize SME responses into a ranked experiment plan.
        
        Args:
            sme_responses: List of responses from consulted SMEs
            sme_ids: IDs of SMEs that produced each response (parallel list)
            current_config: Current vLLM configuration
            iteration: Current optimization iteration number
            parent_experiment_id: ID of parent experiment (for tracking lineage)
            
        Returns:
            ExperimentPlan with ranked experiments
        """
        logger.info(f"Synthesizing {len(sme_responses)} SME responses")
        
        # Step 1: Collect all suggestions
        all_suggestions = []
        for sme_id, response in zip(sme_ids, sme_responses):
            for suggestion in response.suggestions:
                all_suggestions.append((sme_id, suggestion))
        
        logger.info(f"Total suggestions: {len(all_suggestions)}")
        
        if not all_suggestions:
            logger.warning("No suggestions to synthesize")
            return self._create_empty_plan(iteration, parent_experiment_id)
        
        # FALLBACK: If filtering would remove all suggestions, pick best raw suggestion
        # This ensures we always have something to try, even without consensus
        if all(sug.confidence < self.config.min_confidence for _, sug in all_suggestions):
            logger.warning(f"All suggestions below confidence threshold {self.config.min_confidence}")
            logger.warning("FALLBACK: Picking highest confidence suggestion anyway")
            best = max(all_suggestions, key=lambda x: x[1].confidence)
            return self._create_single_experiment_plan(
                best[0], best[1], current_config, iteration, 
                parent_experiment_id, sme_ids, sme_responses,
                fallback_reason="low_confidence"
            )
        
        # Step 2: Filter by minimum confidence
        filtered = [
            (sme_id, sug) for sme_id, sug in all_suggestions
            if sug.confidence >= self.config.min_confidence
        ]
        
        logger.info(f"After confidence filter (>{self.config.min_confidence}): {len(filtered)}")
        
        # Step 3: Merge similar suggestions
        if self.config.merge_similar:
            merged = self._merge_suggestions(filtered)
        else:
            merged = [
                MergedExperiment(
                    config_changes=sug.config_changes,
                    sources=[(sme_id, sug.confidence)],
                    expected_improvements=[sug.expected_improvement],
                    rationales=[sug.rationale]
                )
                for sme_id, sug in filtered
            ]
        
        logger.info(f"After merging: {len(merged)} unique experiments")
        
        # FALLBACK: If no experiments after filtering/merging, pick best from all
        if not merged:
            logger.warning("No experiments after filtering/merging")
            logger.warning("FALLBACK: Picking highest confidence raw suggestion")
            best = max(all_suggestions, key=lambda x: x[1].confidence)
            return self._create_single_experiment_plan(
                best[0], best[1], current_config, iteration,
                parent_experiment_id, sme_ids, sme_responses,
                fallback_reason="no_consensus"
            )
        
        # Step 4: Score and rank
        scored = self._score_experiments(merged)
        
        # Step 5: Select top N
        top_experiments = scored[:self.config.max_experiments]
        
        # Step 6: Detect conflicts
        conflicts = self._detect_conflicts(top_experiments)
        if conflicts:
            logger.warning(f"Detected {len(conflicts)} conflicts between experiments")
        
        # Step 7: Create ranked experiments
        ranked = self._create_ranked_experiments(
            top_experiments, 
            current_config,
            iteration
        )
        
        # Step 8: Build synthesis reasoning
        reasoning = self._build_reasoning(sme_responses, sme_ids, ranked)
        
        return ExperimentPlan(
            iteration=iteration,
            parent_experiment_id=parent_experiment_id,
            experiments=ranked,
            signals_detected=list(set(
                sig for resp in sme_responses 
                for sig in resp.findings.get("triggers", [])
            )),
            smes_consulted=sme_ids,
            synthesis_reasoning=reasoning
        )
    
    def _merge_suggestions(
        self, 
        suggestions: List[Tuple[str, ExperimentSuggestion]]
    ) -> List[MergedExperiment]:
        """Merge suggestions with similar config_changes."""
        clusters: List[MergedExperiment] = []
        
        for sme_id, suggestion in suggestions:
            # Find matching cluster
            merged = False
            for cluster in clusters:
                if self._configs_similar(
                    suggestion.config_changes, 
                    cluster.config_changes
                ):
                    # Merge into existing cluster
                    cluster.sources.append((sme_id, suggestion.confidence))
                    cluster.expected_improvements.append(suggestion.expected_improvement)
                    cluster.rationales.append(suggestion.rationale)
                    
                    # Merge config changes (take union)
                    cluster.config_changes.update(suggestion.config_changes)
                    merged = True
                    break
            
            if not merged:
                # Create new cluster
                clusters.append(MergedExperiment(
                    config_changes=dict(suggestion.config_changes),
                    sources=[(sme_id, suggestion.confidence)],
                    expected_improvements=[suggestion.expected_improvement],
                    rationales=[suggestion.rationale]
                ))
        
        return clusters
    
    def _configs_similar(
        self, 
        config_a: Dict[str, Any], 
        config_b: Dict[str, Any]
    ) -> bool:
        """Check if two configs are similar enough to merge."""
        if not config_a or not config_b:
            return False
        
        # Calculate similarity score
        keys_a = set(config_a.keys())
        keys_b = set(config_b.keys())
        
        # Key overlap
        common_keys = keys_a & keys_b
        all_keys = keys_a | keys_b
        
        if not all_keys:
            return True  # Both empty = similar
        
        key_score = len(common_keys) / len(all_keys)
        
        # Value similarity for common keys
        value_scores = []
        for key in common_keys:
            val_a = str(config_a[key]).lower().strip()
            val_b = str(config_b[key]).lower().strip()
            
            if val_a == val_b:
                value_scores.append(1.0)
            else:
                value_scores.append(
                    SequenceMatcher(None, val_a, val_b).ratio()
                )
        
        value_score = sum(value_scores) / len(value_scores) if value_scores else 0.0
        
        # Combined score
        similarity = 0.6 * key_score + 0.4 * value_score
        return similarity >= self.config.merge_threshold
    
    def _score_experiments(
        self, 
        experiments: List[MergedExperiment]
    ) -> List[Tuple[float, MergedExperiment]]:
        """Score and sort experiments by combined metric."""
        scored = []
        
        for exp in experiments:
            # Combined score = confidence * consensus
            score = (
                self.config.confidence_weight * exp.avg_confidence +
                self.config.agreement_weight * exp.consensus_score
            )
            scored.append((score, exp))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored
    
    def _detect_conflicts(
        self, 
        experiments: List[Tuple[float, MergedExperiment]]
    ) -> List[Tuple[int, int, str]]:
        """Detect conflicts between experiments."""
        conflicts = []
        
        for i, (score_i, exp_i) in enumerate(experiments):
            for j, (score_j, exp_j) in enumerate(experiments[i+1:], i+1):
                # Check for parameter conflicts
                common_params = set(exp_i.config_changes.keys()) & set(exp_j.config_changes.keys())
                
                for param in common_params:
                    val_i = exp_i.config_changes[param]
                    val_j = exp_j.config_changes[param]
                    
                    # Different values for same parameter = conflict
                    if val_i != val_j:
                        conflicts.append((
                            i, j, 
                            f"Conflict on '{param}': {val_i} vs {val_j}"
                        ))
        
        return conflicts
    
    def _create_ranked_experiments(
        self,
        scored_experiments: List[Tuple[float, MergedExperiment]],
        current_config: Dict[str, Any],
        iteration: int
    ) -> List[RankedExperiment]:
        """Convert merged experiments to RankedExperiment format."""
        ranked = []
        
        for priority, (score, exp) in enumerate(scored_experiments, 1):
            # Determine what to change vs current config
            config_patch = {}
            config_removals = []
            
            for key, value in exp.config_changes.items():
                if key not in current_config:
                    config_patch[key] = value
                elif current_config[key] != value:
                    config_patch[key] = value
                # If same as current, skip (no change needed)
            
            # Build hypothesis from rationales
            hypothesis = self._build_hypothesis(exp)
            
            # Build expected improvement (aggregate)
            expected = self._aggregate_expected_improvement(exp.expected_improvements)
            
            # Build success criteria
            success_criteria = self._build_success_criteria(exp)
            
            ranked.append(RankedExperiment(
                priority=priority,
                config_patch=config_patch,
                config_removals=config_removals,
                hypothesis=hypothesis,
                expected_improvement=expected,
                success_criteria=success_criteria,
                source_experts=list(set(s[0] for s in exp.sources)),
                confidence=exp.avg_confidence,
                abort_conditions=["oom", "error_rate > 0.1"]
            ))
        
        return ranked
    
    def _build_hypothesis(self, exp: MergedExperiment) -> str:
        """Build hypothesis string from merged experiment."""
        # Use the most common theme from rationales
        if len(exp.rationales) == 1:
            return exp.rationales[0][:200]
        
        # Multiple rationales - summarize
        sme_count = len(set(s[0] for s in exp.sources))
        return (
            f"[{sme_count} SMEs agree] "
            f"{exp.rationales[0][:150]}..."
        )
    
    def _aggregate_expected_improvement(self, improvements: List[str]) -> str:
        """Aggregate multiple expected improvement strings."""
        if len(improvements) == 1:
            return improvements[0]
        
        # Look for numeric ranges and average them (simplified)
        # In practice, might want more sophisticated parsing
        return improvements[0] + f" (plus {len(improvements)-1} similar estimates)"
    
    def _build_success_criteria(self, exp: MergedExperiment) -> str:
        """Build success criteria based on expected improvements."""
        # Default criteria - can be enhanced with actual metric parsing
        return "throughput improvement > 0% AND no errors"
    
    def _build_reasoning(
        self,
        sme_responses: List[SMEResponse],
        sme_ids: List[str],
        ranked_experiments: List[RankedExperiment]
    ) -> str:
        """Build human-readable synthesis reasoning."""
        lines = [
            f"Synthesized {len(ranked_experiments)} experiments from {len(sme_ids)} SMEs.",
            "",
            "SME Findings:"
        ]
        
        for sme_id, response in zip(sme_ids, sme_responses):
            findings = response.findings
            bottleneck = findings.get('primary_bottleneck', 'unknown')
            lines.append(f"  - {sme_id}: {bottleneck} (confidence: {response.confidence:.2f})")
        
        lines.extend(["", "Ranked Experiments:"])
        
        for exp in ranked_experiments:
            lines.append(
                f"  #{exp.priority}: {exp.config_patch} "
                f"(confidence: {exp.confidence:.2f}, sources: {', '.join(exp.source_experts)})"
            )
        
        return "\n".join(lines)
    
    def _create_empty_plan(
        self, 
        iteration: int, 
        parent_experiment_id: Optional[str]
    ) -> ExperimentPlan:
        """Create an empty plan when no suggestions available."""
        return ExperimentPlan(
            iteration=iteration,
            parent_experiment_id=parent_experiment_id,
            experiments=[],
            signals_detected=[],
            smes_consulted=[],
            synthesis_reasoning="No suggestions from SMEs. Consider baseline exploration."
        )
    
    def _create_single_experiment_plan(
        self,
        sme_id: str,
        suggestion: Any,  # ExperimentSuggestion
        current_config: Dict[str, Any],
        iteration: int,
        parent_experiment_id: Optional[str],
        sme_ids: List[str],
        sme_responses: List[Any],  # List[SMEResponse]
        fallback_reason: str = "unknown"
    ) -> ExperimentPlan:
        """Create a plan with a single experiment (fallback mode)."""
        # Build config patch
        config_patch = {}
        for key, value in suggestion.config_changes.items():
            if key not in current_config or current_config[key] != value:
                config_patch[key] = value
        
        # Create single ranked experiment
        experiment = RankedExperiment(
            priority=1,
            config_patch=config_patch,
            config_removals=[],
            hypothesis=f"[FALLBACK - {fallback_reason}] {suggestion.rationale[:200]}",
            expected_improvement=suggestion.expected_improvement,
            success_criteria="throughput improvement > 0% AND no errors",
            source_experts=[sme_id],
            confidence=suggestion.confidence,
            abort_conditions=["oom", "error_rate > 0.1"]
        )
        
        # Build reasoning
        reasoning = (
            f"FALLBACK MODE ({fallback_reason}): No consensus reached among {len(sme_ids)} SMEs. "
            f"Picking highest confidence suggestion from {sme_id} (conf={suggestion.confidence:.2f}). "
            f"Config changes: {suggestion.config_changes}"
        )
        
        logger.warning(reasoning)
        
        return ExperimentPlan(
            iteration=iteration,
            parent_experiment_id=parent_experiment_id,
            experiments=[experiment],
            signals_detected=list(set(
                sig for resp in sme_responses
                for sig in resp.findings.get("triggers", [])
            )),
            smes_consulted=sme_ids,
            synthesis_reasoning=reasoning
        )
