"""Consensus engine for aggregating multi-LLM responses."""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from difflib import SequenceMatcher

from .pool import IntelligenceSource, CallResult

logger = logging.getLogger(__name__)


@dataclass
class ConsensusConfig:
    """Configuration for consensus computation."""
    # Similarity threshold for clustering suggestions (0.0-1.0)
    # Two suggestions are in same cluster if their config_changes similarity >= threshold
    similarity_threshold: float = 0.8
    
    # Minimum agreement ratio to include a suggestion (0.0-1.0)
    # e.g., 0.5 means at least 50% of models must agree
    min_agreement_ratio: float = 0.5
    
    # Whether to require unanimous agreement for priority 1 suggestions
    require_unanimous_for_p1: bool = False
    
    # Weight by model quality when computing confidence
    weight_by_quality: bool = True
    
    # Normalize config values for comparison
    normalize_values: bool = True
    
    # Temperature for LLM sampling (0.0 = deterministic, higher = more creative)
    # Default is 0.0 for reproducible results
    temperature: float = 0.0


@dataclass
class ParsedSuggestion:
    """A suggestion parsed from a model response."""
    priority: int
    config_changes: Dict[str, Any]
    expected_improvement: str
    confidence: float
    rationale: str
    
    # Metadata
    source_model: str
    source_provider: str
    model_quality: float = 1.0


@dataclass
class ConsensusSuggestion:
    """A suggestion with consensus metadata."""
    # The suggestion (from the "best" representative or merged)
    priority: int
    config_changes: Dict[str, Any]
    expected_improvement: str
    rationale: str
    
    # Consensus metrics
    supporting_models: List[str]  # ["anthropic/claude-3-7", "openai/gpt-4"]
    agreement_score: float  # 0.0-1.0 (fraction of models agreeing)
    weighted_confidence: float  # Combined confidence score
    
    # Source suggestions that formed this consensus
    source_suggestions: List[ParsedSuggestion] = field(default_factory=list)
    
    # Divergent models (for debugging)
    divergent_models: List[str] = field(default_factory=list)


@dataclass
class ModelResponse:
    """Simplified model response for consensus."""
    provider: str
    model: str
    suggestions: List[ParsedSuggestion] = field(default_factory=list)
    raw_findings: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error is None and len(self.suggestions) > 0
    
    @property
    def full_model_name(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass
class DivergenceReport:
    """Report on where models disagreed."""
    # Bottleneck disagreement
    bottleneck_votes: Dict[str, List[str]] = field(default_factory=dict)
    # Suggestions that only appeared from single models
    unique_suggestions: List[Tuple[str, str]] = field(default_factory=list)  # (model, suggestion_desc)
    # Overall agreement score
    overall_agreement: float = 0.0
    
    @property
    def summary(self) -> str:
        lines = []
        if self.bottleneck_votes:
            lines.append(f"Bottleneck votes: {dict(self.bottleneck_votes)}")
        if self.unique_suggestions:
            lines.append(f"Unique suggestions: {len(self.unique_suggestions)}")
        lines.append(f"Overall agreement: {self.overall_agreement:.0%}")
        return "; ".join(lines)


@dataclass
class ConsensusResult:
    """Complete result of consensus computation."""
    suggestions: List[ConsensusSuggestion]
    divergence_report: DivergenceReport
    model_responses: List[ModelResponse]
    
    # Metadata
    total_models: int = 0
    successful_models: int = 0
    failed_models: int = 0


class ConsensusEngine:
    """
    Computes consensus across multiple LLM responses.
    
    Algorithm:
    1. Parse all valid JSON responses into structured suggestions
    2. Normalize suggestions (canonical form for comparison)
    3. Cluster similar suggestions (config_changes similarity)
    4. Score clusters by: model_confidence × agreement × model_quality
    5. Sort by priority and confidence
    6. Generate divergence report
    """
    
    @staticmethod
    def compute(
        call_results: List[CallResult],
        config: ConsensusConfig
    ) -> ConsensusResult:
        """
        Compute consensus from parallel LLM call results.
        
        Args:
            call_results: Results from IntelligencePool.call_all()
            config: Consensus configuration
        
        Returns:
            ConsensusResult with aggregated suggestions
        """
        # Step 1: Parse all valid responses
        model_responses = ConsensusEngine._parse_responses(call_results)
        
        # Step 2: Flatten all suggestions
        all_suggestions = []
        for mr in model_responses:
            all_suggestions.extend(mr.suggestions)
        
        logger.info(
            f"Computing consensus: {len(model_responses)} models, "
            f"{len(all_suggestions)} total suggestions"
        )
        
        # Debug: print each model's suggestions
        for mr in model_responses:
            if mr.success:
                print(f"      [Consensus] {mr.full_model_name}: {len(mr.suggestions)} suggestions")
                for s in mr.suggestions:
                    print(f"         → {s.config_changes} (conf={s.confidence:.2f})")
            else:
                print(f"      [Consensus] {mr.full_model_name}: ERROR - {mr.error}")
        
        # Step 3: Cluster suggestions by similarity
        clusters = ConsensusEngine._cluster_suggestions(
            all_suggestions, 
            config.similarity_threshold
        )
        
        print(f"      [Consensus] Formed {len(clusters)} clusters from {len(all_suggestions)} suggestions")
        
        # Step 4: Convert clusters to consensus suggestions
        total_models = len(model_responses)
        consensus_suggestions = []
        
        # Adjust min_agreement_ratio for single-model scenarios
        effective_min_agreement = config.min_agreement_ratio
        if total_models == 1:
            # With only 1 model, any suggestion should be accepted
            effective_min_agreement = 0.0
            print(f"      [Consensus] Single model mode - accepting all valid suggestions")
        
        for i, cluster in enumerate(clusters):
            cs = ConsensusEngine._create_consensus_suggestion(
                cluster, 
                total_models,
                config
            )
            
            print(f"      [Consensus] Cluster {i+1}: agreement={cs.agreement_score:.2f}, min_required={effective_min_agreement}")
            
            # Include ALL suggestions, marking whether they meet consensus threshold
            # The synthesizer will rank them: consensus first, then by confidence
            if cs.agreement_score >= effective_min_agreement:
                print(f"         ✓ CONSENSUS: {cs.config_changes}")
            else:
                print(f"         → Individual: {cs.config_changes} (conf={cs.weighted_confidence:.2f})")
            
            consensus_suggestions.append(cs)
        
        # NOTE: We no longer filter out low-agreement suggestions.
        # All suggestions are included and will be ranked by the synthesizer:
        # - High agreement (consensus) suggestions get priority
        # - Within same agreement level, higher confidence comes first
        # This ensures we exhaust all experiments before re-consulting SMEs.
        
        # Step 5: Sort by priority, then by weighted confidence
        consensus_suggestions.sort(
            key=lambda x: (x.priority, -x.weighted_confidence)
        )
        
        # Step 6: Generate divergence report
        divergence = ConsensusEngine._create_divergence_report(
            model_responses, 
            clusters
        )
        
        return ConsensusResult(
            suggestions=consensus_suggestions,
            divergence_report=divergence,
            model_responses=model_responses,
            total_models=len(call_results),
            successful_models=len([r for r in call_results if r.success]),
            failed_models=len([r for r in call_results if not r.success])
        )
    
    @staticmethod
    def _parse_responses(call_results: List[CallResult]) -> List[ModelResponse]:
        """Parse JSON responses into structured format."""
        model_responses = []
        
        for result in call_results:
            source = result.source
            
            if not result.success or not result.response:
                error_msg = result.error or "No response"
                print(f"      [Consensus] {source.provider}/{source.model}: Call failed - {error_msg}")
                model_responses.append(ModelResponse(
                    provider=source.provider,
                    model=source.model,
                    error=error_msg
                ))
                continue
            
            try:
                content = result.response.content
                
                # Try to extract JSON (handle markdown code blocks)
                data = ConsensusEngine._extract_json(content)
                
                # Parse findings
                findings = data.get("findings", {})
                
                # Parse suggestions
                suggestions = []
                for sug_data in data.get("suggestions", []):
                    suggestions.append(ParsedSuggestion(
                        priority=sug_data.get("priority", 99),
                        config_changes=sug_data.get("config_changes", {}),
                        expected_improvement=sug_data.get("expected_improvement", ""),
                        confidence=sug_data.get("confidence", 0.5),
                        rationale=sug_data.get("rationale", ""),
                        source_model=source.model,
                        source_provider=source.provider,
                        model_quality=source.capabilities.get("quality_tier", 0.8)
                    ))
                
                model_responses.append(ModelResponse(
                    provider=source.provider,
                    model=source.model,
                    suggestions=suggestions,
                    raw_findings=findings
                ))
                
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse JSON from {source.provider}/{source.model}: {e}"
                )
                model_responses.append(ModelResponse(
                    provider=source.provider,
                    model=source.model,
                    error=f"JSON parse error: {e}"
                ))
            except Exception as e:
                logger.warning(
                    f"Error processing response from {source.provider}/{source.model}: {e}"
                )
                model_responses.append(ModelResponse(
                    provider=source.provider,
                    model=source.model,
                    error=f"Processing error: {e}"
                ))
        
        return model_responses
    
    @staticmethod
    def _extract_json(content: str) -> Dict[str, Any]:
        """Extract JSON from response (handle markdown code blocks)."""
        content = content.strip()
        
        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code block with language specifier
        if "```json" in content:
            start = content.find("```json") + 7  # Skip past "```json"
            # Skip newline after ```json if present
            if content[start:start+1] == "\n":
                start += 1
            end = content.find("```", start)
            if end > start:
                return json.loads(content[start:end].strip())
        
        # Try extracting from generic markdown code block
        if "```" in content:
            start = content.find("```") + 3
            # Skip any language specifier (like "json")
            lang_end = content.find("\n", start)
            if lang_end > start:
                start = lang_end + 1
            end = content.find("```", start)
            if end > start:
                return json.loads(content[start:end].strip())
        
        # Try finding JSON object bounds
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start:end+1])
        
        raise json.JSONDecodeError("Could not extract JSON", content, 0)
    
    @staticmethod
    def _cluster_suggestions(
        suggestions: List[ParsedSuggestion],
        threshold: float
    ) -> List[List[ParsedSuggestion]]:
        """Cluster suggestions by config_changes similarity."""
        clusters: List[List[ParsedSuggestion]] = []
        
        for suggestion in suggestions:
            # Find matching cluster
            merged = False
            for cluster in clusters:
                if ConsensusEngine._suggestions_similar(
                    suggestion, 
                    cluster[0],  # Compare to cluster representative
                    threshold
                ):
                    cluster.append(suggestion)
                    merged = True
                    break
            
            if not merged:
                clusters.append([suggestion])
        
        return clusters
    
    @staticmethod
    def _suggestions_similar(
        a: ParsedSuggestion, 
        b: ParsedSuggestion,
        threshold: float
    ) -> bool:
        """Check if two suggestions are similar enough to cluster."""
        # Compare config_changes
        similarity = ConsensusEngine._config_similarity(
            a.config_changes, 
            b.config_changes
        )
        return similarity >= threshold
    
    @staticmethod
    def _config_similarity(config_a: Dict[str, Any], config_b: Dict[str, Any]) -> float:
        """
        Compute similarity between two config dicts (0.0-1.0).
        
        Simple approach: Jaccard similarity on keys + value similarity
        """
        if not config_a and not config_b:
            return 1.0
        if not config_a or not config_b:
            return 0.0
        
        keys_a = set(config_a.keys())
        keys_b = set(config_b.keys())
        
        # Key overlap
        common_keys = keys_a & keys_b
        all_keys = keys_a | keys_b
        
        if not all_keys:
            return 1.0
        
        key_score = len(common_keys) / len(all_keys)
        
        # Value similarity for common keys
        value_scores = []
        for key in common_keys:
            val_a = str(config_a[key]).lower().strip()
            val_b = str(config_b[key]).lower().strip()
            
            if val_a == val_b:
                value_scores.append(1.0)
            else:
                # Use sequence matcher for string similarity
                value_scores.append(SequenceMatcher(None, val_a, val_b).ratio())
        
        value_score = sum(value_scores) / len(value_scores) if value_scores else 0.0
        
        # Combined score: 60% keys, 40% values
        return 0.6 * key_score + 0.4 * value_score
    
    @staticmethod
    def _create_consensus_suggestion(
        cluster: List[ParsedSuggestion],
        total_models: int,
        config: ConsensusConfig
    ) -> ConsensusSuggestion:
        """Create a ConsensusSuggestion from a cluster."""
        if not cluster:
            raise ValueError("Empty cluster")
        
        # Use the suggestion with highest model quality as representative
        representative = max(cluster, key=lambda x: x.model_quality)
        
        # Get supporting models
        supporting = [
            f"{s.source_provider}/{s.source_model}" 
            for s in cluster
        ]
        
        # Compute agreement score
        agreement_score = len(set(supporting)) / total_models if total_models > 0 else 0
        
        # Compute weighted confidence
        if config.weight_by_quality:
            total_weight = sum(s.model_quality for s in cluster)
            weighted_confidence = sum(
                s.confidence * s.model_quality 
                for s in cluster
            ) / total_weight if total_weight > 0 else 0
        else:
            weighted_confidence = sum(s.confidence for s in cluster) / len(cluster)
        
        # Merge expected improvement (take from highest confidence)
        best_suggestion = max(cluster, key=lambda x: x.confidence)
        
        # Build merged rationale
        if len(cluster) == 1:
            rationale = f"[{cluster[0].source_provider}] {cluster[0].rationale}"
        else:
            models_str = ", ".join(set(s.source_provider for s in cluster))
            rationale = (
                f"[Consensus: {len(cluster)} models ({models_str}) agree] "
                f"{best_suggestion.rationale}"
            )
        
        return ConsensusSuggestion(
            priority=representative.priority,
            config_changes=representative.config_changes,
            expected_improvement=best_suggestion.expected_improvement,
            rationale=rationale,
            supporting_models=supporting,
            agreement_score=agreement_score,
            weighted_confidence=weighted_confidence,
            source_suggestions=cluster
        )
    
    @staticmethod
    def _create_divergence_report(
        model_responses: List[ModelResponse],
        clusters: List[List[ParsedSuggestion]]
    ) -> DivergenceReport:
        """Analyze where models disagreed."""
        # Count bottleneck votes
        bottleneck_votes: Dict[str, List[str]] = {}
        for mr in model_responses:
            if mr.success and mr.raw_findings:
                bottleneck = mr.raw_findings.get("primary_bottleneck", "unknown")
                if bottleneck not in bottleneck_votes:
                    bottleneck_votes[bottleneck] = []
                bottleneck_votes[bottleneck].append(mr.full_model_name)
        
        # Find unique suggestions (in clusters of size 1)
        unique_suggestions = []
        for cluster in clusters:
            if len(cluster) == 1:
                s = cluster[0]
                desc = f"priority={s.priority}, changes={s.config_changes}"
                unique_suggestions.append((s.source_provider, desc))
        
        # Compute overall agreement
        all_models = len(model_responses)
        models_agreeing_on_bottleneck = max(
            (len(v) for v in bottleneck_votes.values()),
            default=0
        )
        overall_agreement = models_agreeing_on_bottleneck / all_models if all_models > 0 else 0
        
        return DivergenceReport(
            bottleneck_votes=bottleneck_votes,
            unique_suggestions=unique_suggestions,
            overall_agreement=overall_agreement
        )
