"""HuggingFace quantized model discovery utility.

Searches HuggingFace Hub for pre-quantized models matching recommendations.
"""

import re
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class QuantizedModelInfo:
    """Information about a quantized model on HuggingFace."""
    model_id: str
    base_model: str
    quantization_method: str  # "awq", "gptq", "fp8", "int8", etc.
    bits: int
    author: str
    downloads: int
    likes: int
    tags: List[str]
    
    def __str__(self) -> str:
        return f"{self.model_id} ({self.quantization_method}-{self.bits}bit)"


class HFQuantizedModelFinder:
    """Find pre-quantized models on HuggingFace Hub."""
    
    # Common naming patterns for quantized models
    QUANT_PATTERNS = {
        "awq": [r"-awq-?", r"-AWQ-?", r"_awq_?", r"_AWQ_?"],
        "gptq": [r"-gptq-?", r"-GPTQ-?", r"_gptq_?", r"_GPTQ_?"],
        "fp8": [r"-fp8-?", r"-FP8-?", r"_fp8_?", r"_FP8_?"],
        "int8": [r"-int8-?", r"-INT8-?", r"_int8_?", r"_int8_?"],
        "int4": [r"-int4-?", r"-INT4-?", r"-4bit-?", r"_4bit_?"],
    }
    
    # Bits extraction patterns
    BITS_PATTERNS = [
        (r"-(\d+)bit", lambda m: int(m.group(1))),
        (r"-int(\d+)-", lambda m: int(m.group(1))),
        (r"-(\d+)-bit", lambda m: int(m.group(1))),
    ]
    
    def __init__(self):
        self.api = HfApi() if HF_AVAILABLE else None
        
    def is_available(self) -> bool:
        """Check if HuggingFace Hub is available."""
        return HF_AVAILABLE and self.api is not None
    
    async def find_model(
        self,
        base_model: str,
        quantization: str,
        bits: int = 4,
        prefer_author: Optional[str] = None,
    ) -> Optional[str]:
        """Find a quantized model matching the recommendation.
        
        Args:
            base_model: Base model ID (e.g., "meta-llama/Llama-2-7b-hf")
            quantization: Quantization method ("awq", "gptq", "fp8", "int8")
            bits: Bit width (4, 8)
            prefer_author: Prefer models from this author (e.g., "TheBloke")
            
        Returns:
            Model ID if found, None otherwise
        """
        if not self.is_available():
            logger.warning("HuggingFace Hub not available, cannot search for quantized models")
            return None
        
        # Extract base model name without org
        base_name = base_model.split("/")[-1] if "/" in base_model else base_model
        
        # Search strategy 1: Direct search by tags
        models = self._search_by_tags(quantization, base_name)
        
        # Search strategy 2: Search by model name patterns
        if not models:
            models = self._search_by_name(base_name, quantization)
        
        if not models:
            logger.info(f"No quantized models found for {base_model} with {quantization}")
            return None
        
        # Filter and rank models
        candidates = self._rank_models(models, base_model, quantization, bits, prefer_author)
        
        if candidates:
            best = candidates[0]
            logger.info(f"Found quantized model: {best.model_id}")
            return best.model_id
        
        return None
    
    def _search_by_tags(self, quantization: str, base_name: str) -> List[Dict[str, Any]]:
        """Search for models using HuggingFace tags."""
        try:
            # Map quantization methods to HF tags
            tag_map = {
                "awq": "awq",
                "gptq": "gptq",
                "fp8": "fp8",
                "int8": "int8",
                "int4": "int4",
            }
            
            tag = tag_map.get(quantization.lower())
            if not tag:
                return []
            
            # Search for models with the quantization tag
            models = list(self.api.list_models(
                filter=tag,
                search=base_name.replace("-", " "),
                limit=20,
            ))
            
            return [self._model_to_dict(m) for m in models]
            
        except Exception as e:
            logger.debug(f"Tag search failed: {e}")
            return []
    
    def _search_by_name(self, base_name: str, quantization: str) -> List[Dict[str, Any]]:
        """Search for models by name patterns."""
        try:
            # Common quantized model naming patterns
            search_terms = [
                f"{base_name}-{quantization}",
                f"{base_name}_{quantization}",
                f"{quantization}-{base_name}",
            ]
            
            all_models = []
            for term in search_terms:
                models = list(self.api.list_models(
                    search=term,
                    limit=10,
                ))
                all_models.extend([self._model_to_dict(m) for m in models])
            
            return all_models
            
        except Exception as e:
            logger.debug(f"Name search failed: {e}")
            return []
    
    def _model_to_dict(self, model) -> Dict[str, Any]:
        """Convert HF model object to dict."""
        return {
            "model_id": model.modelId,
            "author": getattr(model, "author", None),
            "downloads": getattr(model, "downloads", 0),
            "likes": getattr(model, "likes", 0),
            "tags": list(getattr(model, "tags", [])),
        }
    
    def _rank_models(
        self,
        models: List[Dict[str, Any]],
        base_model: str,
        quantization: str,
        bits: int,
        prefer_author: Optional[str],
    ) -> List[QuantizedModelInfo]:
        """Rank models by relevance to the request."""
        candidates = []
        base_name = base_model.split("/")[-1]
        
        for model in models:
            model_id = model["model_id"]
            
            # Check if model matches quantization method
            detected_method = self._detect_quantization_method(model_id, model.get("tags", []))
            if detected_method != quantization.lower():
                continue
            
            # Check if model is for the right base model
            if not self._matches_base_model(model_id, base_name):
                continue
            
            # Extract bits
            detected_bits = self._detect_bits(model_id, model.get("tags", []))
            if detected_bits != bits:
                continue
            
            info = QuantizedModelInfo(
                model_id=model_id,
                base_model=base_model,
                quantization_method=detected_method,
                bits=detected_bits or bits,
                author=model.get("author", ""),
                downloads=model.get("downloads", 0),
                likes=model.get("likes", 0),
                tags=model.get("tags", []),
            )
            candidates.append(info)
        
        # Sort by relevance score
        candidates.sort(key=lambda m: self._relevance_score(m, prefer_author), reverse=True)
        
        return candidates
    
    def _detect_quantization_method(self, model_id: str, tags: List[str]) -> Optional[str]:
        """Detect quantization method from model name and tags."""
        model_id_lower = model_id.lower()
        
        # Check tags first (most reliable)
        tag_mapping = {
            "awq": "awq",
            "gptq": "gptq",
            "fp8": "fp8",
            "int8": "int8",
            "int4": "int4",
            "4bit": "int4",
            "8bit": "int8",
        }
        
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in tag_mapping:
                return tag_mapping[tag_lower]
        
        # Check model name patterns
        for method, patterns in self.QUANT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, model_id_lower):
                    return method
        
        return None
    
    def _detect_bits(self, model_id: str, tags: List[str]) -> Optional[int]:
        """Detect bit width from model name and tags."""
        model_id_lower = model_id.lower()
        
        # Check tags
        for tag in tags:
            if "4bit" in tag.lower() or "int4" in tag.lower():
                return 4
            if "8bit" in tag.lower() or "int8" in tag.lower():
                return 8
            if "fp8" in tag.lower():
                return 8
        
        # Check model name patterns
        for pattern, extractor in self.BITS_PATTERNS:
            match = re.search(pattern, model_id_lower)
            if match:
                return extractor(match)
        
        # Default inference from quantization method
        if "awq" in model_id_lower or "gptq" in model_id_lower:
            return 4  # Most AWQ/GPTQ models are 4-bit
        if "fp8" in model_id_lower or "int8" in model_id_lower:
            return 8
        
        return None
    
    def _matches_base_model(self, model_id: str, base_name: str) -> bool:
        """Check if quantized model matches the base model."""
        model_id_lower = model_id.lower()
        base_lower = base_name.lower()
        
        # Direct substring match
        if base_lower in model_id_lower:
            return True
        
        # Handle common variations
        variations = [
            base_lower,
            base_lower.replace("-", "_"),
            base_lower.replace("_", "-"),
            base_lower.replace("-", ""),
        ]
        
        for var in variations:
            if var in model_id_lower:
                return True
        
        return False
    
    def _relevance_score(self, model: QuantizedModelInfo, prefer_author: Optional[str]) -> float:
        """Calculate relevance score for ranking."""
        score = 0.0
        
        # Prefer popular models (downloads)
        score += min(model.downloads / 10000, 10.0)  # Cap at 10
        
        # Prefer liked models
        score += min(model.likes / 100, 5.0)  # Cap at 5
        
        # Prefer specific authors (e.g., "TheBloke" for GPTQ)
        author = model.author or ""
        if prefer_author and author.lower() == prefer_author.lower():
            score += 20.0
        
        # Prefer official or well-known quantizers
        trusted_authors = ["thebloke", "hugging-quants", "neuralmagic", "octoai"]
        if author.lower() in trusted_authors:
            score += 10.0
        
        return score


# Convenience function for direct usage
async def find_quantized_model(
    base_model: str,
    quantization: str,
    bits: int = 4,
    prefer_author: Optional[str] = None,
) -> Optional[str]:
    """Find a quantized model on HuggingFace Hub.
    
    Args:
        base_model: Base model ID (e.g., "meta-llama/Llama-2-7b-hf")
        quantization: Quantization method ("awq", "gptq", "fp8", "int8")
        bits: Bit width (4, 8)
        prefer_author: Prefer models from this author
        
    Returns:
        Model ID if found, None otherwise
        
    Example:
        >>> model_id = await find_quantized_model(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     "awq",
        ...     bits=4,
        ...     prefer_author="TheBloke"
        ... )
        >>> print(model_id)
        "TheBloke/Llama-2-7b-AWQ"
    """
    finder = HFQuantizedModelFinder()
    return await finder.find_model(base_model, quantization, bits, prefer_author)


# Preferred authors by quantization method
PREFERRED_AUTHORS = {
    "gptq": "TheBloke",  # TheBloke is the most prolific GPTQ quantizer
    "awq": "TheBloke",   # TheBloke also does AWQ
    "fp8": None,         # FP8 is newer, no dominant quantizer yet
    "int8": None,
}


async def find_best_quantized_model(
    base_model: str,
    recommendation: str,  # From QuantizationSME
    bits: int = 4,
) -> Optional[str]:
    """Find the best quantized model with automatic author preference.
    
    Args:
        base_model: Base model ID
        recommendation: Quantization method recommended by SME
        bits: Bit width
        
    Returns:
        Best matching model ID
    """
    prefer_author = PREFERRED_AUTHORS.get(recommendation.lower())
    return await find_quantized_model(base_model, recommendation, bits, prefer_author)
