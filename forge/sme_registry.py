"""SME Registry - Discovers, registers, and coordinates Subject Matter Experts."""

import pkgutil
import importlib
from pathlib import Path
from typing import Dict, List, Type, Optional, Set, Any, Tuple
from dataclasses import dataclass, field

from forge.smes.base import BaseSME, DataRequirement, RegistrationInfo, SMEResponse


@dataclass
class RegisteredSME:
    """An SME instance that has successfully registered."""
    instance: BaseSME
    info: RegistrationInfo


class SMERegistry:
    """
    Registry for Subject Matter Expert agents.
    
    Responsibilities:
    1. Discover and load SME classes
    2. Register SMEs based on platform compatibility
    3. Aggregate data requirements for the profiler
    4. Consult ALL SMEs - each SME determines its own relevance by scanning data
    
    NOTE: The Coordinator NO LONGER decides which SME gets triggered.
    All profiling data is available to all SMEs in the profile directory.
    Each SME scans the data to determine if it reveals bottlenecks for their expertise.
    """
    
    def __init__(self):
        self.registered_smes: List[RegisteredSME] = []
        self.platform_info: Dict[str, Any] = {}
        self._sme_classes: List[Type[BaseSME]] = []
    
    def discover_sme_classes(self, module_path: str = "forge.smes") -> List[Type[BaseSME]]:
        """
        Auto-discover SME classes in the given module path.
        
        Looks for all classes that inherit from BaseSME (excluding BaseSME itself).
        
        Args:
            module_path: Python module path to search (e.g., "forge.smes")
            
        Returns:
            List of SME class types
        """
        sme_classes = []
        
        try:
            module = importlib.import_module(module_path)
            
            # Iterate through all modules in the package
            for importer, modname, ispkg in pkgutil.iter_modules(
                module.__path__ if hasattr(module, '__path__') else [],
                module.__name__ + "."
            ):
                try:
                    submodule = importlib.import_module(modname)
                    
                    # Find SME classes in this submodule
                    for name in dir(submodule):
                        obj = getattr(submodule, name)
                        if (
                            isinstance(obj, type) and
                            issubclass(obj, BaseSME) and
                            obj is not BaseSME and
                            not getattr(obj, '__abstractmethods__', None)
                        ):
                            sme_classes.append(obj)
                            
                except Exception as e:
                    print(f"[SMERegistry] Warning: Could not load {modname}: {e}")
                    
        except ImportError:
            print(f"[SMERegistry] Warning: Could not import {module_path}")
        
        self._sme_classes = sme_classes
        print(f"[SMERegistry] Discovered {len(sme_classes)} SME classes: "
              f"{[c.__name__ for c in sme_classes]}")
        return sme_classes
    
    def load_sme_classes(self, sme_classes: List[Type[BaseSME]]) -> None:
        """
        Manually provide SME classes (useful for testing or explicit control).
        
        Args:
            sme_classes: List of SME class types to register
        """
        self._sme_classes = sme_classes
        print(f"[SMERegistry] Loaded {len(sme_classes)} SME classes")
    
    def register_all(self, platform_info: Dict[str, Any]) -> Dict[str, RegistrationInfo]:
        """
        Attempt to register all discovered/loaded SMEs for this platform.
        
        Args:
            platform_info: Dict with platform details:
                - type: "nvidia_cuda", "amd_rocm", "cpu"
                - gpu_count: number of GPUs
                - gpu_model: GPU model name
                - cuda_version: CUDA version (if applicable)
                - driver_version: driver version
        
        Returns:
            Dict mapping sme_id to RegistrationInfo for successfully registered SMEs
        """
        self.platform_info = platform_info
        self.registered_smes = []
        registered = {}
        
        print(f"\n[SMERegistry] Registering SMEs for platform: "
              f"{platform_info.get('type', 'unknown')} "
              f"({platform_info.get('gpu_count', 0)} GPU(s))")
        print("-" * 60)
        
        for sme_class in self._sme_classes:
            instance = sme_class()
            
            try:
                reg_info = instance.register(platform_info)
                
                if reg_info:
                    self.registered_smes.append(RegisteredSME(instance, reg_info))
                    registered[reg_info.sme_id] = reg_info
                    print(f"✅ {sme_class.__name__:25} | ID: {reg_info.sme_id:20} | "
                          f"Data sources: {len(reg_info.data_requirements)}")
                else:
                    print(f"❌ {sme_class.__name__:25} | Not applicable for this platform")
                    
            except Exception as e:
                print(f"⚠️  {sme_class.__name__:25} | Registration failed: {e}")
        
        print("-" * 60)
        print(f"[SMERegistry] Registered {len(registered)}/{len(self._sme_classes)} SMEs\n")
        return registered
    
    def get_all_data_requirements(self) -> Dict[str, List[DataRequirement]]:
        """
        Get all data requirements grouped by data type.
        
        Returns a deduplicated mapping of data_type -> list of requirements.
        Multiple SMEs may need the same data type with different extractors.
        
        Returns:
            Dict mapping data_type to list of DataRequirements
        """
        requirements_by_type: Dict[str, List[DataRequirement]] = {}
        
        for registered in self.registered_smes:
            for req in registered.info.data_requirements:
                if req.data_type not in requirements_by_type:
                    requirements_by_type[req.data_type] = []
                requirements_by_type[req.data_type].append(req)
        
        return requirements_by_type
    
    def get_profiler_requirements(self) -> Dict[str, Dict[str, any]]:
        """
        Get consolidated requirements for the profiler agent.
        
        Returns a simplified structure telling the profiler exactly what to collect:
        
        Returns:
            Dict like:
            {
                "ncu_report": {"required": True, "extractors": [...]},
                "nsys_report": {"required": False, "extractors": [...]},
                ...
            }
        """
        consolidated: Dict[str, Dict[str, any]] = {}
        
        for registered in self.registered_smes:
            for req in registered.info.data_requirements:
                data_type = req.data_type
                
                if data_type not in consolidated:
                    consolidated[data_type] = {
                        "required": req.required,
                        "extractors": set(),
                        "requested_by": []
                    }
                else:
                    # If any SME requires it, mark as required
                    consolidated[data_type]["required"] = (
                        consolidated[data_type]["required"] or req.required
                    )
                
                # Merge extractors
                consolidated[data_type]["extractors"].update(req.extractors)
                consolidated[data_type]["requested_by"].append(registered.info.sme_id)
        
        # Convert sets to lists for serialization
        for data_type in consolidated:
            consolidated[data_type]["extractors"] = list(
                consolidated[data_type]["extractors"]
            )
        
        return consolidated
    
    def get_required_data_types(self) -> Set[str]:
        """
        Get set of all required data types (must be collected).
        
        Returns:
            Set of data_type strings that are required by at least one SME
        """
        required = set()
        for registered in self.registered_smes:
            for req in registered.info.data_requirements:
                if req.required:
                    required.add(req.data_type)
        return required
    
    async def consult(self, 
                      profile_dir: Path,
                      profiling_data: Dict[str, Any],
                      benchmark_metrics: Dict[str, Any]) -> Tuple[List[SMEResponse], List[str]]:
        """
        Consult ALL registered SMEs.
        
        Each SME scans the data to determine its own relevance.
        The Coordinator NO LONGER filters SMEs based on signals.
        
        This is async to enable parallel LLM calls across all consulted SMEs.
        
        Args:
            profile_dir: Path to directory containing profiling data files
            profiling_data: Data collected by the profiler
            benchmark_metrics: Results from benchmark run
            
        Returns:
            Tuple of (responses, sme_ids) where:
                - responses: List of SMEResponses from all SMEs (including non-relevant ones)
                - sme_ids: List of SME IDs corresponding to each response (parallel list)
        """
        import asyncio
        
        print(f"\n[SMERegistry] Consulting ALL {len(self.registered_smes)} SMEs")
        print(f"   Profile directory: {profile_dir}")
        print("-" * 60)
        
        # Create async tasks for ALL SMEs (no filtering)
        tasks = []
        sme_info = []
        
        for registered in self.registered_smes:
            sme_id = registered.info.sme_id
            
            # First, let SME scan data to determine relevance
            is_relevant, relevance_score, reason = registered.instance.scan_data(
                profile_dir, profiling_data, benchmark_metrics
            )
            
            if is_relevant:
                print(f"🔍 {sme_id}: RELEVANT (score={relevance_score:.2f}) - {reason}")
                # Create async task for this SME
                task = registered.instance.analyze(profile_dir, profiling_data, benchmark_metrics)
                tasks.append(task)
                sme_info.append(sme_id)
            else:
                print(f"   {sme_id}: NOT RELEVANT (score={relevance_score:.2f}) - {reason}")
        
        if not tasks:
            print("-" * 60)
            print(f"[SMERegistry] No SMEs found relevant to this data\n")
            return [], []
        
        print(f"\n[SMERegistry] Running analysis for {len(tasks)} relevant SME(s)...")
        
        # Run all SME analyses in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        responses = []
        consulted_sme_ids = []
        for sme_id, result in zip(sme_info, results):
            if isinstance(result, Exception):
                print(f"   ⚠️  {sme_id} analysis failed: {result}")
            else:
                responses.append(result)
                consulted_sme_ids.append(sme_id)
                status = "✓ relevant" if result.is_relevant else "✗ not relevant"
                print(f"   → {sme_id}: {len(result.suggestions)} suggestion(s), "
                      f"confidence: {result.confidence:.2f}, {status}")
        
        print("-" * 60)
        print(f"[SMERegistry] Completed {len(responses)}/{len(tasks)} analysis(es)\n")
        return responses, consulted_sme_ids
    
    def get_registered_sme_ids(self) -> List[str]:
        """Get list of registered SME IDs."""
        return [r.info.sme_id for r in self.registered_smes]
    
    def print_profiler_plan(self) -> None:
        """Print a summary of what the profiler needs to collect."""
        requirements = self.get_profiler_requirements()
        
        print("\n" + "=" * 60)
        print("PROFILER COLLECTION PLAN")
        print("=" * 60)
        
        for data_type, info in sorted(requirements.items()):
            required_mark = "✅ REQUIRED" if info["required"] else "☑️  Optional"
            print(f"\n{required_mark} | {data_type}")
            print(f"   Requested by: {', '.join(info['requested_by'])}")
            if info["extractors"]:
                print(f"   Extractors: {', '.join(info['extractors'][:5])}", end="")
                if len(info["extractors"]) > 5:
                    print(f" (+{len(info['extractors']) - 5} more)", end="")
                print()
        
        print("\n" + "=" * 60 + "\n")


# Convenience function for quick setup
def create_registry_with_platform(platform_info: Dict[str, Any]) -> SMERegistry:
    """
    Convenience function to create and populate a registry.
    
    Args:
        platform_info: Platform detection results
        
    Returns:
        Configured SMERegistry with registered SMEs
    """
    registry = SMERegistry()
    registry.discover_sme_classes()
    registry.register_all(platform_info)
    return registry
