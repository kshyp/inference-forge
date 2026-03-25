"""Host/System Metrics Collectors - CPU, memory, IO stats.

These are "always available" profilers that don't require special hardware
or root privileges. They provide baseline system context.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseProfiler, ProfilingContext, ProfilerResult, RawProfilerOutput


class VMStatCollector(BaseProfiler):
    """
    Collect system-wide CPU, memory, and IO statistics using vmstat.
    
    Always available on Linux. Shows:
    - CPU: user, system, idle, wait IO percentages
    - Memory: free, buffered, cached
    - IO: blocks in/out
    - System: interrupts, context switches
    """
    
    PROFILER_NAME = "vmstat"
    DATA_TYPE = "vmstat_report"
    
    def is_available(self) -> bool:
        """Check if vmstat is available."""
        try:
            result = subprocess.run(
                ["vmstat", "--version"],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def run(self, context: ProfilingContext) -> ProfilerResult:
        """Run vmstat and collect metrics."""
        
        # Run vmstat with 1-second interval, 3 samples
        # Skip first sample (average since boot), use samples 2-3
        cmd = ["vmstat", "-n", "-a", "1", "3"]  # -a shows active/inactive memory
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return ProfilerResult(
                    success=False,
                    data_type=self.DATA_TYPE,
                    extractors={},
                    raw_output=RawProfilerOutput(
                        report_path=None,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        metadata={"profiler_name": self.PROFILER_NAME}
                    ),
                    error_message=f"vmstat failed: {result.stderr}"
                )
            
            # Parse vmstat output
            lines = result.stdout.strip().split("\n")
            
            # vmstat output format:
            # procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----
            #  r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
            #  1  0      0 123456  12345 234567    0    0    10    20  100  200  5  3 92  0  0
            
            samples = []
            header = None
            
            for line in lines:
                if line.startswith("procs") or line.startswith(" r "):
                    header = line.split()
                    continue
                
                if header and line.strip() and not line.startswith("procs"):
                    values = line.split()
                    if len(values) >= len(header):
                        sample = dict(zip(header, values))
                        samples.append(sample)
            
            # Use last sample (most recent)
            if samples:
                latest = samples[-1]
                
                metrics = {
                    # Process stats
                    "runnable_processes": int(latest.get("r", 0)),
                    "blocked_processes": int(latest.get("b", 0)),
                    
                    # Memory stats (in KB)
                    "memory_free_kb": int(latest.get("free", 0)),
                    "memory_buff_kb": int(latest.get("buff", 0)),
                    "memory_cache_kb": int(latest.get("cache", 0)),
                    "memory_active_kb": int(latest.get("active", 0)) if "active" in latest else 0,
                    "memory_inactive_kb": int(latest.get("inact", 0)) if "inact" in latest else 0,
                    
                    # Swap stats
                    "swap_in_kb_s": int(latest.get("si", 0)),
                    "swap_out_kb_s": int(latest.get("so", 0)),
                    
                    # IO stats (blocks/s)
                    "blocks_in_s": int(latest.get("bi", 0)),
                    "blocks_out_s": int(latest.get("bo", 0)),
                    
                    # System stats
                    "interrupts_s": int(latest.get("in", 0)),
                    "context_switches_s": int(latest.get("cs", 0)),
                    
                    # CPU percentages
                    "cpu_user_pct": int(latest.get("us", 0)),
                    "cpu_system_pct": int(latest.get("sy", 0)),
                    "cpu_idle_pct": int(latest.get("id", 0)),
                    "cpu_wait_pct": int(latest.get("wa", 0)),
                    "cpu_stolen_pct": int(latest.get("st", 0)),
                }
                
                # Save to file
                output_file = context.output_dir / "vmstat.json"
                with open(output_file, "w") as f:
                    json.dump({
                        "samples": samples,
                        "aggregated": metrics,
                        "collection_time": time.time()
                    }, f, indent=2)
                
                return ProfilerResult(
                    success=True,
                    data_type=self.DATA_TYPE,
                    extractors=metrics,
                    raw_output=RawProfilerOutput(
                        report_path=output_file,
                        stdout="",
                        stderr="",
                        metadata={"profiler_name": self.PROFILER_NAME}
                    )
                )
            
            return ProfilerResult(
                success=False,
                data_type=self.DATA_TYPE,
                extractors={},
                raw_output=RawProfilerOutput(
                    report_path=None,
                    stdout="",
                    stderr="No vmstat samples collected",
                    metadata={"profiler_name": self.PROFILER_NAME}
                ),
                error_message="No vmstat samples collected"
            )
            
        except Exception as e:
            return ProfilerResult(
                success=False,
                data_type=self.DATA_TYPE,
                extractors={},
                raw_output=RawProfilerOutput(
                    report_path=None,
                    stdout="",
                    stderr=str(e),
                    metadata={"profiler_name": self.PROFILER_NAME}
                ),
                error_message=str(e)
            )
    
    async def extract(self, raw_output: Path) -> Dict[str, Any]:
        """Extract metrics from saved file."""
        with open(raw_output) as f:
            data = json.load(f)
        return data.get("aggregated", {})


class MPStatCollector(BaseProfiler):
    """
    Collect per-CPU statistics using mpstat.
    
    Shows CPU usage breakdown per core:
    - usr, sys, iowait, irq, soft, steal, guest, idle
    """
    
    PROFILER_NAME = "mpstat"
    DATA_TYPE = "mpstat_report"
    
    def is_available(self) -> bool:
        """Check if mpstat is available."""
        try:
            result = subprocess.run(
                ["mpstat", "-V"],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def run(self, context: ProfilingContext) -> ProfilerResult:
        """Run mpstat and collect per-CPU metrics."""
        
        # Run mpstat once, get average since boot
        cmd = ["mpstat", "-P", "ALL", "1", "1"]  # All CPUs, 1-second, 1 sample
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return ProfilerResult(
                    success=False,
                    data_type=self.DATA_TYPE,
                    extractors={},
                    raw_output=RawProfilerOutput(
                        report_path=None,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        metadata={"profiler_name": self.PROFILER_NAME}
                    ),
                    error_message=f"mpstat failed: {result.stderr}"
                )
            
            # Parse mpstat output
            lines = result.stdout.strip().split("\n")
            
            # Header: 04:32:11 PM  CPU    %usr   %nice    %sys %iowait    %irq   %soft  %steal  %guest  %gnice   %idle
            per_cpu = []
            
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 12 and parts[1] != "CPU":  # Skip repeated headers
                    per_cpu.append({
                        "cpu": parts[1],
                        "usr_pct": float(parts[2]),
                        "nice_pct": float(parts[3]),
                        "sys_pct": float(parts[4]),
                        "iowait_pct": float(parts[5]),
                        "irq_pct": float(parts[6]),
                        "soft_pct": float(parts[7]),
                        "steal_pct": float(parts[8]),
                        "guest_pct": float(parts[9]),
                        "gnice_pct": float(parts[10]),
                        "idle_pct": float(parts[11]),
                    })
            
            # Aggregate across CPUs
            if per_cpu:
                # Separate all-core average from individual CPUs
                all_cpu = None
                individual_cpus = []
                
                for cpu_data in per_cpu:
                    if cpu_data["cpu"] == "all":
                        all_cpu = cpu_data
                    else:
                        individual_cpus.append(cpu_data)
                
                aggregated = {
                    "num_cpus": len(individual_cpus),
                    "all_cpu_avg": all_cpu,
                    "per_cpu": individual_cpus[:4],  # First 4 CPUs only (reduce size)
                    "avg_usr_pct": sum(c["usr_pct"] for c in individual_cpus) / len(individual_cpus),
                    "avg_sys_pct": sum(c["sys_pct"] for c in individual_cpus) / len(individual_cpus),
                    "avg_iowait_pct": sum(c["iowait_pct"] for c in individual_cpus) / len(individual_cpus),
                    "avg_idle_pct": sum(c["idle_pct"] for c in individual_cpus) / len(individual_cpus),
                }
                
                # Save to file
                output_file = context.output_dir / "mpstat.json"
                with open(output_file, "w") as f:
                    json.dump({
                        "per_cpu": per_cpu,
                        "aggregated": aggregated,
                        "collection_time": time.time()
                    }, f, indent=2)
                
                return ProfilerResult(
                    success=True,
                    data_type=self.DATA_TYPE,
                    extractors=aggregated,
                    raw_output=RawProfilerOutput(
                        report_path=output_file,
                        stdout="",
                        stderr="",
                        metadata={"profiler_name": self.PROFILER_NAME}
                    )
                )
            
            return ProfilerResult(
                success=False,
                data_type=self.DATA_TYPE,
                extractors={},
                raw_output=RawProfilerOutput(
                    report_path=None,
                    stdout="",
                    stderr="No mpstat data collected",
                    metadata={"profiler_name": self.PROFILER_NAME}
                ),
                error_message="No mpstat data collected"
            )
            
        except Exception as e:
            return ProfilerResult(
                success=False,
                data_type=self.DATA_TYPE,
                extractors={},
                raw_output=RawProfilerOutput(
                    report_path=None,
                    stdout="",
                    stderr=str(e),
                    metadata={"profiler_name": self.PROFILER_NAME}
                ),
                error_message=str(e)
            )
    
    async def extract(self, raw_output: Path) -> Dict[str, Any]:
        """Extract metrics from saved file."""
        with open(raw_output) as f:
            data = json.load(f)
        return data.get("aggregated", {})


class BaseMetricsCollector:
    """
    Convenience class to collect all "Tier 1" base metrics.
    
    These are metrics that are always collected because they're cheap
    and provide essential context for all SMEs.
    """
    
    @staticmethod
    def get_all_collectors() -> List[BaseProfiler]:
        """Get all base metric collectors."""
        return [
            SystemMetricsCollector(),  # nvidia-smi
            VMStatCollector(),         # vmstat
            MPStatCollector(),         # mpstat
        ]
    
    @staticmethod
    def get_data_types() -> List[str]:
        """Get data types for all base collectors."""
        return [
            "system_metrics_report",
            "vmstat_report",
            "mpstat_report",
        ]
