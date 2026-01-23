#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
RUN ALL ABLATIONS - Comprehensive Ablation Study Runner
================================================================================

This script runs ALL ablation experiments sequentially and generates a 
comprehensive comparison report at the end.

Ablations included:
1. visual_tokens: [16, 32, 64, 128, 256, 512]
2. lora_r: [8, 16, 32, 64, 128]
3. lora_alpha: [16, 32, 64, 128, 256]
4. learning_rate: [1e-5, 5e-5, 1e-4, 2e-4, 5e-4]
5. num_frames: [2, 4, 8, 16]
6. projector_type: [mlp, linear, attention]
7. lora_dropout: [0.0, 0.05, 0.1, 0.15]

Usage:
------
python run_all_ablations.py \
    --train_data /path/to/train.json \
    --val_data /path/to/val.json \
    --test_data /path/to/test.json \
    --output_dir /path/to/ablation_results

# Run only specific ablations:
python run_all_ablations.py \
    --ablations visual_tokens lora_r \
    --train_data /path/to/train.json

# Quick test mode (reduced epochs and values):
python run_all_ablations.py --quick_test

Author: [Your Name]
Date: 2024
================================================================================
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# ABLATION CONFIGURATIONS
# ==============================================================================

# Full ablation configurations
FULL_ABLATIONS = {
    "visual_tokens": {
        "values": [16, 32, 64, 128, 256, 512, 1024],
        "type": "int",
        "description": "Number of visual tokens (compression level)",
        "priority": 1,  # Run order priority
    },
    "lora_r": {
        "values": [8, 16, 32, 64, 128],
        "type": "int",
        "description": "LoRA rank (model capacity)",
        "priority": 2,
    },
    "learning_rate": {
        "values": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
        "type": "float",
        "description": "Learning rate",
        "priority": 3,
    },
    "num_frames": {
        "values": [2, 4, 8],
        "type": "int",
        "description": "Number of video frames",
        "priority": 4,
    },
    "projector_type": {
        "values": ["mlp", "linear", "attention"],
        "type": "str",
        "description": "Visual projector architecture",
        "priority": 5,
    },
    "lora_alpha": {
        "values": [16, 32, 64, 128, 256],
        "type": "int",
        "description": "LoRA alpha (scaling factor)",
        "priority": 6,
    },
    "lora_dropout": {
        "values": [0.0, 0.05, 0.1, 0.15],
        "type": "float",
        "description": "LoRA dropout rate",
        "priority": 7,
    },
}

# Quick test configurations (reduced for faster testing)
QUICK_ABLATIONS = {
    "visual_tokens": {
        "values": [32, 128, 256],
        "type": "int",
        "description": "Number of visual tokens (compression level)",
        "priority": 1,
    },
    "lora_r": {
        "values": [16, 32, 64],
        "type": "int",
        "description": "LoRA rank (model capacity)",
        "priority": 2,
    },
    "learning_rate": {
        "values": [1e-4, 2e-4],
        "type": "float",
        "description": "Learning rate",
        "priority": 3,
    },
    "projector_type": {
        "values": ["mlp", "linear"],
        "type": "str",
        "description": "Visual projector architecture",
        "priority": 4,
    },
}


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_values_for_cli(values: List[Any], value_type: str) -> List[str]:
    """Format values for command line arguments."""
    if value_type == "float":
        return [f"{v:.0e}" if v < 0.001 else str(v) for v in values]
    else:
        return [str(v) for v in values]


def load_results(results_dir: Path) -> Optional[Dict]:
    """Load results from an ablation run."""
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def find_best_checkpoint_metrics(run_dir: Path) -> Optional[Dict]:
    """Find metrics from the best checkpoint in a run directory."""
    # Look for final test results
    for subdir in run_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith("run_"):
            test_results = subdir / "final_test_results.json"
            if test_results.exists():
                with open(test_results) as f:
                    return json.load(f)
    return None


# ==============================================================================
# ABLATION RUNNER
# ==============================================================================

class AblationRunner:
    """
    Runs multiple ablation studies and generates comparison reports.
    """
    
    def __init__(
        self,
        train_data: str,
        val_data: Optional[str],
        test_data: str,
        output_dir: str,
        vision_model: str,
        language_model: str,
        tasks: List[str],
        base_epochs: int = 20,
        base_batch_size: int = 2,
        base_grad_accum: int = 8,
        quick_test: bool = False,
        script_path: str = "train_surgical_vlm_v2.py",
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.output_dir = Path(output_dir)
        self.vision_model = vision_model
        self.language_model = language_model
        self.tasks = tasks
        self.base_epochs = base_epochs
        self.base_batch_size = base_batch_size
        self.base_grad_accum = base_grad_accum
        self.quick_test = quick_test
        self.script_path = script_path
        
        # Select ablation config
        self.ablations = QUICK_ABLATIONS if quick_test else FULL_ABLATIONS
        
        # Results storage
        self.all_results: Dict[str, Any] = {}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_single_ablation(self, param_name: str) -> Dict[str, Any]:
        """
        Run a single ablation study.
        
        Args:
            param_name: Name of parameter to ablate
        
        Returns:
            Results dictionary
        """
        config = self.ablations[param_name]
        values = config["values"]
        value_type = config["type"]
        
        logger.info("=" * 70)
        logger.info(f"ABLATION: {param_name}")
        logger.info(f"Values: {values}")
        logger.info(f"Description: {config['description']}")
        logger.info("=" * 70)
        
        # Build command
        cmd = [
            sys.executable,
            self.script_path,
            "--train_data", self.train_data,
            "--test_data", self.test_data,
            "--output_dir", str(self.output_dir / f"ablation_{param_name}"),
            "--vision_model", self.vision_model,
            "--language_model", self.language_model,
            "--tasks", *self.tasks,
            "--epochs", str(1 if self.quick_test else self.base_epochs),
            "--batch_size", str(self.base_batch_size),
            "--grad_accum", str(self.base_grad_accum),
            "--run_ablation",
            "--ablation_param", param_name,
            "--ablation_values", *format_values_for_cli(values, value_type),
            "--early_stopping",
        ]
        
        # Add validation data if provided
        if self.val_data:
            cmd.extend(["--val_data", self.val_data])
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Run ablation
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=False,  # Show output in real-time
                text=True,
                check=True,
            )
            success = True
            error = None
        except subprocess.CalledProcessError as e:
            success = False
            error = str(e)
            logger.error(f"Ablation {param_name} failed: {e}")
        
        elapsed = time.time() - start_time
        
        # Load results
        results_dir = self.output_dir / f"ablation_{param_name}"
        results = load_results(results_dir)
        
        return {
            "param_name": param_name,
            "values": values,
            "description": config["description"],
            "success": success,
            "error": error,
            "elapsed_time": elapsed,
            "elapsed_formatted": format_time(elapsed),
            "results": results,
        }
    
    def run_all_ablations(self, ablation_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run all specified ablation studies.
        
        Args:
            ablation_names: List of ablation names to run (None = all)
        
        Returns:
            Combined results dictionary
        """
        # Determine which ablations to run
        if ablation_names:
            ablations_to_run = {k: v for k, v in self.ablations.items() if k in ablation_names}
        else:
            ablations_to_run = self.ablations
        
        # Sort by priority
        sorted_ablations = sorted(
            ablations_to_run.items(),
            key=lambda x: x[1].get("priority", 999)
        )
        
        total_ablations = len(sorted_ablations)
        total_runs = sum(len(v["values"]) for _, v in sorted_ablations)
        
        logger.info("=" * 70)
        logger.info("COMPREHENSIVE ABLATION STUDY")
        logger.info("=" * 70)
        logger.info(f"Total ablation parameters: {total_ablations}")
        logger.info(f"Total individual runs: {total_runs}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 70)
        
        # Run each ablation
        all_start_time = time.time()
        
        for idx, (param_name, _) in enumerate(sorted_ablations, 1):
            logger.info(f"\n[{idx}/{total_ablations}] Starting ablation: {param_name}")
            
            result = self.run_single_ablation(param_name)
            self.all_results[param_name] = result
            
            # Save intermediate results
            self._save_combined_results()
        
        total_elapsed = time.time() - all_start_time
        
        # Generate final report
        self._generate_report(total_elapsed)
        
        return self.all_results
    
    def _save_combined_results(self) -> None:
        """Save combined results to JSON."""
        results_path = self.output_dir / "all_ablations_results.json"
        
        # Make results JSON serializable
        serializable_results = {}
        for param_name, result in self.all_results.items():
            serializable_results[param_name] = {
                k: v for k, v in result.items()
                if k != "results" or v is None  # Handle nested results separately
            }
            if result.get("results"):
                serializable_results[param_name]["results"] = result["results"]
        
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
    
    def _generate_report(self, total_elapsed: float) -> None:
        """Generate comprehensive comparison report."""
        report_path = self.output_dir / "ablation_report.md"
        
        with open(report_path, "w") as f:
            f.write("# Comprehensive Ablation Study Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total runtime: {format_time(total_elapsed)}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- Train data: `{self.train_data}`\n")
            f.write(f"- Validation data: `{self.val_data}`\n")
            f.write(f"- Test data: `{self.test_data}`\n")
            f.write(f"- Tasks: {self.tasks}\n")
            f.write(f"- Quick test mode: {self.quick_test}\n\n")
            
            f.write("## Summary Table\n\n")
            f.write("| Parameter | Best Value | Best Accuracy | Values Tested |\n")
            f.write("|-----------|------------|---------------|---------------|\n")
            
            best_configs = {}
            
            for param_name, result in self.all_results.items():
                if not result["success"] or not result.get("results"):
                    f.write(f"| {param_name} | FAILED | - | {result['values']} |\n")
                    continue
                
                # Find best value
                runs = result["results"].get("runs", [])
                best_run = None
                best_acc = -1
                
                for run in runs:
                    if run.get("success") and run.get("metrics"):
                        acc = run["metrics"].get("accuracy", 0)
                        if acc > best_acc:
                            best_acc = acc
                            best_run = run
                
                if best_run:
                    best_value = best_run["value"]
                    best_configs[param_name] = best_value
                    f.write(f"| {param_name} | {best_value} | {best_acc:.4f} | {result['values']} |\n")
                else:
                    f.write(f"| {param_name} | N/A | N/A | {result['values']} |\n")
            
            f.write("\n## Detailed Results\n\n")
            
            for param_name, result in self.all_results.items():
                f.write(f"### {param_name}\n\n")
                f.write(f"**Description:** {result['description']}\n\n")
                f.write(f"**Runtime:** {result['elapsed_formatted']}\n\n")
                
                if not result["success"]:
                    f.write(f"**Status:** ❌ FAILED\n\n")
                    f.write(f"**Error:** {result.get('error', 'Unknown')}\n\n")
                    continue
                
                f.write("| Value | Accuracy | Status |\n")
                f.write("|-------|----------|--------|\n")
                
                runs = result.get("results", {}).get("runs", [])
                for run in runs:
                    value = run["value"]
                    if run.get("success") and run.get("metrics"):
                        acc = run["metrics"].get("accuracy", 0)
                        f.write(f"| {value} | {acc:.4f} | ✅ |\n")
                    else:
                        f.write(f"| {value} | - | ❌ |\n")
                
                f.write("\n")
            
            f.write("## Recommended Configuration\n\n")
            f.write("Based on the ablation results, the recommended configuration is:\n\n")
            f.write("```python\n")
            f.write("config = TrainConfig(\n")
            for param, value in best_configs.items():
                if isinstance(value, str):
                    f.write(f'    {param}="{value}",\n')
                else:
                    f.write(f"    {param}={value},\n")
            f.write(")\n")
            f.write("```\n")
        
        logger.info(f"Report saved to: {report_path}")
        
        # Also print summary to console
        logger.info("\n" + "=" * 70)
        logger.info("ABLATION STUDY COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total runtime: {format_time(total_elapsed)}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"Report: {report_path}")
        logger.info("\nBest configurations found:")
        for param, value in best_configs.items():
            logger.info(f"  {param}: {value}")
        logger.info("=" * 70)


# ==============================================================================
# BASH SCRIPT GENERATOR
# ==============================================================================

def generate_bash_script(
    output_path: str,
    train_data: str,
    val_data: Optional[str],
    test_data: str,
    output_dir: str,
    ablations: Optional[List[str]] = None,
) -> None:
    """
    Generate a bash script to run all ablations.
    
    This is useful for running on clusters or when you want more control.
    
    Args:
        output_path: Path to save bash script
        train_data: Path to training data
        val_data: Path to validation data
        test_data: Path to test data
        output_dir: Output directory
        ablations: List of ablations to include (None = all)
    """
    ablations_config = FULL_ABLATIONS if ablations is None else {
        k: v for k, v in FULL_ABLATIONS.items() if k in ablations
    }
    
    with open(output_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#" + "=" * 69 + "\n")
        f.write("# Comprehensive Ablation Study - Auto-generated Script\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#" + "=" * 69 + "\n\n")
        
        f.write("# Exit on error\n")
        f.write("set -e\n\n")
        
        f.write("# Configuration\n")
        f.write(f'TRAIN_DATA="{train_data}"\n')
        if val_data:
            f.write(f'VAL_DATA="{val_data}"\n')
        f.write(f'TEST_DATA="{test_data}"\n')
        f.write(f'OUTPUT_DIR="{output_dir}"\n')
        f.write('SCRIPT="train_surgical_vlm_v2.py"\n\n')
        
        f.write("# Create output directory\n")
        f.write('mkdir -p "$OUTPUT_DIR"\n\n')
        
        f.write("# Log file\n")
        f.write('LOGFILE="$OUTPUT_DIR/ablation_$(date +%Y%m%d_%H%M%S).log"\n')
        f.write('exec 1> >(tee -a "$LOGFILE") 2>&1\n\n')
        
        f.write('echo "Starting ablation study at $(date)"\n')
        f.write('echo "=" | head -c 70 && echo ""\n\n')
        
        for param_name, config in ablations_config.items():
            values = config["values"]
            value_type = config["type"]
            
            f.write(f"# Ablation: {param_name}\n")
            f.write(f'echo ""\n')
            f.write(f'echo "Running ablation: {param_name}"\n')
            f.write(f'echo "Values: {values}"\n')
            f.write(f'echo "=" | head -c 70 && echo ""\n\n')
            
            formatted_values = format_values_for_cli(values, value_type)
            
            cmd_parts = [
                'python "$SCRIPT"',
                '--train_data "$TRAIN_DATA"',
                '--test_data "$TEST_DATA"',
                f'--output_dir "$OUTPUT_DIR/ablation_{param_name}"',
                '--run_ablation',
                f'--ablation_param {param_name}',
                f'--ablation_values {" ".join(formatted_values)}',
                '--early_stopping',
            ]
            
            if val_data:
                cmd_parts.insert(2, '--val_data "$VAL_DATA"')
            
            f.write(" \\\n    ".join(cmd_parts) + "\n\n")
        
        f.write('echo ""\n')
        f.write('echo "=" | head -c 70 && echo ""\n')
        f.write('echo "Ablation study complete at $(date)"\n')
        f.write('echo "Results saved to: $OUTPUT_DIR"\n')
    
    # Make executable
    os.chmod(output_path, 0o755)
    logger.info(f"Bash script saved to: {output_path}")


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
# Run all ablations:
python run_all_ablations.py \\
    --train_data train.json \\
    --test_data test.json

# Run specific ablations:
python run_all_ablations.py \\
    --ablations visual_tokens lora_r learning_rate

# Quick test mode (fewer values, 1 epoch):
python run_all_ablations.py --quick_test

# Generate bash script instead of running:
python run_all_ablations.py --generate_bash ablations.sh
        """
    )
    
    # Data paths
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data JSON")
    parser.add_argument("--val_data", type=str, default=None,
                       help="Path to validation data JSON")
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to test data JSON")
    parser.add_argument("--output_dir", type=str, default="./ablation_results",
                       help="Output directory")
    
    # Model paths
    parser.add_argument("--vision_model", type=str,
                       default="/mnt/share/ali/VLMs/hf_cache/hub/siglip2-model/",
                       help="Path to vision model")
    parser.add_argument("--language_model", type=str,
                       default="/mnt/share/ali/VLMs/hf_cache/hub/Qwen3-0.6B/",
                       help="Path to language model")
    
    # Tasks
    parser.add_argument("--tasks", nargs="+",
                       default=["step_classification", "stage_classification"],
                       help="Tasks to train on")
    
    # Ablation selection
    parser.add_argument("--ablations", nargs="+", default=None,
                       choices=list(FULL_ABLATIONS.keys()),
                       help="Specific ablations to run (default: all)")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=20,
                       help="Base number of epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=8,
                       help="Gradient accumulation steps")
    
    # Mode selection
    parser.add_argument("--quick_test", action="store_true",
                       help="Quick test mode (reduced values and epochs)")
    parser.add_argument("--generate_bash", type=str, default=None,
                       help="Generate bash script instead of running")
    
    # Script path
    parser.add_argument("--script", type=str, default="train_surgical_vlm_v2.py",
                       help="Path to training script")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Generate bash script if requested
    if args.generate_bash:
        generate_bash_script(
            output_path=args.generate_bash,
            train_data=args.train_data,
            val_data=args.val_data,
            test_data=args.test_data,
            output_dir=args.output_dir,
            ablations=args.ablations,
        )
        return
    
    # Create runner
    runner = AblationRunner(
        train_data=args.train_data,
        val_data=args.val_data,
        test_data=args.test_data,
        output_dir=args.output_dir,
        vision_model=args.vision_model,
        language_model=args.language_model,
        tasks=args.tasks,
        base_epochs=args.epochs,
        base_batch_size=args.batch_size,
        base_grad_accum=args.grad_accum,
        quick_test=args.quick_test,
        script_path=args.script,
    )
    
    # Run ablations
    runner.run_all_ablations(args.ablations)


if __name__ == "__main__":
    main()