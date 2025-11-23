#!/usr/bin/env python3
"""
Patch script to add rollout timing and token length metrics to validation.
This script modifies verl's validation function to record:
- Total rollout time per batch
- Total response token length per batch
- Per-worker timing: fastest worker (min), slowest worker (max), averages
- All metrics saved to JSON file for detailed analysis

Note: Since verl aggregates timing across workers, we extract min/max from
the meta_info timing dict to approximate fastest/slowest worker times.
"""

import os
import sys
from pathlib import Path

# Path to verl trainer file
VERL_TRAINER_PATH = Path(__file__).parent / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py"

def patch_validation_metrics():
    """Patch the _validate method to add rollout timing and token metrics."""
    
    if not VERL_TRAINER_PATH.exists():
        print(f"Error: {VERL_TRAINER_PATH} does not exist!")
        return False
    
    with open(VERL_TRAINER_PATH, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'total_rollout_time' in content and 'total_response_tokens' in content:
        print("Already patched!")
        return True
    
    # Find the location to insert metrics collection
    # We want to add metrics after generate_sequences but before processing validation results
    
    # Find where we collect rollout timing info
    insert_marker = "print(\"validation generation end\")"
    
    if insert_marker not in content:
        print(f"Error: Cannot find insertion point in {VERL_TRAINER_PATH}")
        return False
    
    # Add code to collect metrics
    patch_code = '''
            print("validation generation end")
            
            # Collect rollout timing and token metrics
            rollout_timing_info = test_output_gen_batch.meta_info.get("timing", {})
            rollout_time = rollout_timing_info.get("generate_sequences", 0.0)
            
            # Extract per-worker timing info (min/max from timing dict)
            # These represent fastest and slowest worker times
            timing_min = rollout_timing_info.get("generation_timing/min", rollout_time)
            timing_max = rollout_timing_info.get("generation_timing/max", rollout_time)
            timing_avg = rollout_time
            
            # Calculate total response tokens
            from verl.trainer.ppo.metric_utils import _compute_response_info
            response_info = _compute_response_info(test_output_gen_batch)
            total_response_tokens = response_info["response_length"].sum().item()
            total_prompt_tokens = response_info["prompt_length"].sum().item()
            
            # Store metrics for later aggregation
            if not hasattr(self, '_validation_rollout_metrics'):
                self._validation_rollout_metrics = {
                    'total_rollout_time': 0.0,
                    'total_response_tokens': 0,
                    'total_prompt_tokens': 0,
                    'num_batches': 0,
                    'worker_times': [],  # List of (batch_idx, avg_time, min_time, max_time)
                    'worker_min_times': [],  # Fastest worker times per batch
                    'worker_max_times': [],  # Slowest worker times per batch
                }
            
            batch_idx = self._validation_rollout_metrics['num_batches']
            self._validation_rollout_metrics['total_rollout_time'] += rollout_time
            self._validation_rollout_metrics['total_response_tokens'] += total_response_tokens
            self._validation_rollout_metrics['total_prompt_tokens'] += total_prompt_tokens
            self._validation_rollout_metrics['num_batches'] += 1
            
            # Store per-batch worker timing info
            self._validation_rollout_metrics['worker_times'].append({
                'batch': batch_idx,
                'avg_time': timing_avg,
                'min_time': timing_min,
                'max_time': timing_max,
                'response_tokens': total_response_tokens,
            })
            self._validation_rollout_metrics['worker_min_times'].append(timing_min)
            self._validation_rollout_metrics['worker_max_times'].append(timing_max)
            
            print(f"Rollout batch {batch_idx + 1}: "
                  f"avg_time={timing_avg:.2f}s, "
                  f"min_time={timing_min:.2f}s (fastest worker), "
                  f"max_time={timing_max:.2f}s (slowest worker), "
                  f"response_tokens={total_response_tokens}")
'''
    
    # Replace the marker with our patched code
    new_content = content.replace(insert_marker, patch_code)
    
    # Find where to add final metrics to metric_dict (at the end of _validate)
    # Look for the return statement
    return_marker = "        return metric_dict"
    
    if return_marker not in new_content:
        print("Warning: Cannot find return statement, metrics may not be added to final output")
    else:
        # Add final aggregated metrics before return
        final_metrics_code = '''
        # Add rollout timing and token metrics
        if hasattr(self, '_validation_rollout_metrics') and self._validation_rollout_metrics['num_batches'] > 0:
            metrics = self._validation_rollout_metrics
            
            # Calculate per-worker statistics
            if metrics['worker_min_times']:
                fastest_worker_time = min(metrics['worker_min_times'])
                slowest_worker_time = max(metrics['worker_max_times'])
                avg_fastest_time = sum(metrics['worker_min_times']) / len(metrics['worker_min_times'])
                avg_slowest_time = sum(metrics['worker_max_times']) / len(metrics['worker_max_times'])
            else:
                fastest_worker_time = slowest_worker_time = avg_fastest_time = avg_slowest_time = 0.0
            
            metric_dict.update({
                'rollout/total_time_s': metrics['total_rollout_time'],
                'rollout/total_response_tokens': metrics['total_response_tokens'],
                'rollout/total_prompt_tokens': metrics['total_prompt_tokens'],
                'rollout/num_batches': metrics['num_batches'],
                'rollout/avg_time_per_batch_s': metrics['total_rollout_time'] / metrics['num_batches'],
                'rollout/avg_response_tokens_per_batch': metrics['total_response_tokens'] / metrics['num_batches'],
                'rollout/throughput_tokens_per_sec': metrics['total_response_tokens'] / (metrics['total_rollout_time'] + 1e-9),
                'rollout/fastest_worker_time_s': fastest_worker_time,
                'rollout/slowest_worker_time_s': slowest_worker_time,
                'rollout/avg_fastest_worker_time_s': avg_fastest_time,
                'rollout/avg_slowest_worker_time_s': avg_slowest_time,
            })
            
            # Save detailed per-worker metrics to file
            import json
            import os
            from datetime import datetime
            # Try multiple possible log directories
            log_dir = None
            for possible_dir in [
                self.config.trainer.get("default_local_dir", None),
                os.environ.get("VERL_LOG_DIR", None),
                os.environ.get("LOG_DIR", None),
                os.path.join(os.getcwd(), "logs"),
                os.getcwd(),
            ]:
                if possible_dir:
                    log_dir = possible_dir
                    if not os.path.isabs(log_dir):
                        log_dir = os.path.join(os.getcwd(), log_dir)
                    os.makedirs(log_dir, exist_ok=True)
                    break
            
            if not log_dir:
                log_dir = os.getcwd()
                os.makedirs(log_dir, exist_ok=True)
            
            worker_metrics_file = os.path.join(log_dir, f"rollout_worker_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(worker_metrics_file, 'w') as f:
                json.dump({
                    'summary': {
                        'total_rollout_time_s': metrics['total_rollout_time'],
                        'total_response_tokens': metrics['total_response_tokens'],
                        'total_prompt_tokens': metrics['total_prompt_tokens'],
                        'num_batches': metrics['num_batches'],
                        'fastest_worker_time_s': fastest_worker_time,
                        'slowest_worker_time_s': slowest_worker_time,
                        'avg_fastest_worker_time_s': avg_fastest_time,
                        'avg_slowest_worker_time_s': avg_slowest_time,
                        'time_difference_s': slowest_worker_time - fastest_worker_time,
                    },
                    'per_batch_metrics': metrics['worker_times'],
                }, f, indent=2)
            print(f"\\nSaved per-worker metrics to: {worker_metrics_file}")
            
            print(f"\\n=== Rollout Metrics Summary ===")
            print(f"Total rollout time: {metrics['total_rollout_time']:.2f}s")
            print(f"Total response tokens: {metrics['total_response_tokens']}")
            print(f"Total prompt tokens: {metrics['total_prompt_tokens']}")
            print(f"Number of batches: {metrics['num_batches']}")
            print(f"Throughput: {metrics['total_response_tokens'] / (metrics['total_rollout_time'] + 1e-9):.2f} tokens/sec")
            print(f"\\n--- Per-Worker Timing ---")
            print(f"Fastest worker (min): {fastest_worker_time:.2f}s")
            print(f"Slowest worker (max): {slowest_worker_time:.2f}s")
            print(f"Avg fastest worker: {avg_fastest_time:.2f}s")
            print(f"Avg slowest worker: {avg_slowest_time:.2f}s")
            print(f"Time difference (max - min): {slowest_worker_time - fastest_worker_time:.2f}s")
            print(f"===============================\\n")
            # Reset for next validation
            delattr(self, '_validation_rollout_metrics')
        
        return metric_dict'''
        
        new_content = new_content.replace(return_marker, final_metrics_code)
    
    # Write back
    backup_path = VERL_TRAINER_PATH.with_suffix('.py.backup')
    if not backup_path.exists():
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"Created backup: {backup_path}")
    
    with open(VERL_TRAINER_PATH, 'w') as f:
        f.write(new_content)
    
    print(f"Successfully patched {VERL_TRAINER_PATH}")
    print("The validation will now record:")
    print("  - rollout/total_time_s: Total rollout time in seconds")
    print("  - rollout/total_response_tokens: Total generated response tokens")
    print("  - rollout/total_prompt_tokens: Total prompt tokens")
    print("  - rollout/throughput_tokens_per_sec: Tokens per second")
    print("  - rollout/fastest_worker_time_s: Fastest worker time (min)")
    print("  - rollout/slowest_worker_time_s: Slowest worker time (max)")
    print("  - rollout/avg_fastest_worker_time_s: Average fastest worker time")
    print("  - rollout/avg_slowest_worker_time_s: Average slowest worker time")
    print("")
    print("Per-worker metrics will be saved to JSON file in checkpoint directory")
    print("  (same directory as trainer.default_local_dir)")
    return True

if __name__ == "__main__":
    success = patch_validation_metrics()
    sys.exit(0 if success else 1)

