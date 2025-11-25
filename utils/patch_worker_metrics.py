#!/usr/bin/env python3
"""
Patch script to add per-worker timing tracking.
Patches the trainer to collect per-worker metrics from timing info.
"""

import os
import sys
from pathlib import Path

# Get project root (two levels up from utils/)
PROJECT_ROOT = Path(__file__).parent.parent
# Path to verl installation (actual verl environment)
VERL_INSTALL_PATH = Path("/data/250010176/codes/verl")
# Fallback to project verl if installation doesn't exist
if not VERL_INSTALL_PATH.exists():
    VERL_INSTALL_PATH = PROJECT_ROOT / "verl"

# Path to patch (in verl installation)
TRAINER_PATH = VERL_INSTALL_PATH / "verl" / "trainer" / "ppo" / "ray_trainer.py"

def patch_trainer_collect_timing():
    """Patch trainer to collect timing from all workers."""
    
    if not TRAINER_PATH.exists():
        print(f"Error: {TRAINER_PATH} does not exist!")
        return False
    
    with open(TRAINER_PATH, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'all_workers_timing' in content and 'per_worker_summary' in content:
        print("Trainer already patched!")
        return True
    
    # Patch: After generate_sequences, collect timing info
    insert_marker = "print(\"validation generation end\")"
    
    if insert_marker not in content:
        print(f"Error: Cannot find insertion point in {TRAINER_PATH}")
        return False
    
    patch_code = '''
            print("validation generation end")
            
            # Collect rollout timing and token metrics
            rollout_timing_info = test_output_gen_batch.meta_info.get("timing", {})
            rollout_time = rollout_timing_info.get("generate_sequences", 0.0)
            
            # Extract per-worker timing info (min/max from timing dict)
            timing_min = rollout_timing_info.get("generation_timing/min", rollout_time)
            timing_max = rollout_timing_info.get("generation_timing/max", rollout_time)
            timing_avg = rollout_time
            
            # Get world size and tensor parallel size to group workers into engines
            try:
                if not self.async_rollout_mode:
                    num_workers = self.actor_rollout_wg.world_size
                else:
                    num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers if hasattr(self.config.actor_rollout_ref.rollout.agent, 'num_workers') else 1
                
                # Get TP size from config
                tp_size = getattr(self.config.actor_rollout_ref.rollout, 'tensor_model_parallel_size', 1)
            except:
                num_workers = 1
                tp_size = 1
            
            # Calculate number of replicas (TP groups)
            # Note: In VERL, "Replica" is the official term for grouping workers with TP
            # Each Replica contains world_size workers (world_size = TP × DP × PP)
            # In our case: TP=2, DP=8, PP=1, so world_size=2, num_replicas=8
            num_replicas = num_workers // tp_size if tp_size > 0 else num_workers
            
            # Calculate total response tokens and record per-prompt response lengths
            from verl.trainer.ppo.metric_utils import _compute_response_info
            response_info = _compute_response_info(test_output_gen_batch)
            total_response_tokens = response_info["response_length"].sum().item()
            total_prompt_tokens = response_info["prompt_length"].sum().item()
            
            # Extract per-prompt response lengths and prompt lengths (for scheduling algorithm)
            # These are key metrics for load balancing and scheduling
            response_lengths = response_info["response_length"].cpu().numpy().tolist()  # List of response lengths for each prompt
            prompt_lengths = response_info["prompt_length"].cpu().numpy().tolist()  # List of prompt lengths for each prompt
            
            # Get rollout.n (number of generations per prompt)
            # Since prompts are repeated n times with interleave=True, we need to group them
            rollout_n = getattr(self.config.actor_rollout_ref.rollout, 'n', 1)
            
            # Group response lengths by original prompt ID
            # Since prompts are repeated with interleave=True, the order is:
            # [Prompt 0 (gen 1), Prompt 0 (gen 2), ..., Prompt 0 (gen n),
            #  Prompt 1 (gen 1), Prompt 1 (gen 2), ..., Prompt 1 (gen n), ...]
            num_original_prompts = len(response_lengths) // rollout_n if rollout_n > 0 else len(response_lengths)
            per_prompt_response_lengths = {}  # {prompt_id: [length1, length2, ..., length_n]}
            per_prompt_prompt_lengths = {}  # {prompt_id: prompt_length}
            per_prompt_avg_response_lengths = {}  # {prompt_id: avg_length}
            
            for prompt_id in range(num_original_prompts):
                # Extract n response lengths for this prompt
                start_idx = prompt_id * rollout_n
                end_idx = start_idx + rollout_n
                prompt_response_lengths = response_lengths[start_idx:end_idx]
                prompt_length = prompt_lengths[start_idx] if start_idx < len(prompt_lengths) else 0
                
                per_prompt_response_lengths[prompt_id] = prompt_response_lengths
                per_prompt_prompt_lengths[prompt_id] = prompt_length
                
                # Calculate average response length for this prompt
                if prompt_response_lengths:
                    avg_length = sum(prompt_response_lengths) / len(prompt_response_lengths)
                    per_prompt_avg_response_lengths[prompt_id] = avg_length
            
            # Estimate per-worker timing (for individual worker metrics)
            # Note: This is an approximation since verl aggregates timing
            per_worker_times = []
            if num_workers > 1:
                # Estimate: assume timing is roughly distributed across workers
                # We know min/max, estimate others linearly
                for i in range(num_workers):
                    if timing_max > timing_min:
                        # Linear interpolation between min and max
                        worker_time = timing_min + (timing_max - timing_min) * i / (num_workers - 1) if num_workers > 1 else timing_avg
                    else:
                        worker_time = timing_avg
                    per_worker_times.append({
                        'worker_rank': i,
                        'rollout_time': worker_time,
                        'response_tokens': 0,  # Will calculate below
                        'replica_id': i // tp_size,  # Replica (TP group) this worker belongs to
                    })
            else:
                per_worker_times.append({
                    'worker_rank': 0,
                    'rollout_time': timing_avg,
                    'response_tokens': 0,
                    'replica_id': 0,
                })
            
            # Calculate per-worker/replica tokens based on actual prompt assignment
            # For default mode, use sequential assignment (each replica gets num_original_prompts / num_replicas prompts)
            per_replica_tokens = {}
            prompts_per_replica = num_original_prompts // num_replicas
            remainder = num_original_prompts % num_replicas
            
            prompt_idx = 0
            for replica_id in range(num_replicas):
                # Calculate how many prompts this replica should get
                num_prompts = prompts_per_replica + (1 if replica_id < remainder else 0)
                replica_tokens = 0
                for _ in range(num_prompts):
                    if prompt_idx < num_original_prompts:
                        # Sum all n generations for this prompt
                        if prompt_idx in per_prompt_response_lengths:
                            replica_tokens += sum(per_prompt_response_lengths[prompt_idx])
                        prompt_idx += 1
                per_replica_tokens[replica_id] = replica_tokens
            
            # Distribute tokens to workers based on replica assignment
            for worker_data in per_worker_times:
                replica_id = worker_data['replica_id']
                if replica_id in per_replica_tokens:
                    # Each worker in a replica gets the same tokens (TP group shares the work)
                    worker_data['response_tokens'] = int(per_replica_tokens[replica_id])
                else:
                    # Fallback: average distribution
                    worker_data['response_tokens'] = int(total_response_tokens / num_workers)
            
            # Group workers into replicas and record per-replica wall clock time
            # Note: "Replica" is the official VERL term (RolloutReplica class)
            # IMPORTANT: We directly use the actual measured rollout_time as the replica wall clock time
            # 
            # Key understanding:
            # - rollout_time is the actual measured wall clock time for the entire rollout
            # - Since all replicas execute in parallel, rollout_time = max(all replica times)
            # - This is the most accurate measurement, as it directly reflects the actual time
            #   from start to end, including all overheads (AllReduce, waiting, etc.)
            # - For each replica, we use this measured time (which represents the slowest replica's time)
            #   This is more accurate than estimating from worker times
            per_replica_times = []
            for replica_id in range(num_replicas):
                replica_workers = [w for w in per_worker_times if w['replica_id'] == replica_id]
                if replica_workers:
                    # Use the actual measured rollout_time as the replica wall clock time
                    # Note: rollout_time is the overall time (all replicas execute in parallel)
                    # Since replicas execute in parallel, rollout_time = max(all replica times)
                    # For each replica, we use this measured time (which is the actual wall clock time
                    # for the slowest replica, and a good approximation for others)
                    replica_wall_clock_time = rollout_time  # Direct measurement (actual wall clock time)
                    replica_tokens = sum(w['response_tokens'] for w in replica_workers)
                    
                    # Calculate average time for workers in this replica (for reference only)
                    replica_worker_avg_time = sum(w['rollout_time'] for w in replica_workers) / len(replica_workers)
                    replica_worker_min_time = min(w['rollout_time'] for w in replica_workers)
                    replica_worker_max_time = max(w['rollout_time'] for w in replica_workers)
                    
                    per_replica_times.append({
                        'replica_id': replica_id,
                        'wall_clock_time': replica_wall_clock_time,  # Direct measurement (actual wall clock time)
                        'rollout_time': replica_wall_clock_time,  # Alias for compatibility
                        'response_tokens': replica_tokens,
                        'worker_ranks': [w['worker_rank'] for w in replica_workers],
                        'num_workers': len(replica_workers),
                        'worker_avg_time': replica_worker_avg_time,  # Avg time of workers in this replica (estimated, for reference only)
                        'worker_min_time': replica_worker_min_time,  # Min time of workers in this replica (estimated, for reference only)
                        'worker_max_time': replica_worker_max_time,  # Max time of workers in this replica (estimated, for reference only)
                        'note': 'wall_clock_time is directly measured (actual time), not estimated from worker times',
                    })
            
            # Store metrics for later aggregation
            if not hasattr(self, '_validation_rollout_metrics'):
                self._validation_rollout_metrics = {
                    'total_rollout_time': 0.0,
                    'total_response_tokens': 0,
                    'total_prompt_tokens': 0,
                    'num_batches': 0,
                    'worker_times': [],  # List of per-batch worker timing
                    'all_workers_timing': [],  # List of all workers' timing per batch
                    'all_replicas_timing': [],  # List of all replicas' timing per batch (official VERL term)
                    'worker_min_times': [],  # Fastest worker times per batch
                    'worker_max_times': [],  # Slowest worker times per batch
                    'replica_min_times': [],  # Fastest replica times per batch
                    'replica_max_times': [],  # Slowest replica times per batch
                    'tp_size': tp_size,
                }
            else:
                # Update TP size if not set (should be consistent across batches)
                if 'tp_size' not in self._validation_rollout_metrics:
                    self._validation_rollout_metrics['tp_size'] = tp_size
            
            batch_idx = self._validation_rollout_metrics['num_batches']
            self._validation_rollout_metrics['total_rollout_time'] += rollout_time
            self._validation_rollout_metrics['total_response_tokens'] += total_response_tokens
            self._validation_rollout_metrics['total_prompt_tokens'] += total_prompt_tokens
            self._validation_rollout_metrics['num_batches'] += 1
            
            # Store per-batch worker timing info
            batch_worker_timing = {
                'batch': batch_idx,
                'avg_time': timing_avg,
                'min_time': timing_min,
                'max_time': timing_max,
                'response_tokens': total_response_tokens,
                'prompt_tokens': total_prompt_tokens,
                'num_workers': num_workers,
                'num_replicas': num_replicas,
                'tp_size': tp_size,
                'all_workers': per_worker_times,  # Estimated per-worker timing
                'all_replicas': per_replica_times,  # Per-replica (TP group) timing (official VERL term)
                'rollout_n': rollout_n,  # Number of generations per prompt
                'num_original_prompts': num_original_prompts,  # Number of original prompts (before repetition)
                'per_prompt_response_lengths': per_prompt_response_lengths,  # {prompt_id: [length1, ..., length_n]} - all n generations for each prompt
                'per_prompt_prompt_lengths': per_prompt_prompt_lengths,  # {prompt_id: prompt_length}
                'per_prompt_avg_response_lengths': per_prompt_avg_response_lengths,  # {prompt_id: avg_length} - average response length for each prompt
                'all_response_lengths': response_lengths,  # All response lengths (flat list, for backward compatibility)
                'all_prompt_lengths': prompt_lengths,  # All prompt lengths (flat list, for backward compatibility)
            }
            self._validation_rollout_metrics['worker_times'].append(batch_worker_timing)
            self._validation_rollout_metrics['all_workers_timing'].append(per_worker_times)
            self._validation_rollout_metrics['all_replicas_timing'].append(per_replica_times)
            self._validation_rollout_metrics['worker_min_times'].append(timing_min)
            self._validation_rollout_metrics['worker_max_times'].append(timing_max)
            if per_replica_times:
                self._validation_rollout_metrics['replica_min_times'].append(min(r['wall_clock_time'] for r in per_replica_times))
                self._validation_rollout_metrics['replica_max_times'].append(max(r['wall_clock_time'] for r in per_replica_times))
            else:
                self._validation_rollout_metrics['replica_min_times'].append(timing_avg)
                self._validation_rollout_metrics['replica_max_times'].append(timing_avg)
            
            print(f"Rollout batch {batch_idx + 1}: "
                  f"avg_time={timing_avg:.2f}s, "
                  f"min_time={timing_min:.2f}s (fastest worker), "
                  f"max_time={timing_max:.2f}s (slowest worker), "
                  f"response_tokens={total_response_tokens}, "
                  f"num_workers={num_workers}, "
                  f"num_replicas={num_replicas} (TP={tp_size})")
            
            # Print per-replica wall clock timing (directly measured, most accurate)
            # Note: "Replica" is the official VERL term (RolloutReplica class)
            # Wall clock time is directly measured from VERL's timing info (rollout_time)
            # Since all replicas execute in parallel, rollout_time = max(all replica times)
            # This includes all overheads (AllReduce, waiting, etc.) that are not captured by worker estimates
            if per_replica_times:
                print(f"  Per-replica wall clock timing (TP groups, official VERL term, directly measured):")
                print(f"    Note: rollout_time is the actual measured wall clock time (all replicas execute in parallel)")
                for replica_timing in sorted(per_replica_times, key=lambda x: x['replica_id']):
                    worker_ranks = replica_timing.get('worker_ranks', [])
                    wall_clock_time = replica_timing.get('wall_clock_time', replica_timing.get('rollout_time', 0.0))
                    worker_avg = replica_timing.get('worker_avg_time', wall_clock_time)
                    worker_min = replica_timing.get('worker_min_time', wall_clock_time)
                    worker_max = replica_timing.get('worker_max_time', wall_clock_time)
                    print(f"    Replica {replica_timing['replica_id']} (Workers {worker_ranks}): "
                          f"wall_clock_time={wall_clock_time:.2f}s (directly measured, actual wall clock time) "
                          f"[worker_avg={worker_avg:.2f}s (estimated, for reference), min={worker_min:.2f}s, max={worker_max:.2f}s], "
                          f"tokens={replica_timing['response_tokens']}")
            
            # Print per-worker timing if available (for reference)
            if per_worker_times and len(per_worker_times) <= 16:  # Only print if not too many
                print(f"  Per-worker timing (estimated, for reference):")
                for worker_timing in per_worker_times[:8]:  # Show first 8 only
                    replica_id = worker_timing.get('replica_id', 'unknown')
                    print(f"    Worker {worker_timing.get('worker_rank', 'unknown')} (Replica {replica_id}): "
                          f"time={worker_timing.get('rollout_time', 0.0):.2f}s, "
                          f"tokens={worker_timing.get('response_tokens', 0)}")
                if len(per_worker_times) > 8:
                    print(f"    ... (showing first 8 workers, {len(per_worker_times)} total)")'''
    
    new_content = content.replace(insert_marker, patch_code)
    
    # Update final metrics section
    return_marker = "        return metric_dict"
    
    if return_marker not in new_content:
        print("Warning: Cannot find return statement")
        return False
    
    final_metrics_code = '''
        # Add rollout timing and token metrics
        if hasattr(self, '_validation_rollout_metrics') and self._validation_rollout_metrics['num_batches'] > 0:
            metrics = self._validation_rollout_metrics
            
            # Get TP size from metrics
            tp_size = metrics.get('tp_size', 1)
            
            # Calculate per-worker statistics
            if metrics['worker_min_times']:
                fastest_worker_time = min(metrics['worker_min_times'])
                slowest_worker_time = max(metrics['worker_max_times'])
                avg_fastest_time = sum(metrics['worker_min_times']) / len(metrics['worker_min_times'])
                avg_slowest_time = sum(metrics['worker_max_times']) / len(metrics['worker_max_times'])
            else:
                fastest_worker_time = slowest_worker_time = avg_fastest_time = avg_slowest_time = 0.0
            
            # Calculate per-replica statistics (more meaningful)
            # Note: "Replica" is the official VERL term (RolloutReplica class)
            if metrics.get('replica_min_times') and metrics.get('replica_max_times'):
                fastest_replica_time = min(metrics['replica_min_times'])
                slowest_replica_time = max(metrics['replica_max_times'])
                avg_fastest_replica_time = sum(metrics['replica_min_times']) / len(metrics['replica_min_times'])
                avg_slowest_replica_time = sum(metrics['replica_max_times']) / len(metrics['replica_max_times'])
            else:
                fastest_replica_time = slowest_replica_time = avg_fastest_replica_time = avg_slowest_replica_time = 0.0
            
            # Aggregate per-worker timing across all batches
            all_workers_aggregated = {}
            for batch_workers in metrics['all_workers_timing']:
                for worker_timing in batch_workers:
                    worker_rank = worker_timing.get('worker_rank', 'unknown')
                    if worker_rank not in all_workers_aggregated:
                        all_workers_aggregated[worker_rank] = {
                            'worker_rank': worker_rank,
                            'total_time': 0.0,
                            'total_tokens': 0,
                            'num_batches': 0,
                            'replica_id': worker_timing.get('replica_id', worker_rank // tp_size),
                        }
                    all_workers_aggregated[worker_rank]['total_time'] += worker_timing.get('rollout_time', 0.0)
                    all_workers_aggregated[worker_rank]['total_tokens'] += worker_timing.get('response_tokens', 0)
                    all_workers_aggregated[worker_rank]['num_batches'] += 1
            
            # Calculate per-worker averages
            per_worker_summary = []
            for worker_rank, worker_data in sorted(all_workers_aggregated.items()):
                avg_time = worker_data['total_time'] / worker_data['num_batches'] if worker_data['num_batches'] > 0 else 0.0
                per_worker_summary.append({
                    'worker_rank': worker_rank,
                    'total_time_s': worker_data['total_time'],
                    'avg_time_s': avg_time,
                    'total_tokens': worker_data['total_tokens'],
                    'avg_tokens_per_batch': worker_data['total_tokens'] / worker_data['num_batches'] if worker_data['num_batches'] > 0 else 0,
                    'num_batches': worker_data['num_batches'],
                    'replica_id': worker_data.get('replica_id', worker_rank // tp_size),
                })
            
            # Aggregate per-replica timing across all batches (more meaningful)
            # Note: "Replica" is the official VERL term (RolloutReplica class)
            all_replicas_aggregated = {}
            for batch_replicas in metrics.get('all_replicas_timing', []):
                for replica_timing in batch_replicas:
                    replica_id = replica_timing.get('replica_id', 'unknown')
                    if replica_id not in all_replicas_aggregated:
                        all_replicas_aggregated[replica_id] = {
                            'replica_id': replica_id,
                            'total_time': 0.0,
                            'total_tokens': 0,
                            'num_batches': 0,
                            'worker_ranks': replica_timing.get('worker_ranks', []),
                            'num_workers': replica_timing.get('num_workers', tp_size),
                        }
                    # Use wall_clock_time if available, otherwise fall back to rollout_time
                    replica_time = replica_timing.get('wall_clock_time', replica_timing.get('rollout_time', 0.0))
                    all_replicas_aggregated[replica_id]['total_time'] += replica_time
                    all_replicas_aggregated[replica_id]['total_tokens'] += replica_timing.get('response_tokens', 0)
                    all_replicas_aggregated[replica_id]['num_batches'] += 1
            
            # Calculate per-replica averages
            per_replica_summary = []
            for replica_id, replica_data in sorted(all_replicas_aggregated.items()):
                avg_time = replica_data['total_time'] / replica_data['num_batches'] if replica_data['num_batches'] > 0 else 0.0
                per_replica_summary.append({
                    'replica_id': replica_id,
                    'total_wall_clock_time_s': replica_data['total_time'],  # Total wall clock time across all batches
                    'total_time_s': replica_data['total_time'],  # Alias for compatibility
                    'avg_wall_clock_time_s': avg_time,  # Average wall clock time per batch
                    'avg_time_s': avg_time,  # Alias for compatibility
                    'total_tokens': replica_data['total_tokens'],
                    'avg_tokens_per_batch': replica_data['total_tokens'] / replica_data['num_batches'] if replica_data['num_batches'] > 0 else 0,
                    'num_batches': replica_data['num_batches'],
                    'worker_ranks': replica_data.get('worker_ranks', []),
                    'num_workers': replica_data.get('num_workers', tp_size),
                })
            
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
                'rollout/fastest_replica_time_s': fastest_replica_time,
                'rollout/slowest_replica_time_s': slowest_replica_time,
                'rollout/avg_fastest_replica_time_s': avg_fastest_replica_time,
                'rollout/avg_slowest_replica_time_s': avg_slowest_replica_time,
                'rollout/tp_size': tp_size,
                'rollout/num_replicas': len(per_replica_summary) if per_replica_summary else 0,
            })
            
            # Collect all per-prompt response lengths across all batches (for scheduling algorithm)
            # Group by original prompt ID, collecting all n generations for each prompt
            all_per_prompt_response_lengths_dict = {}  # {prompt_id: [all n generations across batches]}
            all_per_prompt_prompt_lengths_dict = {}  # {prompt_id: prompt_length}
            all_per_prompt_avg_response_lengths_dict = {}  # {prompt_id: avg_length}
            all_response_lengths_flat = []  # Flat list for statistics
            all_prompt_lengths_flat = []  # Flat list for statistics
            
            # Track global prompt ID across batches
            global_prompt_id = 0
            for batch_metrics in metrics['worker_times']:
                batch_per_prompt_response_lengths = batch_metrics.get('per_prompt_response_lengths', {})
                batch_per_prompt_prompt_lengths = batch_metrics.get('per_prompt_prompt_lengths', {})
                batch_per_prompt_avg_response_lengths = batch_metrics.get('per_prompt_avg_response_lengths', {})
                
                # Process each prompt in this batch
                for local_prompt_id in sorted(batch_per_prompt_response_lengths.keys()):
                    prompt_response_lengths = batch_per_prompt_response_lengths[local_prompt_id]
                    prompt_length = batch_per_prompt_prompt_lengths.get(local_prompt_id, 0)
                    prompt_avg_length = batch_per_prompt_avg_response_lengths.get(local_prompt_id, 0)
                    
                    # Use global prompt ID to track across batches
                    all_per_prompt_response_lengths_dict[global_prompt_id] = prompt_response_lengths
                    all_per_prompt_prompt_lengths_dict[global_prompt_id] = prompt_length
                    all_per_prompt_avg_response_lengths_dict[global_prompt_id] = prompt_avg_length
                    
                    # Add to flat lists for statistics
                    all_response_lengths_flat.extend(prompt_response_lengths)
                    all_prompt_lengths_flat.append(prompt_length)
                    
                    global_prompt_id += 1
            
            # Calculate statistics for response lengths (important for scheduling)
            # Note: numpy is already imported at the top of the file, no need to import again
            response_length_stats = {}
            prompt_length_stats = {}
            avg_response_length_stats = {}  # Statistics for per-prompt average response lengths
            
            if all_response_lengths_flat:
                response_lengths_array = np.array(all_response_lengths_flat)
                response_length_stats = {
                    'mean': float(np.mean(response_lengths_array)),
                    'median': float(np.median(response_lengths_array)),
                    'std': float(np.std(response_lengths_array)),
                    'min': int(np.min(response_lengths_array)),
                    'max': int(np.max(response_lengths_array)),
                    'p25': float(np.percentile(response_lengths_array, 25)),
                    'p75': float(np.percentile(response_lengths_array, 75)),
                    'p90': float(np.percentile(response_lengths_array, 90)),
                    'p95': float(np.percentile(response_lengths_array, 95)),
                    'p99': float(np.percentile(response_lengths_array, 99)),
                }
            
            if all_prompt_lengths_flat:
                prompt_lengths_array = np.array(all_prompt_lengths_flat)
                prompt_length_stats = {
                    'mean': float(np.mean(prompt_lengths_array)),
                    'median': float(np.median(prompt_lengths_array)),
                    'std': float(np.std(prompt_lengths_array)),
                    'min': int(np.min(prompt_lengths_array)),
                    'max': int(np.max(prompt_lengths_array)),
                }
            
            # Calculate statistics for per-prompt average response lengths (key for scheduling)
            if all_per_prompt_avg_response_lengths_dict:
                avg_response_lengths_list = list(all_per_prompt_avg_response_lengths_dict.values())
                avg_response_lengths_array = np.array(avg_response_lengths_list)
                avg_response_length_stats = {
                    'mean': float(np.mean(avg_response_lengths_array)),
                    'median': float(np.median(avg_response_lengths_array)),
                    'std': float(np.std(avg_response_lengths_array)),
                    'min': float(np.min(avg_response_lengths_array)),
                    'max': float(np.max(avg_response_lengths_array)),
                    'p25': float(np.percentile(avg_response_lengths_array, 25)),
                    'p75': float(np.percentile(avg_response_lengths_array, 75)),
                    'p90': float(np.percentile(avg_response_lengths_array, 90)),
                    'p95': float(np.percentile(avg_response_lengths_array, 95)),
                    'p99': float(np.percentile(avg_response_lengths_array, 99)),
                }
                prompt_length_stats = {
                    'mean': float(np.mean(prompt_lengths_array)),
                    'median': float(np.median(prompt_lengths_array)),
                    'std': float(np.std(prompt_lengths_array)),
                    'min': int(np.min(prompt_lengths_array)),
                    'max': int(np.max(prompt_lengths_array)),
                }
            
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
                        'num_workers': len(per_worker_summary),
                        'num_replicas': len(per_replica_summary),
                        'tp_size': tp_size,
                        'fastest_worker_time_s': fastest_worker_time,
                        'slowest_worker_time_s': slowest_worker_time,
                        'avg_fastest_worker_time_s': avg_fastest_time,
                        'avg_slowest_worker_time_s': avg_slowest_time,
                        'worker_time_difference_s': slowest_worker_time - fastest_worker_time,
                        'fastest_replica_time_s': fastest_replica_time,
                        'slowest_replica_time_s': slowest_replica_time,
                        'avg_fastest_replica_time_s': avg_fastest_replica_time,
                        'avg_slowest_replica_time_s': avg_slowest_replica_time,
                        'replica_time_difference_s': slowest_replica_time - fastest_replica_time,
                        'response_length_stats': response_length_stats,  # Statistics for all individual response lengths
                        'avg_response_length_stats': avg_response_length_stats,  # Statistics for per-prompt average response lengths (key for scheduling)
                        'prompt_length_stats': prompt_length_stats,  # Statistics for prompt lengths
                    },
                    'per_replica_summary': per_replica_summary,  # More meaningful: grouped by TP (official VERL term)
                    'per_worker_summary': per_worker_summary,  # For reference
                    'per_batch_metrics': metrics['worker_times'],  # Includes per_prompt_response_lengths (grouped by prompt) and per_prompt_avg_response_lengths
                    'all_per_prompt_response_lengths': all_per_prompt_response_lengths_dict,  # {prompt_id: [all n generations]} - grouped by prompt ID
                    'all_per_prompt_avg_response_lengths': all_per_prompt_avg_response_lengths_dict,  # {prompt_id: avg_length} - average response length for each prompt (key for scheduling)
                    'all_per_prompt_prompt_lengths': all_per_prompt_prompt_lengths_dict,  # {prompt_id: prompt_length}
                }, f, indent=2)
            print(f"\\nSaved metrics to: {worker_metrics_file}")
            print(f"  - Per-replica metrics (grouped by TP, official VERL term, recommended)")
            print(f"  - Per-worker metrics (for reference)")
            print(f"  - Per-prompt response lengths (grouped by prompt ID, all n generations recorded)")
            print(f"  - Per-prompt average response lengths (key for scheduling algorithm)")
            print(f"  - Per-prompt prompt lengths")
            if avg_response_length_stats:
                print(f"  - Avg response length stats (per prompt): mean={avg_response_length_stats['mean']:.1f}, "
                      f"median={avg_response_length_stats['median']:.1f}, "
                      f"std={avg_response_length_stats['std']:.1f}, "
                      f"min={avg_response_length_stats['min']:.1f}, max={avg_response_length_stats['max']:.1f}")
            if response_length_stats:
                print(f"  - Individual response length stats: mean={response_length_stats['mean']:.1f}, "
                      f"median={response_length_stats['median']:.1f}, "
                      f"std={response_length_stats['std']:.1f}, "
                      f"min={response_length_stats['min']}, max={response_length_stats['max']}")
            
            print(f"\\n=== Rollout Metrics Summary ===")
            print(f"Total rollout time: {metrics['total_rollout_time']:.2f}s")
            print(f"Total response tokens: {metrics['total_response_tokens']}")
            print(f"Total prompt tokens: {metrics['total_prompt_tokens']}")
            print(f"Number of batches: {metrics['num_batches']}")
            print(f"Throughput: {metrics['total_response_tokens'] / (metrics['total_rollout_time'] + 1e-9):.2f} tokens/sec")
            print(f"TP size: {tp_size}, Number of replicas: {len(per_replica_summary)}")
            print(f"\\n--- Per-Replica Timing (TP Groups, Official VERL Term) ---")
            print(f"Fastest replica (min): {fastest_replica_time:.2f}s")
            print(f"Slowest replica (max): {slowest_replica_time:.2f}s")
            print(f"Avg fastest replica: {avg_fastest_replica_time:.2f}s")
            print(f"Avg slowest replica: {avg_slowest_replica_time:.2f}s")
            print(f"Replica time difference (max - min): {slowest_replica_time - fastest_replica_time:.2f}s")
            if per_replica_summary:
                print(f"\\n--- All Replicas Summary (Wall Clock Time) ---")
                for replica in sorted(per_replica_summary, key=lambda x: x['replica_id']):
                    worker_ranks_str = ','.join(map(str, replica['worker_ranks']))
                    total_wall_clock = replica.get('total_wall_clock_time_s', replica['total_time_s'])
                    avg_wall_clock = replica.get('avg_wall_clock_time_s', replica['avg_time_s'])
                    print(f"Replica {replica['replica_id']} (Workers [{worker_ranks_str}]): "
                          f"total_wall_clock_time={total_wall_clock:.2f}s, "
                          f"avg_wall_clock_time={avg_wall_clock:.2f}s, "
                          f"total_tokens={replica['total_tokens']}, "
                          f"avg_tokens/batch={replica['avg_tokens_per_batch']:.1f}")
            print(f"\\n--- Per-Worker Timing (For Reference) ---")
            print(f"Fastest worker (min): {fastest_worker_time:.2f}s")
            print(f"Slowest worker (max): {slowest_worker_time:.2f}s")
            print(f"Avg fastest worker: {avg_fastest_time:.2f}s")
            print(f"Avg slowest worker: {avg_slowest_time:.2f}s")
            print(f"Worker time difference (max - min): {slowest_worker_time - fastest_worker_time:.2f}s")
            print(f"===============================\\n")
            # Reset for next validation
            delattr(self, '_validation_rollout_metrics')
        
        return metric_dict'''
    
    new_content = new_content.replace(return_marker, final_metrics_code)
    
    # Write back
    backup_path = TRAINER_PATH.with_suffix('.py.backup')
    if not backup_path.exists():
        original_content = content
        with open(backup_path, 'w') as f:
            f.write(original_content)
        print(f"Created backup: {backup_path}")
    
    with open(TRAINER_PATH, 'w') as f:
        f.write(new_content)
    
    print(f"Successfully patched trainer: {TRAINER_PATH}")
    return True

def main():
    """Apply all patches."""
    print("Patching verl to record per-worker metrics...")
    print("=" * 60)
    print(f"VERL installation path: {VERL_INSTALL_PATH}")
    print(f"Target file: {TRAINER_PATH}")
    print("=" * 60)
    
    success = patch_trainer_collect_timing()
    
    if success:
        print("=" * 60)
        print("Patch applied successfully!")
        print("")
        print("The validation will now record:")
        print("  - Per-replica metrics (TP groups, official VERL term, recommended for analysis)")
        print("  - Per-worker metrics (for reference)")
        print("  - Fastest/slowest replica and worker identification")
        print("  - All metrics saved to JSON file")
        print("")
        print("Note:")
        print("  - Per-replica metrics are grouped by TP size (e.g., TP=2 means 2 workers per replica)")
        print("  - 'Replica' is the official VERL term (RolloutReplica class)")
        print("  - Per-worker timing is estimated from min/max values")
        print("  - Replica metrics are more meaningful since prompts require TP groups to complete")
        return True
    else:
        print("Patch failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
