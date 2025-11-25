# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        print(f"[DEBUG] Validation dataset info:")
        print(f"  val_dataset length: {len(self.val_dataset)}")
        print(f"  val_batch_size (from config): {self.config.data.val_batch_size}")
        print(f"  val_batch_size (final): {val_batch_size}")

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
        
        print(f"  val_dataloader length: {len(self.val_dataloader)}")
        print(f"  Expected batches: {len(self.val_dataloader)}")

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        # DEBUG_BREAKPOINT_RayPPOTrainer._validate
        import os
        if os.getenv("VERL_DEBUG", "0") == "1":
            import ipdb
            print("\n======================================================================")
            print(f"🐛 DEBUG BREAKPOINT: RayPPOTrainer._validate")
            print(f"======================================================================\n")
            ipdb.set_trace()
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        batch_idx = 0
        for test_data in self.val_dataloader:
            print(f"[DEBUG] Validation batch {batch_idx}:")
            print(f"  test_data keys: {list(test_data.keys()) if isinstance(test_data, dict) else 'N/A'}")
            if isinstance(test_data, dict) and "input_ids" in test_data:
                print(f"  test_data batch size (input_ids): {len(test_data['input_ids'])}")
            
            test_batch = DataProto.from_single_dict(test_data)
            print(f"  test_batch length: {len(test_batch)}")

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs (must be done before _get_gen_batch which removes input_ids)
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            # Get generation batch (removes input_ids, attention_mask, position_ids)
            test_gen_batch = self._get_gen_batch(test_batch)
            
            # Apply scheduling BEFORE repeat (prompt-level scheduling) - aligned with training phase
            # This ensures we schedule original prompts, not repeated ones
            print(f"  Before scheduling: {len(test_gen_batch)} prompts")
            test_gen_batch = self._apply_scheduling(test_gen_batch)
            print(f"  After scheduling: {len(test_gen_batch)} prompts")
            
            # Repeat after scheduling: each prompt is repeated n times
            val_kwargs_n = self.config.actor_rollout_ref.rollout.val_kwargs.n
            print(f"  val_kwargs.n: {val_kwargs_n}")
            test_gen_batch = test_gen_batch.repeat(
                repeat_times=val_kwargs_n, interleave=True
            )
            print(f"  After repeat: {len(test_gen_batch)} prompts")
            
            # Now we need to repeat test_batch as well to align with test_gen_batch
            # But we need to preserve the original test_batch for later union
            # So we'll create a repeated version for union later
            test_batch_repeated = test_batch.repeat(
                repeat_times=val_kwargs_n, interleave=True
            )
            
            # Get TP size from config to include in meta_info
            tp_size = getattr(self.config.actor_rollout_ref.rollout, 'tensor_model_parallel_size', 1)
            
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
                "tp_size": tp_size,  # Include TP size for bin packing dispatch
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")
            # Preserve scheduling metadata if present on the repeated batch
            for key in ["shuffle_scheduled", "shuffle_seed", "shuffled_indices", "bin_packing_scheduled"]:
                if key in test_batch.meta_info:
                    test_gen_batch.meta_info[key] = test_batch.meta_info[key]

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            
            # Record start time before generation
            import time
            generation_start_time = time.time()
            
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            
            # Record end time after generation
            generation_end_time = time.time()
            measured_wall_clock_time = generation_end_time - generation_start_time

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print(f"  After unpad: {len(test_output_gen_batch)} prompts in output")

            
            print("validation generation end")
            
            # Collect rollout timing and token metrics
            rollout_timing_info = test_output_gen_batch.meta_info.get("timing", {})
            rollout_time = rollout_timing_info.get("generate_sequences", 0.0)
            
            # Use measured wall clock time if available, otherwise use rollout_time from timing info
            # measured_wall_clock_time is more accurate as it's directly recorded start->end
            if measured_wall_clock_time > 0:
                overall_wall_clock_time = measured_wall_clock_time
            else:
                overall_wall_clock_time = rollout_time
            
            # Extract per-worker timing info
            # NEW: Try to get actual per-worker timings from meta_info (if available from all_gather)
            all_worker_timings = rollout_timing_info.get("generation_timing/all_workers", None)
            
            if all_worker_timings is not None and len(all_worker_timings) > 0:
                # We have actual per-worker timings from all_gather!
                timing_min = min(all_worker_timings)
                timing_max = max(all_worker_timings)
                timing_avg = sum(all_worker_timings) / len(all_worker_timings)
                print(f"[DEBUG] Got actual per-worker timings from all_gather: {len(all_worker_timings)} workers")
            else:
                # Fall back to min/max from timing dict (old behavior)
                timing_min = rollout_timing_info.get("generation_timing/min", rollout_time)
                timing_max = rollout_timing_info.get("generation_timing/max", rollout_time)
                timing_avg = rollout_time
                print(f"[DEBUG] Using min/max from timing dict (actual per-worker timings not available)")
            
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
            
            if 'val_kwargs_n' in locals():
                actual_n = val_kwargs_n
            else:
                actual_n = getattr(self.config.actor_rollout_ref.rollout.val_kwargs, 'n', 
                                 getattr(self.config.actor_rollout_ref.rollout, 'n', 1))
            
            # Also keep rollout.n for reference (used in training)
            rollout_n = getattr(self.config.actor_rollout_ref.rollout, 'n', 1)
            print(f"  val_kwargs.n (actual_n for validation): {actual_n}")
            print(f"  rollout.n (for reference): {rollout_n}")
            
            # Group response lengths by original prompt ID
            # IMPORTANT: If data was shuffled, we need to recover the original prompt_id mapping
            # Check if shuffle was applied
            shuffled_indices = test_output_gen_batch.meta_info.get("shuffled_indices", None)
            is_shuffled = shuffled_indices is not None
            
            if is_shuffled:
                print(f"[DEBUG] Data was shuffled, recovering original prompt_id mapping using shuffled_indices")
                # Create reverse mapping: from shuffled position to original prompt_id
                # shuffled_indices[i] is the original index of the item at position i after shuffle
                num_original_prompts = len(response_lengths) // actual_n if actual_n > 0 else len(response_lengths)
                
                # Build mapping: shuffled_position -> original_prompt_id
                shuffled_to_original_prompt = {}
                for shuffled_pos in range(len(shuffled_indices)):
                    original_idx = shuffled_indices[shuffled_pos]
                    original_prompt_id = original_idx // actual_n
                    shuffled_to_original_prompt[shuffled_pos] = original_prompt_id
                
                # Group response_lengths by original prompt_id
                per_prompt_response_lengths = {}  # {prompt_id: [length1, length2, ..., length_n]}
                per_prompt_prompt_lengths = {}  # {prompt_id: prompt_length}
                per_prompt_avg_response_lengths = {}  # {prompt_id: avg_length}
                
                for shuffled_pos in range(len(response_lengths)):
                    original_prompt_id = shuffled_to_original_prompt.get(shuffled_pos, shuffled_pos // actual_n)
                    if original_prompt_id not in per_prompt_response_lengths:
                        per_prompt_response_lengths[original_prompt_id] = []
                        per_prompt_prompt_lengths[original_prompt_id] = prompt_lengths[shuffled_pos] if shuffled_pos < len(prompt_lengths) else 0
                    per_prompt_response_lengths[original_prompt_id].append(response_lengths[shuffled_pos])
                
                # Calculate average for each prompt
                for prompt_id in per_prompt_response_lengths:
                    prompt_response_lengths = per_prompt_response_lengths[prompt_id]
                    if prompt_response_lengths:
                        avg_length = sum(prompt_response_lengths) / len(prompt_response_lengths)
                        per_prompt_avg_response_lengths[prompt_id] = avg_length
                
            else:
                # Original order: [Prompt 0 (gen 1), Prompt 0 (gen 2), ..., Prompt 0 (gen n), Prompt 1 (gen 1), ...]
                num_original_prompts = len(response_lengths) // actual_n if actual_n > 0 else len(response_lengths)
                print(f"  Calculated num_original_prompts: {num_original_prompts} (from {len(response_lengths)} responses / {actual_n})")
                per_prompt_response_lengths = {}  # {prompt_id: [length1, length2, ..., length_n]}
                per_prompt_prompt_lengths = {}  # {prompt_id: prompt_length}
                per_prompt_avg_response_lengths = {}  # {prompt_id: avg_length}
                
                for prompt_id in range(num_original_prompts):
                    # Extract n response lengths for this prompt
                    # Use actual_n (val_kwargs.n) for validation, not rollout.n
                    start_idx = prompt_id * actual_n
                    end_idx = start_idx + actual_n
                    prompt_response_lengths = response_lengths[start_idx:end_idx]
                    prompt_length = prompt_lengths[start_idx] if start_idx < len(prompt_lengths) else 0
                    
                    per_prompt_response_lengths[prompt_id] = prompt_response_lengths
                    per_prompt_prompt_lengths[prompt_id] = prompt_length
                    
                    # Calculate average response length for this prompt
                    if prompt_response_lengths:
                        avg_length = sum(prompt_response_lengths) / len(prompt_response_lengths)
                        per_prompt_avg_response_lengths[prompt_id] = avg_length
            
            # Get per-worker timing (actual measurements if available, otherwise estimate)
            per_worker_times = []
            if all_worker_timings is not None and len(all_worker_timings) == num_workers:
                # Use actual per-worker timings from all_gather (directly measured!)
                for i in range(num_workers):
                    if i < len(all_worker_timings):
                        worker_time = all_worker_timings[i]
                    else:
                        worker_time = timing_avg
                    per_worker_times.append({
                        'worker_rank': i,
                        'rollout_time': worker_time,
                        'response_tokens': 0,  # Will calculate below
                        'replica_id': i // tp_size,  # Replica (TP group) this worker belongs to
                        'note': 'actual time from all_gather (directly measured)',
                    })
                print(f"[DEBUG] Using actual per-worker timings from all_gather")
            elif num_workers > 1:
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
                        'note': 'estimated time (linear interpolation between min/max)',
                    })
                print(f"[DEBUG] Using estimated per-worker timings (linear interpolation)")
            else:
                per_worker_times.append({
                    'worker_rank': 0,
                    'rollout_time': timing_avg,
                    'response_tokens': 0,
                    'replica_id': 0,
                    'note': 'single worker',
                })
            
            # Calculate per-worker/replica tokens based on actual prompt assignment
            # Try to get actual prompt assignment from scheduler (bin_packing or shuffle)
            per_replica_tokens = {}
            per_worker_tokens_calculated = False
            
            try:
                # Check schedule mode (check both VERL_SCHEDULE_MODE and VERL_ENABLE_SHUFFLE)
                import os
                schedule_mode_check = os.getenv("VERL_SCHEDULE_MODE", "default").lower()
                shuffle_enabled = os.getenv("VERL_ENABLE_SHUFFLE", "false").lower() in ["true", "1", "yes"]
                
                # Determine actual schedule mode
                if schedule_mode_check == "bin_packing":
                    actual_schedule_mode = "bin_packing"
                elif schedule_mode_check == "shuffle" or shuffle_enabled:
                    actual_schedule_mode = "shuffle"
                else:
                    actual_schedule_mode = "default"
                
                # Safety check: If shuffle is expected but schedule_mode is still default, raise error
                expected_shuffle = (schedule_mode_check == "shuffle" or shuffle_enabled)
                if expected_shuffle and actual_schedule_mode == "default":
                    error_msg = (
                        f"ERROR: Shuffle scheduling was requested but schedule_mode is still 'default'!\n"
                        f"  VERL_SCHEDULE_MODE={os.getenv('VERL_SCHEDULE_MODE', 'not set')}\n"
                        f"  VERL_ENABLE_SHUFFLE={os.getenv('VERL_ENABLE_SHUFFLE', 'not set')}\n"
                        f"  Detected schedule_mode={actual_schedule_mode}\n"
                        f"  This indicates that environment variables are not being passed to Ray workers correctly.\n"
                        f"  Please check that get_ppo_ray_runtime_env() includes these variables in runtime_env."
                    )
                    print(f"[FATAL ERROR] {error_msg}")
                    raise RuntimeError(error_msg)
                
                # Check if bin packing scheduler is available
                if actual_schedule_mode == "bin_packing":
                    from verl.utils.bin_packing_scheduler import get_bin_packing_scheduler
                    bin_packing_scheduler = get_bin_packing_scheduler()
                    
                    if bin_packing_scheduler is not None and per_prompt_avg_response_lengths:
                        # Calculate tokens per replica based on bin packing schedule
                        for replica_id in range(num_replicas):
                            prompt_ids = bin_packing_scheduler.get_prompts_for_replica(replica_id)
                            replica_tokens = 0
                            for prompt_id in prompt_ids:
                                # per_prompt_avg_response_lengths uses integer keys
                                if prompt_id in per_prompt_avg_response_lengths:
                                    replica_tokens += per_prompt_avg_response_lengths[prompt_id]
                            per_replica_tokens[replica_id] = replica_tokens
                        
                        # Distribute replica tokens to workers
                        for worker_data in per_worker_times:
                            replica_id = worker_data['replica_id']
                            if replica_id in per_replica_tokens:
                                worker_data['response_tokens'] = int(per_replica_tokens[replica_id])
                            else:
                                worker_data['response_tokens'] = int(total_response_tokens / num_workers)
                        per_worker_tokens_calculated = True
                        print(f"[DEBUG] Calculated per-worker tokens based on bin packing schedule")
                
                # Check if shuffle scheduler is available
                elif actual_schedule_mode == "shuffle":
                    from verl.utils.shuffle_scheduler import get_shuffle_scheduler
                    shuffle_scheduler = get_shuffle_scheduler()
                    
                    if shuffle_scheduler is not None and per_prompt_avg_response_lengths:
                        # Get shuffled indices to determine which prompts go to which replica
                        shuffled_indices = shuffle_scheduler.get_shuffled_indices(len(response_lengths), actual_n)
                        prompts_per_replica = len(response_lengths) // num_replicas
                        
                        # Calculate tokens per replica based on shuffled assignment
                        for replica_id in range(num_replicas):
                            start_pos = replica_id * prompts_per_replica
                            end_pos = (replica_id + 1) * prompts_per_replica if replica_id < num_replicas - 1 else len(shuffled_indices)
                            
                            # Get the shuffled indices for this replica's positions
                            replica_shuffled_indices = shuffled_indices[start_pos:end_pos]
                            
                            # Convert to original prompt IDs and calculate tokens
                            replica_tokens = 0
                            replica_prompt_ids_set = set()
                            for shuffled_idx in replica_shuffled_indices:
                                original_prompt_id = shuffled_idx // actual_n
                                replica_prompt_ids_set.add(original_prompt_id)
                            
                            # Calculate tokens for this replica
                            for prompt_id in replica_prompt_ids_set:
                                # per_prompt_avg_response_lengths uses integer keys
                                if prompt_id in per_prompt_avg_response_lengths:
                                    replica_tokens += per_prompt_avg_response_lengths[prompt_id]
                            
                            per_replica_tokens[replica_id] = replica_tokens
                        
                        # Distribute replica tokens to workers
                        for worker_data in per_worker_times:
                            replica_id = worker_data['replica_id']
                            if replica_id in per_replica_tokens:
                                worker_data['response_tokens'] = int(per_replica_tokens[replica_id])
                            else:
                                worker_data['response_tokens'] = int(total_response_tokens / num_workers)
                        per_worker_tokens_calculated = True
                        print(f"[DEBUG] Calculated per-worker tokens based on shuffle schedule")
                
                # For default mode, use sequential assignment
                elif actual_schedule_mode == "default":
                    # Default: sequential assignment (each replica gets num_original_prompts / num_replicas prompts)
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
                    
                    # Distribute replica tokens to workers
                    for worker_data in per_worker_times:
                        replica_id = worker_data['replica_id']
                        if replica_id in per_replica_tokens:
                            worker_data['response_tokens'] = int(per_replica_tokens[replica_id])
                        else:
                            worker_data['response_tokens'] = int(total_response_tokens / num_workers)
                    per_worker_tokens_calculated = True
                    print(f"[DEBUG] Calculated per-worker tokens based on sequential assignment (default mode)")
                
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to calculate per-worker tokens from scheduler: {e}")
            
            # Fallback: if we couldn't calculate from scheduler, use average distribution
            if not per_worker_tokens_calculated:
                tokens_per_worker = total_response_tokens / num_workers if num_workers > 0 else total_response_tokens
                for worker_data in per_worker_times:
                    worker_data['response_tokens'] = int(tokens_per_worker)
                print(f"[DEBUG] Using average token distribution (scheduler not available or failed)")
            
            
            # Store metrics for later aggregation
            if not hasattr(self, '_validation_rollout_metrics'):
                self._validation_rollout_metrics = {
                    'total_rollout_time': 0.0,  # Total wall clock time (all replicas combined)
                    'total_response_tokens': 0,
                    'total_prompt_tokens': 0,
                    'num_batches': 0,
                    'worker_times': [],  # List of per-batch worker timing
                    'all_workers_timing': [],  # List of all workers' timing per batch (actual measurements)
                    'worker_min_times': [],  # Fastest worker times per batch
                    'worker_max_times': [],  # Slowest worker times per batch
                    'tp_size': tp_size,
                    'num_replicas': num_replicas,  # Number of replicas (for reference)
                }
            else:
                # Update TP size if not set (should be consistent across batches)
                if 'tp_size' not in self._validation_rollout_metrics:
                    self._validation_rollout_metrics['tp_size'] = tp_size
                if 'num_replicas' not in self._validation_rollout_metrics:
                    self._validation_rollout_metrics['num_replicas'] = num_replicas
            
            batch_idx = self._validation_rollout_metrics['num_batches']
            # Record overall wall clock time (all replicas combined)
            # This is the total time from start to finish for all replicas executing in parallel
            self._validation_rollout_metrics['total_rollout_time'] += overall_wall_clock_time if 'overall_wall_clock_time' in locals() else rollout_time
            self._validation_rollout_metrics['total_response_tokens'] += total_response_tokens
            self._validation_rollout_metrics['total_prompt_tokens'] += total_prompt_tokens
            self._validation_rollout_metrics['num_batches'] += 1
            
            per_replica_prompt_ids = {}
            per_replica_prompt_response_lengths = {}  # NEW: record response lengths for each prompt in each replica
            import os
            # Check schedule mode (check both VERL_SCHEDULE_MODE and VERL_ENABLE_SHUFFLE)
            schedule_mode_check = os.getenv("VERL_SCHEDULE_MODE", "default").lower()
            shuffle_enabled = os.getenv("VERL_ENABLE_SHUFFLE", "false").lower() in ["true", "1", "yes"]
            
            # Debug: print environment variables
            print(f"[DEBUG] Recording per-replica prompt assignment:")
            print(f"  VERL_SCHEDULE_MODE={schedule_mode_check}")
            print(f"  VERL_ENABLE_SHUFFLE={os.getenv('VERL_ENABLE_SHUFFLE', 'not set')}, shuffle_enabled={shuffle_enabled}")
            
            # Determine actual schedule mode
            if schedule_mode_check == "bin_packing":
                schedule_mode = "bin_packing"
            elif schedule_mode_check == "shuffle" or shuffle_enabled:
                schedule_mode = "shuffle"
            else:
                schedule_mode = "default"
            
            print(f"[DEBUG] Determined schedule_mode: {schedule_mode}")
            
            # Safety check: If shuffle is expected but schedule_mode is still default, raise error
            expected_shuffle = (schedule_mode_check == "shuffle" or shuffle_enabled)
            if expected_shuffle and schedule_mode == "default":
                error_msg = (
                    f"ERROR: Shuffle scheduling was requested but schedule_mode is still 'default'!\n"
                    f"  VERL_SCHEDULE_MODE={os.getenv('VERL_SCHEDULE_MODE', 'not set')}\n"
                    f"  VERL_ENABLE_SHUFFLE={os.getenv('VERL_ENABLE_SHUFFLE', 'not set')}\n"
                    f"  Detected schedule_mode={schedule_mode}\n"
                    f"  This indicates that environment variables are not being passed to Ray workers correctly.\n"
                    f"  Please check that get_ppo_ray_runtime_env() includes these variables in runtime_env."
                )
                print(f"[FATAL ERROR] {error_msg}")
                raise RuntimeError(error_msg)
            
            try:
                if schedule_mode == "bin_packing":
                    # Get prompt assignment from bin packing scheduler
                    from verl.utils.bin_packing_scheduler import get_bin_packing_scheduler
                    bin_packing_scheduler = get_bin_packing_scheduler()
                    if bin_packing_scheduler is not None:
                        for replica_id in range(num_replicas):
                            prompt_ids = bin_packing_scheduler.get_prompts_for_replica(replica_id)
                            per_replica_prompt_ids[replica_id] = sorted(prompt_ids)  # Sort for readability
                            
                            # Record response lengths for each prompt in this replica
                        replica_prompt_response_lengths = {}
                        for prompt_id in prompt_ids:
                            # per_prompt_avg_response_lengths uses integer keys
                            if prompt_id in per_prompt_avg_response_lengths:
                                replica_prompt_response_lengths[prompt_id] = per_prompt_avg_response_lengths[prompt_id]
                        per_replica_prompt_response_lengths[replica_id] = replica_prompt_response_lengths
                        print(f"[DEBUG] Bin packing prompt assignment: {per_replica_prompt_ids}")
                elif schedule_mode == "shuffle":
                    from verl.utils.shuffle_scheduler import get_shuffle_scheduler
                    
                    # Debug: check environment variables again
                    import os
                    print(f"[DEBUG] Checking shuffle scheduler:")
                    print(f"  VERL_ENABLE_SHUFFLE={os.getenv('VERL_ENABLE_SHUFFLE', 'not set')}")
                    print(f"  VERL_SHUFFLE_SEED={os.getenv('VERL_SHUFFLE_SEED', 'not set')}")
                    
                    shuffle_scheduler = get_shuffle_scheduler()
                    print(f"[DEBUG] Shuffle scheduler result: {shuffle_scheduler}")
                    
                    if shuffle_scheduler is not None:
                        shuffled_indices = shuffle_scheduler.get_shuffled_indices(len(response_lengths), actual_n)
                        print(f"[DEBUG] Got shuffled_indices, length={len(shuffled_indices)}, first 20: {shuffled_indices[:20]}")
                        
                        prompts_per_replica = len(response_lengths) // num_replicas
                        print(f"[DEBUG] Prompts per replica: {prompts_per_replica}, num_replicas: {num_replicas}")
                        
                        for replica_id in range(num_replicas):
                            start_pos = replica_id * prompts_per_replica
                            end_pos = (replica_id + 1) * prompts_per_replica if replica_id < num_replicas - 1 else len(shuffled_indices)
                            
                            # Get the shuffled indices for this replica's positions
                            replica_shuffled_indices = shuffled_indices[start_pos:end_pos]
                            
                            # Convert to original prompt IDs
                            # When actual_n=1, shuffled_indices[i] is directly the prompt ID
                            replica_prompt_ids = set()
                            for shuffled_idx in replica_shuffled_indices:
                                # When actual_n=1, shuffled_idx is the prompt ID
                                # When actual_n>1, we need to convert: shuffled_idx // actual_n
                                if actual_n == 1:
                                    original_prompt_id = shuffled_idx
                                else:
                                    original_prompt_id = shuffled_idx // actual_n
                                replica_prompt_ids.add(original_prompt_id)
                            
                            per_replica_prompt_ids[replica_id] = sorted(replica_prompt_ids)
                            
                            # Record response lengths for each prompt in this replica
                            replica_prompt_response_lengths = {}
                            for prompt_id in replica_prompt_ids:
                                # per_prompt_avg_response_lengths uses integer keys
                                if prompt_id in per_prompt_avg_response_lengths:
                                    replica_prompt_response_lengths[prompt_id] = per_prompt_avg_response_lengths[prompt_id]
                                else:
                                    # Debug: log missing prompt IDs
                                    print(f"[DEBUG] Prompt ID {prompt_id} not found in per_prompt_avg_response_lengths")
                            per_replica_prompt_response_lengths[replica_id] = replica_prompt_response_lengths
                        print(f"[DEBUG] Shuffle prompt assignment: {per_replica_prompt_ids}")
                        print(f"[DEBUG] Shuffle response lengths recorded: {sum(len(v) for v in per_replica_prompt_response_lengths.values())} prompts")
                    else:
                        print(f"[DEBUG] Shuffle scheduler is None, cannot record prompt assignment - falling back to default")
                        # Fallback to default sequential assignment
                        prompts_per_replica = num_original_prompts // num_replicas
                        for replica_id in range(num_replicas):
                            start_prompt = replica_id * prompts_per_replica
                            end_prompt = (replica_id + 1) * prompts_per_replica if replica_id < num_replicas - 1 else num_original_prompts
                            per_replica_prompt_ids[replica_id] = list(range(start_prompt, end_prompt))
                            
                            replica_prompt_response_lengths = {}
                            for prompt_id in range(start_prompt, end_prompt):
                                # per_prompt_avg_response_lengths uses integer keys
                                if prompt_id in per_prompt_avg_response_lengths:
                                    replica_prompt_response_lengths[prompt_id] = per_prompt_avg_response_lengths[prompt_id]
                            per_replica_prompt_response_lengths[replica_id] = replica_prompt_response_lengths
                else:
                    # Default: sequential assignment (chunk)
                    prompts_per_replica = num_original_prompts // num_replicas
                    for replica_id in range(num_replicas):
                        start_prompt = replica_id * prompts_per_replica
                        end_prompt = (replica_id + 1) * prompts_per_replica if replica_id < num_replicas - 1 else num_original_prompts
                        per_replica_prompt_ids[replica_id] = list(range(start_prompt, end_prompt))
                        
                        # Record response lengths for each prompt in this replica
                        replica_prompt_response_lengths = {}
                        for prompt_id in range(start_prompt, end_prompt):
                            # per_prompt_avg_response_lengths uses integer keys
                            if prompt_id in per_prompt_avg_response_lengths:
                                replica_prompt_response_lengths[prompt_id] = per_prompt_avg_response_lengths[prompt_id]
                        per_replica_prompt_response_lengths[replica_id] = replica_prompt_response_lengths
                    print(f"[DEBUG] Default prompt assignment (sequential chunk): {per_replica_prompt_ids}")
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to record per-replica prompt assignment: {e}")
                # Fallback: sequential assignment
                prompts_per_replica = num_original_prompts // num_replicas
                for replica_id in range(num_replicas):
                    start_prompt = replica_id * prompts_per_replica
                    end_prompt = (replica_id + 1) * prompts_per_replica if replica_id < num_replicas - 1 else num_original_prompts
                    per_replica_prompt_ids[replica_id] = list(range(start_prompt, end_prompt))
                    
                    # Record response lengths for each prompt in this replica (fallback)
                    replica_prompt_response_lengths = {}
                    for prompt_id in range(start_prompt, end_prompt):
                        # per_prompt_avg_response_lengths uses integer keys
                        if prompt_id in per_prompt_avg_response_lengths:
                            replica_prompt_response_lengths[prompt_id] = per_prompt_avg_response_lengths[prompt_id]
                    per_replica_prompt_response_lengths[replica_id] = replica_prompt_response_lengths
            
            # Store per-batch worker timing info
            per_worker_prompt_ids = {}
            if per_worker_times:
                for worker_timing in per_worker_times:
                    worker_rank = worker_timing.get("worker_rank")
                    replica_id = worker_timing.get("replica_id")
                    if worker_rank is None or replica_id is None:
                        continue
                    per_worker_prompt_ids[str(worker_rank)] = per_replica_prompt_ids.get(replica_id, [])

            # Build helper mappings for per-prompt timing attribution
            replica_times = {}
            replica_worker_ranks: dict[int, list[int]] = defaultdict(list)
            for worker_timing in per_worker_times:
                replica_id = worker_timing.get("replica_id")
                if replica_id is None:
                    continue
                worker_time = worker_timing.get("rollout_time", 0.0)
                replica_times[replica_id] = max(replica_times.get(replica_id, 0.0), worker_time)
                if "worker_rank" in worker_timing and worker_timing["worker_rank"] is not None:
                    replica_worker_ranks[replica_id].append(worker_timing["worker_rank"])
            
            prompt_to_replica: dict[int, int] = {}
            for replica_id, prompt_ids in per_replica_prompt_ids.items():
                for prompt_id in prompt_ids:
                    prompt_to_replica[prompt_id] = replica_id
            
            per_prompt_generation_stats: dict[int, dict[str, Any]] = {}
            for prompt_id, response_lengths_list in per_prompt_response_lengths.items():
                total_resp_tokens = int(sum(response_lengths_list))
                num_generations_for_prompt = len(response_lengths_list)
                avg_resp_tokens = (
                    float(total_resp_tokens / num_generations_for_prompt)
                    if num_generations_for_prompt > 0
                    else 0.0
                )
                replica_id = prompt_to_replica.get(prompt_id)
                per_prompt_generation_stats[prompt_id] = {
                    "prompt_length": per_prompt_prompt_lengths.get(prompt_id, 0),
                    "response_lengths": response_lengths_list,
                    "total_response_tokens": total_resp_tokens,
                    "avg_response_tokens": avg_resp_tokens,
                    "num_generations": num_generations_for_prompt,
                    "replica_id": replica_id,
                    "replica_time_s": replica_times.get(replica_id),
                    "worker_ranks": replica_worker_ranks.get(replica_id, []),
                    "schedule_mode": schedule_mode,
                }
            
            batch_worker_timing = {
                'batch': batch_idx,
                'avg_time': timing_avg,
                'min_time': timing_min,
                'max_time': timing_max,
                'overall_wall_clock_time': overall_wall_clock_time if 'overall_wall_clock_time' in locals() else rollout_time,  # Total time for all replicas
                'response_tokens': total_response_tokens,
                'prompt_tokens': total_prompt_tokens,
                'num_workers': num_workers,
                'num_replicas': num_replicas,
                'tp_size': tp_size,
                'all_workers': per_worker_times,  # Actual per-worker timing (from all_gather if available)
                'rollout_n': rollout_n,  # Number of generations per prompt (for training, for reference)
                'val_kwargs_n': actual_n,  # Number of generations per prompt (for validation, actually used)
                'num_original_prompts': num_original_prompts,  # Number of original prompts (before repetition, calculated using val_kwargs.n)
                'per_prompt_response_lengths': per_prompt_response_lengths,  # {prompt_id: [length1, ..., length_n]} - all n generations for each prompt
                'per_prompt_prompt_lengths': per_prompt_prompt_lengths,  # {prompt_id: prompt_length}
                'per_prompt_avg_response_lengths': per_prompt_avg_response_lengths,  # {prompt_id: avg_length} - average response length for each prompt
                'all_response_lengths': response_lengths,  # All response lengths (flat list, for backward compatibility)
                'all_prompt_lengths': prompt_lengths,  # All prompt lengths (flat list, for backward compatibility)
                'per_replica_prompt_ids': per_replica_prompt_ids,  # {replica_id: [prompt_id1, prompt_id2, ...]} - NEW: record which prompts each replica processes
                'per_replica_prompt_response_lengths': per_replica_prompt_response_lengths,  # {replica_id: {prompt_id: avg_response_length}} - NEW: record response lengths for each prompt in each replica
                'per_worker_prompt_ids': per_worker_prompt_ids,  # {worker_rank: [prompt_ids]}
                'schedule_mode': schedule_mode,  # NEW: record the schedule mode used
                'per_prompt_generation_stats': per_prompt_generation_stats,  # {prompt_id: {...}} - detailed per-prompt totals & timing attribution
            }
            self._validation_rollout_metrics['worker_times'].append(batch_worker_timing)
            self._validation_rollout_metrics['all_workers_timing'].append(per_worker_times)
            self._validation_rollout_metrics['worker_min_times'].append(timing_min)
            self._validation_rollout_metrics['worker_max_times'].append(timing_max)
            
            print(f"Rollout batch {batch_idx + 1}: "
                  f"avg_time={timing_avg:.2f}s, "
                  f"min_time={timing_min:.2f}s (fastest worker), "
                  f"max_time={timing_max:.2f}s (slowest worker), "
                  f"response_tokens={total_response_tokens}, "
                  f"num_workers={num_workers}, "
                  f"num_replicas={num_replicas} (TP={tp_size})")
            
            # Print overall wall clock time (all replicas combined)
            print(f"  Overall wall clock time (all replicas combined): {overall_wall_clock_time:.2f}s")
            
            # Print per-worker timing if available (for reference)
            if per_worker_times and len(per_worker_times) <= 16:  # Only print if not too many
                print(f"  Per-worker timing (estimated, for reference):")
                for worker_timing in per_worker_times[:8]:  # Show first 8 only
                    replica_id = worker_timing.get('replica_id', 'unknown')
                    print(f"    Worker {worker_timing.get('worker_rank', 'unknown')} (Replica {replica_id}): "
                          f"time={worker_timing.get('rollout_time', 0.0):.2f}s, "
                          f"tokens={worker_timing.get('response_tokens', 0)}")
                if len(per_worker_times) > 8:
                    print(f"    ... (showing first 8 workers, {len(per_worker_times)} total)")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()


        # Add rollout timing and token metrics
        if hasattr(self, '_validation_rollout_metrics') and self._validation_rollout_metrics['num_batches'] > 0:
            metrics = self._validation_rollout_metrics
            
            # Get TP size from metrics
            tp_size = metrics.get('tp_size', 1)
            
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
            
            # Calculate per-worker averages and find fastest/slowest from actual worker times
            per_worker_summary = []
            all_worker_times = []  # Collect all individual worker times for fastest/slowest calculation
            
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
                # Collect individual worker times from each batch
                all_worker_times.append(avg_time)
            
            # Calculate fastest/slowest from actual per-worker times (not from batch min/max)
            if all_worker_times:
                fastest_worker_time = min(all_worker_times)
                slowest_worker_time = max(all_worker_times)
                # Calculate average of fastest/slowest across batches (for reference)
                if metrics['worker_min_times']:
                    avg_fastest_time = sum(metrics['worker_min_times']) / len(metrics['worker_min_times'])
                    avg_slowest_time = sum(metrics['worker_max_times']) / len(metrics['worker_max_times'])
                else:
                    avg_fastest_time = fastest_worker_time
                    avg_slowest_time = slowest_worker_time
            else:
                # Fallback to batch min/max if no worker times collected
                if metrics['worker_min_times']:
                    fastest_worker_time = min(metrics['worker_min_times'])
                    slowest_worker_time = max(metrics['worker_max_times'])
                    avg_fastest_time = sum(metrics['worker_min_times']) / len(metrics['worker_min_times'])
                    avg_slowest_time = sum(metrics['worker_max_times']) / len(metrics['worker_max_times'])
                else:
                    fastest_worker_time = slowest_worker_time = avg_fastest_time = avg_slowest_time = 0.0
            
            # Note: We don't track per-replica statistics because we cannot directly measure
            # each replica's individual completion time. The overall_wall_clock_time (total_rollout_time)
            # represents the total time from start to finish for all replicas executing in parallel.
            
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
                'rollout/tp_size': tp_size,
                'rollout/num_replicas': metrics.get('num_replicas', 0),  # Number of replicas (for reference)
            })
            
            # Collect all per-prompt response lengths across all batches (for scheduling algorithm)
            # Group by original prompt ID, collecting all n generations for each prompt
            all_per_prompt_response_lengths_dict = {}  # {prompt_id: [all n generations across batches]}
            all_per_prompt_prompt_lengths_dict = {}  # {prompt_id: prompt_length}
            all_per_prompt_avg_response_lengths_dict = {}  # {prompt_id: avg_length}
            all_per_prompt_generation_summary = {}  # {prompt_id: detailed stats incl. timing}
            all_response_lengths_flat = []  # Flat list for statistics
            all_prompt_lengths_flat = []  # Flat list for statistics
            
            # Track global prompt ID across batches
            global_prompt_id = 0
            for batch_metrics in metrics['worker_times']:
                batch_per_prompt_response_lengths = batch_metrics.get('per_prompt_response_lengths', {})
                batch_per_prompt_prompt_lengths = batch_metrics.get('per_prompt_prompt_lengths', {})
                batch_per_prompt_avg_response_lengths = batch_metrics.get('per_prompt_avg_response_lengths', {})
                batch_per_prompt_generation_stats = batch_metrics.get('per_prompt_generation_stats', {})
                
                # Process each prompt in this batch
                for local_prompt_id in sorted(batch_per_prompt_response_lengths.keys()):
                    prompt_response_lengths = batch_per_prompt_response_lengths[local_prompt_id]
                    prompt_length = batch_per_prompt_prompt_lengths.get(local_prompt_id, 0)
                    prompt_avg_length = batch_per_prompt_avg_response_lengths.get(local_prompt_id, 0)
                    
                    # Use global prompt ID to track across batches
                    all_per_prompt_response_lengths_dict[global_prompt_id] = prompt_response_lengths
                    all_per_prompt_prompt_lengths_dict[global_prompt_id] = prompt_length
                    all_per_prompt_avg_response_lengths_dict[global_prompt_id] = prompt_avg_length
                    
                    per_prompt_stats = batch_per_prompt_generation_stats.get(local_prompt_id, {})
                    all_per_prompt_generation_summary[global_prompt_id] = {
                        'global_prompt_id': global_prompt_id,
                        'batch_idx': batch_metrics.get('batch'),
                        'local_prompt_id': local_prompt_id,
                        'prompt_length': per_prompt_stats.get('prompt_length', prompt_length),
                        'response_lengths': per_prompt_stats.get('response_lengths', prompt_response_lengths),
                        'total_response_tokens': int(
                            per_prompt_stats.get('total_response_tokens', sum(prompt_response_lengths))
                        ),
                        'avg_response_tokens': float(
                            per_prompt_stats.get('avg_response_tokens', prompt_avg_length)
                        ),
                        'num_generations': per_prompt_stats.get('num_generations', len(prompt_response_lengths)),
                        'replica_id': per_prompt_stats.get('replica_id'),
                        'replica_time_s': per_prompt_stats.get('replica_time_s'),
                        'worker_ranks': per_prompt_stats.get('worker_ranks', []),
                        'schedule_mode': per_prompt_stats.get('schedule_mode', batch_metrics.get('schedule_mode')),
                    }
                    
                    # Add to flat lists for statistics
                    all_response_lengths_flat.extend(prompt_response_lengths)
                    all_prompt_lengths_flat.append(prompt_length)
                    
                    global_prompt_id += 1
            
            # Calculate statistics for response lengths (important for scheduling)
            # Note: numpy is already imported at the top of the file (line 30), no need to import again
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
            
            # Get schedule mode from environment variable for log file naming
            schedule_mode = os.getenv("VERL_SCHEDULE_MODE", "default").lower()
            # Normalize schedule mode name for filename (replace underscore with hyphen)
            schedule_mode_normalized = schedule_mode.replace("_", "-")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Include schedule mode in filename
            worker_metrics_file = os.path.join(log_dir, f"rollout_worker_metrics_{schedule_mode_normalized}_{timestamp}.json")
            with open(worker_metrics_file, 'w') as f:
                json.dump({
                    'summary': {
                        'total_rollout_time_s': metrics['total_rollout_time'],
                        'total_response_tokens': metrics['total_response_tokens'],
                        'total_prompt_tokens': metrics['total_prompt_tokens'],
                        'num_batches': metrics['num_batches'],
                        'num_workers': len(per_worker_summary),
                        'num_replicas': metrics.get('num_replicas', 0),  # Number of replicas (for reference)
                        'tp_size': tp_size,
                        'fastest_worker_time_s': fastest_worker_time,
                        'slowest_worker_time_s': slowest_worker_time,
                        'avg_fastest_worker_time_s': avg_fastest_time,
                        'avg_slowest_worker_time_s': avg_slowest_time,
                        'worker_time_difference_s': slowest_worker_time - fastest_worker_time,
                        'response_length_stats': response_length_stats,  # Statistics for all individual response lengths
                        'avg_response_length_stats': avg_response_length_stats,  # Statistics for per-prompt average response lengths (key for scheduling)
                        'prompt_length_stats': prompt_length_stats,  # Statistics for prompt lengths
                    },
                    'per_worker_summary': per_worker_summary,  # Per-worker timing (actual measurements)
                    'per_batch_metrics': metrics['worker_times'],  # Includes per_prompt_response_lengths (grouped by prompt) and per_prompt_avg_response_lengths
                    'all_per_prompt_response_lengths': all_per_prompt_response_lengths_dict,  # {prompt_id: [all n generations]} - grouped by prompt ID
                    'all_per_prompt_avg_response_lengths': all_per_prompt_avg_response_lengths_dict,  # {prompt_id: avg_length} - average response length for each prompt (key for scheduling)
                    'all_per_prompt_prompt_lengths': all_per_prompt_prompt_lengths_dict,  # {prompt_id: prompt_length}
                    'all_per_prompt_generation_summary': all_per_prompt_generation_summary,  # Detailed per-prompt stats (tokens + replica timing)
                }, f, indent=2)
            print(f"\nSaved metrics to: {worker_metrics_file}")
            print(f"  - Overall wall clock time (all replicas combined)")
            print(f"  - Per-worker metrics (actual measurements)")
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
            
            print(f"\n=== Rollout Metrics Summary ===")
            print(f"Total rollout time: {metrics['total_rollout_time']:.2f}s")
            print(f"Total response tokens: {metrics['total_response_tokens']}")
            print(f"Total prompt tokens: {metrics['total_prompt_tokens']}")
            print(f"Number of batches: {metrics['num_batches']}")
            print(f"Throughput: {metrics['total_response_tokens'] / (metrics['total_rollout_time'] + 1e-9):.2f} tokens/sec")
            print(f"TP size: {tp_size}, Number of replicas: {metrics.get('num_replicas', 0)}")
            print(f"\n--- Overall Timing (All Replicas Combined) ---")
            print(f"Total wall clock time: {metrics['total_rollout_time']:.2f}s")
            print(f"Average time per batch: {metrics['total_rollout_time'] / metrics['num_batches']:.2f}s")
            print(f"\n--- Per-Worker Timing (Actual Measurements) ---")
            print(f"Fastest worker (min): {fastest_worker_time:.2f}s")
            print(f"Slowest worker (max): {slowest_worker_time:.2f}s")
            print(f"Avg fastest worker: {avg_fastest_time:.2f}s")
            print(f"Avg slowest worker: {avg_slowest_time:.2f}s")
            print(f"Worker time difference (max - min): {slowest_worker_time - fastest_worker_time:.2f}s")
            print(f"===============================\n")
            # Reset for next validation
            delattr(self, '_validation_rollout_metrics')
        
        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _apply_scheduling(self, gen_batch: DataProto) -> DataProto:
        """
        应用调度策略：根据配置选择bin packing、shuffle或默认顺序
        
        Args:
            gen_batch: 未repeat的DataProto（原始prompts）
        
        Returns:
            调度后的DataProto（仍然是原始prompts，未repeat）
        """
        # DEBUG_BREAKPOINT_RayPPOTrainer._apply_scheduling
        import os
        if os.getenv("VERL_DEBUG", "0") == "1":
            import ipdb
            print("\n======================================================================")
            print(f"🐛 DEBUG BREAKPOINT: RayPPOTrainer._apply_scheduling")
            print(f"======================================================================\n")
            ipdb.set_trace()
        import os
        
        # 检查调度模式
        schedule_mode = os.getenv("VERL_SCHEDULE_MODE", "default").lower()
        
        if schedule_mode == "bin_packing":
            return self._apply_bin_packing_schedule(gen_batch)
        elif schedule_mode == "shuffle":
            return self._apply_shuffle_schedule(gen_batch)
        else:
            # 默认：不进行调度，保持原始顺序
            return gen_batch
    
    def _apply_shuffle_schedule(self, gen_batch: DataProto) -> DataProto:
        from verl.utils.shuffle_scheduler import get_shuffle_scheduler
        scheduler = get_shuffle_scheduler()
        if scheduler is None:
            return gen_batch
        shuffled_indices = scheduler.get_shuffled_prompt_indices(len(gen_batch))
        shuffled_data = gen_batch.select_idxs(shuffled_indices)
        shuffled_data.meta_info["shuffle_scheduled"] = True
        if scheduler.seed is not None:
            shuffled_data.meta_info["shuffle_seed"] = scheduler.seed
        return shuffled_data

    def _apply_bin_packing_schedule(self, gen_batch: DataProto) -> DataProto:
        """
        应用Bin Packing调度：根据avg response length重新排列原始prompts，使得每个replica的负载均衡
        
        Args:
            gen_batch: 未repeat的DataProto（原始prompts）
        
        Returns:
            重新排列后的DataProto（仍然是原始prompts，未repeat）
        """
        try:
            from verl.utils.bin_packing_scheduler import get_bin_packing_scheduler
            
            scheduler = get_bin_packing_scheduler()
            if scheduler is None:
                # 未配置bin packing调度，返回原数据
                return gen_batch
            
            # 获取原始prompts数量
            num_prompts = len(gen_batch)
            
            # 获取TP size和replicas数量
            try:
                if not self.async_rollout_mode:
                    num_workers = self.actor_rollout_wg.world_size
                else:
                    num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers if hasattr(self.config.actor_rollout_ref.rollout.agent, 'num_workers') else 1
                
                tp_size = getattr(self.config.actor_rollout_ref.rollout, 'tensor_model_parallel_size', 1)
                num_replicas = num_workers // tp_size if tp_size > 0 else num_workers
            except:
                # 如果获取失败，返回原数据
                return gen_batch
            
            # 验证replicas数量
            expected_replicas = scheduler.schedule.get('num_replicas', num_replicas)
            if expected_replicas != num_replicas:
                import warnings
                warnings.warn(
                    f"Bin packing schedule expects {expected_replicas} replicas, but current setup has {num_replicas} replicas. "
                    f"Skipping bin packing scheduling."
                )
                return gen_batch
            
            # 创建重新排列的索引（只对原始prompts进行重新排列）
            reorder_indices = []
            
            # 按replica顺序收集prompts
            for replica_id in range(num_replicas):
                prompt_ids = scheduler.get_prompts_for_replica(replica_id)
                for prompt_id in prompt_ids:
                    if prompt_id < num_prompts:
                        # 直接使用prompt_id作为索引（因为这是原始prompts，未repeat）
                        reorder_indices.append(prompt_id)
            
            # 验证索引数量
            if len(reorder_indices) != num_prompts:
                import warnings
                warnings.warn(
                    f"Bin packing reorder indices count ({len(reorder_indices)}) != num prompts ({num_prompts}). "
                    f"Skipping bin packing scheduling."
                )
                return gen_batch
            
            # 重新排列数据（只对原始prompts重新排列）
            reordered_data = gen_batch.select_idxs(reorder_indices)
            
            # 在meta_info中记录调度信息
            reordered_data.meta_info["bin_packing_scheduled"] = True
            
            print(f"[Bin Packing] Applied scheduling: {num_prompts} prompts -> {num_replicas} replicas")
            
            return reordered_data
            
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to apply bin packing schedule: {e}. Using original data order.")
            return gen_batch

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def compute_rollout_importance_weights_and_add_to_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Compute IS weights and apply rejection sampling for rollout-training mismatch.

        Computes importance sampling weights to correct for distribution mismatch between
        rollout and training policies. Applies rejection sampling (mask mode/veto) by
        modifying response_mask. Always updates response_mask; conditionally adds IS weights.

        Key behavior:
        - response_mask: ALWAYS updated with rejection (mask mode + veto excluded from training)
        - rollout_is_weights: Added to batch ONLY if config.algorithm.rollout_is=True

        This separation ensures:
        - Rejection works even when IS weights are disabled (rollout_is=False)
        - Metrics can be monitored before enabling IS weight application

        Args:
            batch: DataProto with old_log_probs, rollout_log_probs, response_mask

        Returns:
            Tuple of (updated_batch, metrics):
                updated_batch: Batch with modified response_mask (always) and rollout_is_weights (if rollout_is=True)
                metrics: Dict of IS and mismatch metrics, all with "mismatch/" prefix
        """
        # Compute rollout IS weights if enabled and data is available
        # rollout_is_threshold is the main on/off switch (None = disabled, float = enabled)
        rollout_is_threshold = self.config.algorithm.get("rollout_is_threshold", None)
        if rollout_is_threshold is not None and rollout_is_threshold > 0 and "rollout_log_probs" in batch.batch:
            # Compute IS weights and get modified response_mask
            rollout_is_weights, modified_response_mask, rollout_is_metrics = compute_rollout_importance_weights(
                old_log_prob=batch.batch["old_log_probs"],
                rollout_log_prob=batch.batch["rollout_log_probs"],
                response_mask=batch.batch["response_mask"],
                rollout_is_level=self.config.algorithm.rollout_is_level,
                rollout_is_mode=self.config.algorithm.rollout_is_mode,
                rollout_is_threshold=self.config.algorithm.rollout_is_threshold,
                rollout_is_threshold_lower=self.config.algorithm.get("rollout_is_threshold_lower", None),
                rollout_is_veto_threshold=self.config.algorithm.get("rollout_is_veto_threshold", None),
            )

            # ALWAYS update response_mask with rejection (even if rollout_is=False)
            # - Mask mode: tokens with outlier IS ratios excluded
            # - Veto: sequences with catastrophic tokens excluded
            # This ensures correct loss normalization (rejected samples not in denominator)
            batch.batch["response_mask"] = modified_response_mask

            # Conditionally add IS weights based on rollout_is config flag
            # - rollout_is=True: Enable IS weight correction in policy loss
            # - rollout_is=False: Metrics-only mode (rejection still applied via mask)
            apply_weights = self.config.algorithm.get("rollout_is", False)

            if apply_weights:
                # Add IS weights (safety-bounded, mode-processed) to enable weight correction
                batch = batch.union(rollout_is_weights)

            return batch, rollout_is_metrics

        # Return unchanged batch and empty metrics if IS is disabled
        return batch, {}

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        # DEBUG_BREAKPOINT_RayPPOTrainer.fit
        import os
        if os.getenv("VERL_DEBUG", "0") == "1":
            import ipdb
            print("\n======================================================================")
            print(f"🐛 DEBUG BREAKPOINT: RayPPOTrainer.fit")
            print(f"======================================================================\n")
            ipdb.set_trace()
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = self._apply_scheduling(gen_batch)
                # DEBUG_BREAKPOINT_fit_after_scheduling
                import os
                if os.getenv("VERL_DEBUG", "0") == "1":
                    import ipdb
                    print("\n======================================================================")
                    print(f"🐛 DEBUG BREAKPOINT: fit_after_scheduling")
                    print(f"======================================================================\n")
                    ipdb.set_trace()
                
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
