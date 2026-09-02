"""Closed-loop MIKASA evaluation for RD-VLA persistent memory checkpoints.

This runner follows the benchmark's canonical seeds, wrapper stack,
``success_once`` latch, and action-chunk FIFO.  Unlike the benchmark's generic
policy protocol, it explicitly resets persistent state at every episode.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.robot.mikasa_robo.memory_interventions import MemoryIntervention


PILOT_TASKS = [
    "RememberColor9-VLA-v0",
    "RememberColor9-Long-VLA-v0",
    "ChainOfColors5-VLA-v0",
    "ChainOfColors5-Long-VLA-v0",
    "ShellGameShuffleTouch-VLA-v0",
    "ShellGameShuffleTouch-Long-VLA-v0",
    "BlinkCountButtonPressHard-VLA-v0",
    "BlinkCountButtonPressHard-Long-VLA-v0",
    "TimedTransferHard-VLA-v0",
    "TimedTransferHard-Long-VLA-v0",
]


def _require_canonical_tasks(tasks):
    """Reject silent ``custom`` fallbacks from the benchmark task selector."""
    unknown = [task.env_id for task in tasks if task.split == "custom"]
    if unknown:
        raise SystemExit(
            "Unknown MIKASA task ID(s); refusing custom fallback: "
            + ", ".join(unknown)
        )
    return tasks


def _check_vulkan_render_device() -> None:
    """Fail before model loading when SAPIEN cannot see a Vulkan GPU.

    ManiSkill's SAPIEN build still creates a Vulkan render system for the
    ``sim_backend=cpu`` path. On headless hosts with only llvmpipe this can
    otherwise fail (or segfault) after the 2.5 GB checkpoint has loaded.
    ``vulkaninfo`` is optional; when unavailable, SAPIEN provides the final
    diagnostic.
    """
    vulkaninfo = shutil.which("vulkaninfo")
    if vulkaninfo is None:
        return
    try:
        result = subprocess.run(
            [vulkaninfo, "--summary"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    report = f"{result.stdout}\n{result.stderr}"
    device_types = re.findall(
        r"deviceType\s*=\s*PHYSICAL_DEVICE_TYPE_([A-Z_]+)", report
    )
    gpu_types = {kind for kind in device_types if kind.endswith("_GPU")}
    if gpu_types:
        return
    if "llvmpipe" in report.lower() or "vkcreateinstance" in report.lower():
        raise SystemExit(
            "MIKASA closed-loop evaluation requires a Vulkan GPU render device. "
            "vulkaninfo found no discrete/integrated GPU (only llvmpipe or a "
            "broken ICD). Install/fix the NVIDIA Vulkan ICD, then rerun; the "
            "CPU simulator backend does not remove this SAPIEN render requirement."
        )


def _scalar(value, default=0):
    if value is None:
        return default
    if torch.is_tensor(value):
        return value.detach().reshape(-1)[0].cpu().item()
    array = np.asarray(value)
    return array.reshape(-1)[0].item() if array.size else default


class RDVLAMemoryPolicy:
    chunk_size = 8

    def __init__(self, checkpoint: Path, instruction: str, k: int, memory: str):
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
            get_vla,
        )

        self.cfg = SimpleNamespace(
            pretrained_checkpoint=str(checkpoint),
            load_in_8bit=False,
            load_in_4bit=False,
            use_film=False,
            num_images_in_input=2,
            use_proprio=True,
            unnorm_key="mikasa_combined",
            center_crop=False,
            num_open_loop_steps=self.chunk_size,
            recurrence_strategy="fixed",
            recurrent_num_iter=int(k),
            recurrence_kl_thresh=0.001,
            recurrence_cos_thresh=0.999,
            recurrence_max_iter=32,
        )
        self.vla = get_vla(self.cfg)
        self.processor = get_processor(self.cfg)
        self.action_head = get_action_head(self.cfg, self.vla.llm_dim)
        self.proprio_projector = get_proprio_projector(self.cfg, self.vla.llm_dim, proprio_dim=7)
        self.instruction = instruction
        self.memory_mode = memory
        self.intervention = MemoryIntervention(mode=memory)
        self.memory_state = None
        self.previous_action = None

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction
        self.memory_state = None
        self.previous_action = None
        self.intervention = MemoryIntervention(mode=self.memory_mode)

    def reset_episode(self) -> None:
        self.intervention.reset_episode(self.memory_state)
        self.memory_state = None
        self.previous_action = None

    @torch.inference_mode()
    def forward(self, obs):
        from experiments.robot.openvla_utils import get_vla_action

        rgb = obs["rgb"]
        proprio = obs["proprio"]
        if torch.is_tensor(rgb):
            rgb = rgb.detach().cpu().numpy()
        if torch.is_tensor(proprio):
            proprio = proprio.detach().cpu().numpy()
        rgb = np.asarray(rgb)[0]
        proprio = np.asarray(proprio)[0]

        memory_input = self.intervention.apply(self.memory_state)
        disable_memory = self.memory_mode == "reset"
        observation = {
            "full_image": np.ascontiguousarray(rgb[..., :3]),
            "wrist_image": np.ascontiguousarray(rgb[..., 3:6]),
            "state": proprio.copy(),
        }
        actions, _, _, new_memory = get_vla_action(
            cfg=self.cfg,
            vla=self.vla,
            processor=self.processor,
            obs=observation,
            task_label=self.instruction,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            memory_state=memory_input,
            previous_action=self.previous_action,
            disable_memory=disable_memory,
            return_memory=True,
        )
        chunk = np.asarray(actions, dtype=np.float32)
        self.memory_state = None if disable_memory else new_memory.detach()
        self.previous_action = torch.as_tensor(
            chunk[-1], device=next(self.action_head.parameters()).device, dtype=torch.bfloat16
        ).unsqueeze(0)
        return np.clip(chunk, -1.0, 1.0)


def run_episode(env, policy, seed: int):
    obs, _ = env.reset(seed=seed)
    policy.reset_episode()
    queue = deque()
    success_once = False
    episode_return = 0.0
    max_steps = int(env.max_episode_steps)
    for step in range(max_steps):
        if not queue:
            queue.extend(torch.as_tensor(policy.forward(obs)))
        action = queue.popleft().to(device=env.unwrapped.device, dtype=torch.float32).unsqueeze(0)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(_scalar(reward, 0.0))
        success_once = success_once or bool(_scalar(info.get("success"), False))
        if bool(_scalar(terminated, False)) or bool(_scalar(truncated, False)):
            return success_once, episode_return, step + 1
    return success_once, episode_return, max_steps


def _make_eval_env(task, config, sim_backend: str):
    """Create an env with a render backend matching the requested simulator.

    MIKASA's helper forwards ``sim_backend`` but ManiSkill defaults the render
    backend independently to CUDA.  That makes the documented CPU fallback
    fail on hosts with no NVIDIA Vulkan render ICD.
    """
    if sim_backend != "cpu":
        from mikasa_robo_suite.vla.benchmarking import make_benchmark_env
        return make_benchmark_env(task.env_id, config)

    import gymnasium as gym
    from mikasa_robo_suite.vla.benchmarking import apply_mikasa_vla_wrappers

    render_mode = "rgb_array" if config.save_videos else "all"
    env = gym.make(
        task.env_id,
        num_envs=1,
        obs_mode=config.obs_mode,
        control_mode=config.control_mode,
        render_mode=render_mode,
        reward_mode=config.reward_mode,
        sim_backend="cpu",
        render_backend="cpu",
    )
    return apply_mikasa_vla_wrappers(env, include_overlays=config.include_overlays or config.save_videos)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mikasa-root", type=Path, default=Path("/root/MIKASA-Robo"))
    parser.add_argument("--output", type=Path, required=True, help="Append-only episode JSONL")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=4242424242)
    parser.add_argument("--k", type=int, choices=(1, 2, 4, 8, 12, 16), required=True)
    parser.add_argument("--memory", choices=("correct", "reset", "shuffle", "stale"), required=True)
    parser.add_argument("--sim-backend", choices=("cpu", "gpu"), default="gpu")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.mikasa_root))
    try:
        from mikasa_robo_suite.vla.benchmarking import (
            BenchmarkConfig,
            make_benchmark_env,
            select_benchmark_tasks,
        )
    except ImportError as exc:
        raise SystemExit(
            "MIKASA dependencies are missing. From /root/MIKASA-Robo run `uv sync --frozen`, "
            "then invoke this script with that environment."
        ) from exc

    tasks = _require_canonical_tasks(select_benchmark_tasks(
        env_ids=args.tasks or PILOT_TASKS,
        csv_path=args.mikasa_root / "mikasa_robo_vla_envs.csv",
    ))
    _check_vulkan_render_device()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add((row["task"], int(row["episode_seed"]), row["memory"], int(row["k"])))

    policy = None
    with args.output.open("a", encoding="utf-8") as output_file:
        for task in tasks:
            if policy is None:
                policy = RDVLAMemoryPolicy(args.checkpoint, task.language_instruction, args.k, args.memory)
            else:
                policy.set_instruction(task.language_instruction)
            config = BenchmarkConfig(sim_backend=args.sim_backend)
            env = _make_eval_env(task, config, args.sim_backend)
            try:
                for episode in range(args.episodes):
                    seed = args.start_seed + episode
                    key = (task.env_id, seed, args.memory, args.k)
                    if key in completed:
                        continue
                    success, episode_return, length = run_episode(env, policy, seed)
                    row = {
                        "task": task.env_id,
                        "horizon": task.split,
                        "memory_type": task.memory_type,
                        "episode_seed": seed,
                        "memory": args.memory,
                        "k": args.k,
                        "success": bool(success),
                        "return": float(episode_return),
                        "length": int(length),
                    }
                    output_file.write(json.dumps(row) + "\n")
                    output_file.flush()
                    print(json.dumps(row), flush=True)
            finally:
                env.close()


if __name__ == "__main__":
    main()
