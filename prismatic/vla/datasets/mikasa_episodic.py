"""Episode-preserving MIKASA-Robo RLDS input for persistent-memory training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset

from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


def _version_dir(path: Path) -> Path:
    versions = sorted(p for p in path.iterdir() if p.is_dir())
    if not versions:
        raise FileNotFoundError(f"No TFDS version directory below {path}")
    return versions[-1]


def _dataset(path: Path, episode_limit: int | None = None):
    dataset = tfds.builder_from_directory(str(_version_dir(path))).as_dataset(split="train")
    return dataset.take(episode_limit) if episode_limit is not None else dataset


def _stats_path(root: Path, env_names: list[str], episode_limit: int) -> Path:
    key = f"{episode_limit}:" + ",".join(sorted(env_names))
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return root / f"mikasa_stats_{digest}.json"


def _compute_or_load_stats(root: Path, env_names: list[str], episode_limit: int) -> dict:
    cache = _stats_path(root, env_names, episode_limit)
    if cache.exists():
        return json.loads(cache.read_text())

    actions, proprio = [], []
    trajectories = 0
    for env_name in env_names:
        for episode in _dataset(root / env_name, episode_limit):
            steps = list(episode["steps"])
            if not steps:
                continue
            actions.append(np.stack([s["action"].numpy() for s in steps]))
            proprio.append(np.stack([s["observation"]["proprio"].numpy() for s in steps]))
            trajectories += 1

    def describe(values):
        values = np.concatenate(values, axis=0)
        return {
            "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(),
            "min": values.min(0).tolist(),
            "max": values.max(0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }

    result = {
        "action": describe(actions),
        "proprio": describe(proprio),
        "num_transitions": int(sum(len(x) for x in actions)),
        "num_trajectories": trajectories,
    }
    cache.write_text(json.dumps(result, indent=2))
    return result


def _normalize(value: np.ndarray, stats: dict) -> np.ndarray:
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    return np.clip(2 * (value - low) / (high - low + 1e-8) - 1, -1, 1).astype(np.float32)


def _episode_to_numpy(episode):
    steps = list(episode["steps"])
    if not steps:
        return None
    language = steps[0]["language_instruction"].numpy()
    if isinstance(language, str):
        language = language.encode()
    return {
        "image": np.stack([s["observation"]["image"].numpy() for s in steps]),
        "wrist": np.stack([s["observation"]["wrist_image"].numpy() for s in steps]),
        "proprio": np.stack([s["observation"]["proprio"].numpy() for s in steps]),
        "action": np.stack([s["action"].numpy() for s in steps]),
        "language": language,
    }


class MIKASAEpisodicDataset(IterableDataset):
    """Round-robin episode streams keep batch slot i temporally coherent."""

    def __init__(self, root, env_names, batch_transform, batch_size=1, seed=42, episodes_per_env=200):
        self.root = Path(root)
        self.env_names = list(env_names)
        self.batch_transform = batch_transform
        self.batch_size = batch_size
        self.seed = seed
        self.episodes_per_env = episodes_per_env
        self.stats = _compute_or_load_stats(self.root, self.env_names, episodes_per_env)
        self.dataset_statistics = {"mikasa_combined": self.stats}

    def _env_iterator(self, env_name, seed):
        dataset = _dataset(self.root / env_name, self.episodes_per_env)
        return iter(dataset.shuffle(200, seed=seed).repeat())

    def _stream(self, rng, iterators):
        while True:
            env_name = str(rng.choice(self.env_names))
            episode = _episode_to_numpy(next(iterators[env_name]))
            if episode is None:
                continue
            episode["action"] = _normalize(episode["action"], self.stats["action"])
            episode["proprio"] = _normalize(episode["proprio"], self.stats["proprio"])
            length = len(episode["action"])
            for t in range(length):
                indices = np.minimum(np.arange(t, t + NUM_ACTIONS_CHUNK), length - 1)
                raw = {
                    "dataset_name": f"mikasa_{env_name}",
                    "action": episode["action"][indices],
                    "observation": {
                        "image_primary": episode["image"][t : t + 1],
                        "image_wrist": episode["wrist"][t : t + 1],
                        "proprio": episode["proprio"][t : t + 1],
                    },
                    "task": {"language_instruction": episode["language"]},
                }
                item = self.batch_transform(raw)
                item["is_first"] = t == 0
                item["is_last"] = t == length - 1
                yield item

    def __iter__(self):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        master = np.random.default_rng(self.seed + rank)
        # TFDS iterators are expensive: sharing one iterator per environment keeps
        # loader resources O(num_envs), rather than O(batch_size * num_envs).
        # Iteration is single-threaded here, so each stream still receives a
        # distinct complete episode without concurrent access to an iterator.
        iterators = {
            name: self._env_iterator(name, int(master.integers(2**31))) for name in self.env_names
        }
        streams = [
            self._stream(np.random.default_rng(master.integers(2**63)), iterators)
            for _ in range(self.batch_size)
        ]
        while True:
            for stream in streams:
                yield next(stream)

    def __len__(self):
        return int(self.stats["num_transitions"])


class MIKASAEpisodicCollator(PaddedCollatorForActionPrediction):
    def __call__(self, instances):
        output = super().__call__(instances)
        output["is_first"] = torch.tensor([x["is_first"] for x in instances], dtype=torch.bool)
        output["is_last"] = torch.tensor([x["is_last"] for x in instances], dtype=torch.bool)
        return output
