# Copyright 2026 Inworld AI
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
"""Weights and configuration stored together in release safetensors files."""

import torch

from inworld.sava import config
from inworld.sava import model as model_lib
from inworld.sava import registry

FORMAT = "sava"
_FORMAT_KEY = "format"
_CONFIG_KEY = "config"


def read(path: str) -> tuple[config.SavaConfig, dict[str, torch.Tensor]]:
    """The configuration and the state dict stored in a release file.

    Raises:
        ValueError: If the file is not a SAVA release checkpoint.
    """
    # Checkpoint dependencies are loaded only when a checkpoint is opened.
    # pylint: disable-next=import-outside-toplevel
    import safetensors

    with safetensors.safe_open(path, framework="pt") as f:
        metadata = f.metadata() or {}
        if metadata.get(_FORMAT_KEY) != FORMAT:
            raise ValueError(
                f"[{path}] is not a SAVA checkpoint (metadata "
                f"{_FORMAT_KEY!r} is {metadata.get(_FORMAT_KEY)!r}).")
        if _CONFIG_KEY not in metadata:
            raise ValueError(
                f"[{path}] SAVA checkpoint metadata lacks the 'config' entry.")
        state = {key: f.get_tensor(key) for key in f.keys()}
    return config.from_json(metadata[_CONFIG_KEY]), state


def write(path: str, model: model_lib.SavaModel) -> None:
    """Writes the model's weights and configuration as a release file."""
    # pylint: disable-next=import-outside-toplevel
    import safetensors.torch

    state = {
        k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()
    }
    safetensors.torch.save_file(state,
                                path,
                                metadata={
                                    _FORMAT_KEY: FORMAT,
                                    _CONFIG_KEY: config.to_json(model.cfg)
                                })


def build(cfg: config.SavaConfig,
          state: dict[str, torch.Tensor],
          device: str | torch.device = "cpu",
          dtype: torch.dtype = torch.float32) -> model_lib.SavaModel:
    """A model with ``state`` loaded strictly, on ``device`` in ``dtype``."""
    with torch.device("meta"):
        model = model_lib.SavaModel(cfg)
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device=device, dtype=dtype).eval()


def load(source: str,
         device: str | torch.device = "cpu",
         dtype: torch.dtype = torch.float32) -> model_lib.SavaModel:
    """Loads a released model in evaluation mode.

    ``source`` is a registered Hub repository (``inworld/sava``), a file or
    ``hf://org/repo/file[@revision]``.
    """
    cfg, state = read(registry.resolve(source))
    return build(cfg, state, device, dtype)
