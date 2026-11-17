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
"""Released weights and resolution of checkpoint sources to local files."""

import dataclasses
import hashlib
import os

HF_SCHEME = "hf://"


@dataclasses.dataclass(frozen=True)
class Release:
    """A Hub checkpoint, optionally pinned to a revision and checksum."""

    repo_id: str
    filename: str
    sha256: str | None  # None until the weights are published.
    revision: str | None = None


MODELS = {
    "inworld/sava":
        Release(
            repo_id="inworld/sava",
            filename="sava.safetensors",
            sha256=
            "aa610ee9b5d3d1be96f70581a8a3affe7cf20d06816b4ada1efdd8ab6274be34",
            revision="1beae7c1dd0df6b0f92030644fc905eb087dc6a0"),
}


def sha256_file(path: str) -> str:
    """Returns a file's SHA-256, reading in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(release: Release) -> str:
    # Loading local checkpoints must not import the Hub client.
    # pylint: disable-next=import-outside-toplevel
    import huggingface_hub
    # pylint: disable-next=import-outside-toplevel
    import huggingface_hub.utils

    try:
        path = huggingface_hub.hf_hub_download(repo_id=release.repo_id,
                                               filename=release.filename,
                                               revision=release.revision)
    except (OSError, huggingface_hub.utils.HfHubHTTPError) as error:
        raise RuntimeError(
            f"Could not fetch {release.repo_id}/{release.filename} from the "
            "Hub. Check the location and network access, or load a local "
            "checkpoint by its path instead.") from error
    if release.sha256 is not None:
        digest = sha256_file(path)
        if digest != release.sha256:
            raise ValueError(
                f"[{path}] does not match the registered checksum of "
                f"{release.repo_id}/{release.filename}: expected "
                f"{release.sha256}, got {digest}.")
    return path


def parse_hf(source: str) -> Release:
    """Parses ``hf://org/repo/file[@revision]`` without a checksum."""
    spec = source[len(HF_SCHEME):]
    spec, separator, revision = spec.partition("@")
    parts = spec.split("/")
    if (not source.startswith(HF_SCHEME) or len(parts) < 3 or not all(parts) or
        (separator and not revision)):
        raise ValueError(f"[{source}] is not hf://org/repo/file[@revision].")
    return Release(repo_id="/".join(parts[:2]),
                   filename="/".join(parts[2:]),
                   sha256=None,
                   revision=revision or None)


def resolve(source: str) -> str:
    """Resolves a registered model, an ``hf://`` location or a local file.

    A registered model is named by its Hub repository, ``inworld/sava``, and
    always downloads and verifies the release. Use an explicit path such as
    ``./inworld/sava`` for a local file at the same relative path.
    Unregistered Hub locations have no checksum; pin their revision instead.
    """
    if source in MODELS:
        release = MODELS[source]
        if not release.sha256:
            raise ValueError(
                f"'{source}' is not released yet; load a local checkpoint "
                "by its path instead.")
        return _download(release)
    if source.startswith(HF_SCHEME):
        return _download(parse_hf(source))
    if os.path.isfile(source):
        return source
    raise FileNotFoundError(f"[{source}] is neither a file, a registered model "
                            f"{sorted(MODELS)} nor an hf:// location.")
