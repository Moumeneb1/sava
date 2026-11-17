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
"""Public exports and package import boundaries."""

import os
import subprocess
import sys
import textwrap

from inworld import sava


def test_the_public_api_is_what_all_lists():
    for name in sava.__all__:
        assert hasattr(sava, name), name
    assert sava.load is sava.checkpoint.load
    assert sava.run is sava.streaming.run


def test_importing_the_package_pulls_in_no_optional_dependency():
    code = textwrap.dedent("""
        import socket
        import sys
        from unittest import mock

        with mock.patch.object(socket.socket, 'connect', autospec=True,
                               side_effect=AssertionError('network on import')):
            from inworld import sava

        forbidden = {'sounddevice', 'transformers', 'onnxruntime', 'sklearn',
                     'huggingface_hub', 'safetensors'}
        loaded = forbidden.intersection(sys.modules)
        assert not loaded, loaded
    """)

    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True,
                         text=True,
                         check=True,
                         env={
                             **os.environ, "PYTHONPATH":
                                 os.pathsep.join(sys.path)
                         })

    assert not out.stdout.strip()
