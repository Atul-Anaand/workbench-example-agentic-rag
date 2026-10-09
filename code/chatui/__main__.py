# V2.1: Preload CUDA libraries for faster-whisper on GPU
# ctranslate2 needs libcublas.so.12 and libcudnn.so.9 from the nvidia-*-cu12 wheels
# which aren't on the default search path. We inject them and preload by absolute path.
import os as _os
import ctypes as _ctypes
_CUDA_LIB_DIRS = [
    "/home/workbench/.local/lib/python3.10/site-packages/nvidia/cublas/lib",
    "/home/workbench/.local/lib/python3.10/site-packages/nvidia/cudnn/lib",
]
_existing_ld = _os.environ.get("LD_LIBRARY_PATH", "")
_os.environ["LD_LIBRARY_PATH"] = ":".join(
    [d for d in _CUDA_LIB_DIRS if _os.path.isdir(d)] + ([_existing_ld] if _existing_ld else [])
)
for _d in _CUDA_LIB_DIRS:
    for _name in ("libcublas.so.12", "libcudnn.so.9"):
        _p = _os.path.join(_d, _name)
        if _os.path.exists(_p):
            try:
                _ctypes.CDLL(_p, mode=_ctypes.RTLD_GLOBAL)
            except Exception:
                pass
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entrypoint for the Conversation GUI.

The functions in this module are responsible for bootstrapping then executing the Conversation GUI server.
"""
import gradio_client.utils as _gcu
import gradio_client

_orig_to_py = _gcu._json_schema_to_python_type

def _patched_to_py(schema, defs=None):
    # Guard against boolean schemas (e.g. additionalProperties: true)
    if isinstance(schema, bool):
        return "Any"
    return _orig_to_py(schema, defs)

_gcu._json_schema_to_python_type = _patched_to_py

import argparse
import os
import sys

import uvicorn


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the program.

    :returns: A namespace containing the parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Document Retrieval Service")

    parser.add_argument(
        "--help-config",
        action="store_true",
        default=False,
        help="show the configuration help text",
    )

    parser.add_argument(
        "-c",
        "--config",
        metavar="CONFIGURATION_FILE",
        default="/dev/null",
        help="path to the configuration file (json or yaml)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="increase output verbosity",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="decrease output verbosity",
    )

    parser.add_argument(
        "--host",
        metavar="HOSTNAME",
        type=str,
        default="0.0.0.0",  # nosec # this is intentional
        help="Bind socket to this host.",
    )
    parser.add_argument(
        "--port",
        metavar="PORT_NUM",
        type=int,
        default=8080,
        help="Bind socket to this port.",
    )
    parser.add_argument(
        "--workers",
        metavar="NUM_WORKERS",
        type=int,
        default=1,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--ssl-keyfile", metavar="SSL_KEY", type=str, default=None, help="SSL key file"
    )
    parser.add_argument(
        "--ssl-certfile",
        metavar="SSL_CERT",
        type=str,
        default=None,
        help="SSL certificate file",
    )

    cliargs = parser.parse_args()
    if cliargs.help_config:
        # pylint: disable=import-outside-toplevel; this is intentional to allow for the environment to be configured
        #                                          before any of the application libraries are loaded.
        from chatui.configuration import AppConfig

        sys.stdout.write("\nconfiguration file format:\n")
        AppConfig.print_help(sys.stdout.write)
        sys.exit(0)

    return cliargs


if __name__ == "__main__":
    args = parse_args()
    os.environ["APP_VERBOSITY"] = f"{args.verbose - args.quiet}"
    os.environ["APP_CONFIG_FILE"] = args.config

    from chatui import api, chat_client, configuration, pages

    # load config
    config_file = os.environ.get("APP_CONFIG_FILE", "/dev/null")
    config = configuration.AppConfig.from_file(config_file)
    if not config:
        sys.exit(1)

    # connect to other services
    api_url = f"{config.server_url}:{config.server_port}"
    print(api_url)
    client = chat_client.ChatClient(
       api_url, config.model_name
    )
    proxy_prefix = os.environ.get("PROXY_PREFIX")
    blocks = pages.converse.build_page(client)
    blocks.queue(max_size=10)
    blocks.launch(server_name="0.0.0.0", server_port=8080, root_path=proxy_prefix)