#!/bin/bash
# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

docker run --rm -it -v $(pwd)/..:/home/mqt/mqt-mlir mqt-catalyst
