# pylint: disable=undefined-variable
import os

import lit.formats
from lit.llvm import llvm_config

config.name = "MQT MLIR test suite"
config.test_format = lit.formats.ShTest(True)

# Define the file extensions to treat as test files.
config.suffixes = [".mlir"]
config.excludes = ["lit.cfg.py"]

# Define the root path of where to look for tests.
config.test_source_root = os.path.dirname(__file__)

# Define where to execute tests (and produce the output).
config.test_exec_root = getattr(config, "quantum_test_dir", ".lit")

# Define PATH to include the various tools needed for our tests.
try:
    # From within a build target we have access to cmake variables configured in lit.site.cfg.py.in.
    llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)  # FileCheck
    llvm_config.with_environment("PATH", config.quantum_bin_dir, append_path=True)  # quantum-opt
except AttributeError as e:
    # The system PATH is available by default.
    pass
