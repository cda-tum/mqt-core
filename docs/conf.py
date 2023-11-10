"""Sphinx configuration file."""

from __future__ import annotations

import warnings
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import pybtex.plugin
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.template import field, href

if TYPE_CHECKING:
    from pybtex.database import Entry
    from pybtex.richtext import HRef
    from sphinx.application import Sphinx

ROOT = Path(__file__).parent.parent.resolve()


try:
    from mqt.core import __version__ as version
except ModuleNotFoundError:
    try:
        version = metadata.version("mqt.core")
    except ModuleNotFoundError:
        msg = (
            "Package should be installed to produce documentation! "
            "Assuming a modern git archive was used for version discovery."
        )
        warnings.warn(msg, stacklevel=1)

        from setuptools_scm import get_version

        version = get_version(root=str(ROOT), fallback_root=ROOT)

# Filter git details from version
release = version.split("+")[0]

project = "MQT Core"
author = "Chair for Design Automation, Technical University of Munich"
language = "en"
project_copyright = "2023, Chair for Design Automation, Technical University of Munich"

master_doc = "index"

templates_path = ["_templates"]
html_css_files = ["custom.css"]

extensions = [
    "myst_parser",
    "nbsphinx",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxext.opengraph",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

source_suffix = [".rst", ".md"]

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

pygments_style = "colorful"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "qiskit": ("https://qiskit.org/documentation/", None),
    "mqt": ("https://mqt.readthedocs.io/en/latest/", None),
    "ddsim": ("https://mqt.readthedocs.io/projects/ddsim/en/latest/", None),
    "qmap": ("https://mqt.readthedocs.io/projects/qmap/en/latest/", None),
    "qcec": ("https://mqt.readthedocs.io/projects/qcec/en/latest/", None),
    "qecc": ("https://mqt.readthedocs.io/projects/qecc/en/latest/", None),
    "syrec": ("https://mqt.readthedocs.io/projects/syrec/en/latest/", None),
}

myst_enable_extensions = [
    "colon_fence",
    "substitution",
    "deflist",
]

myst_substitutions = {
    "version": version,
}

nbsphinx_execute = "auto"
highlight_language = "python3"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=200",
]
nbsphinx_kernel_name = "python3"

autosectionlabel_prefix_document = True


class CDAStyle(UnsrtStyle):
    """Custom style for including PDF links."""

    def format_url(self, _e: Entry) -> HRef:  # noqa: PLR6301
        """Format URL field as a link to the PDF."""
        url = field("url", raw=True)
        return href()[url, "[PDF]"]


pybtex.plugin.register_plugin("pybtex.style.formatting", "cda_style", CDAStyle)

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "cda_style"

copybutton_prompt_text = r"(?:\(venv\) )?(?:\[.*\] )?\$ "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"


modindex_common_prefix = ["mqt.core."]

autoapi_dirs = ["../src/mqt"]
autoapi_python_use_implicit_namespaces = True
autoapi_root = "api"
autoapi_add_toctree_entry = False
autoapi_ignore = [
    "*/**/_version.py",
]
autoapi_options = [
    "members",
    "inherited-members",
    "imported-members",
    "show-inheritance",
    "special-members",
    "undoc-members",
]


def skip_cpp_core(_app: Sphinx, what: str, name: str, _obj: object, skip: bool, _options) -> bool:  # noqa: ANN001
    """Skip the _core module in documentation."""
    if what == "package" and "_core" in name or "_compat" in name:
        skip = True
    return skip


def setup(sphinx: Sphinx) -> None:
    """Setup Sphinx."""
    sphinx.connect("autoapi-skip-member", skip_cpp_core)


autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_type_aliases = {
    "QuantumCircuit": "qiskit.circuit.QuantumCircuit",
}

typehints_use_signature = True
typehints_use_signature_return = True

add_module_names = False

napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "light_logo": "mqt_dark.png",
    "dark_logo": "mqt_light.png",
    "source_repository": "https://github.com/cda-tum/mqt-core/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
}
