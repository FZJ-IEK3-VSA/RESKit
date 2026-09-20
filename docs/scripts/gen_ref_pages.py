"""Generate the API reference pages automatically.

This script is run by mkdocs-gen-files during the build process.
It discovers all Python modules in the reskit package and generates
corresponding Markdown files with mkdocstrings directives.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Directories to skip.
# "_test" is the internal test-fixture subpackage (its `data/*/__init__.py`
# files exist only to make the sample datasets importable as package data);
# "data" holds packaged data files, not public API.
SKIP_DIRS = {"__pycache__", "data", "_test"}

# Individual modules to skip. "testen.py" is an empty file at the package root
# (no code, no docstring); it would render as a blank reference page.
SKIP_MODULES = {"reskit/testen.py"}

for path in sorted(Path("reskit").rglob("*.py")):
    if path.as_posix() in SKIP_MODULES:
        continue

    module_path = path.with_suffix("")
    doc_path = path.relative_to("reskit").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    # Skip __pycache__, data directories, and other non-module files
    if any(skip in parts for skip in SKIP_DIRS):
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    # Skip the root package index (reskit/__init__.py → just "reskit")
    if len(parts) == 0:
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write(f"::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.as_posix())

# Write the navigation file for literate-nav
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
