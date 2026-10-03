# Third-party notices

`iae/third_party/teb/` contains the subset of the TEB helper package used by this code: the deterministic plan compiler
(`nc_plan_compiler.py`), the request schema and I/O helpers (`schema.py`, `io.py`, `budget.py`, `prompts.py`), the
WatchAct request importer (`watchact.py`) and the local VLM backend (`backends.py`). It is distributed under the MIT
license in `iae/third_party/teb/LICENSE`.

The WatchAct dataset, simulator and scorer are not redistributed. Download the dataset from its official release and
clone the official code at the commit given in the README.
