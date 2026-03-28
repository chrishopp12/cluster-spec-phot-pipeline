#!/usr/bin/env python3
"""run_xray_pipeline.py — COMPATIBILITY SHIM (v2 migration)

The X-ray pipeline driver has moved to cluster_pipeline.pipelines.xray.
This file re-exports it so `python run_xray_pipeline.py` still works.
Remove this file in Phase 5 once all usage is via the CLI.
"""
from cluster_pipeline.pipelines.xray import *  # noqa: F401, F403

if __name__ == "__main__":
    from cluster_pipeline.pipelines.xray import main
    main()
