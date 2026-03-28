#!/usr/bin/env python3
"""cluster.py — COMPATIBILITY SHIM (v2 migration)

The Cluster class has moved to cluster_pipeline.models.cluster.
This file re-exports it so existing v1 code continues to work.
Remove this file in Phase 5 once all imports are updated.
"""
from cluster_pipeline.models.cluster import Cluster  # noqa: F401
