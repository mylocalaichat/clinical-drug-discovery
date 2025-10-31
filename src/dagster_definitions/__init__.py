"""
Dagster definitions for Clinical-Enriched Drug Discovery.

This module defines all Dagster assets, jobs, and resources.
"""

from dagster import (
    Definitions,
    load_assets_from_package_module,
    multiprocess_executor,
)

from . import assets

# Load all assets from the assets package
all_assets = load_assets_from_package_module(assets)

# Define Dagster configuration - manual execution only (no automation)
# All schedules, sensors, and jobs have been removed
# Assets can be materialized manually from the UI
defs = Definitions(
    assets=all_assets,
    # Configure executor to run assets sequentially (no parallel execution)
    # This prevents resource exhaustion from multiple embedding training jobs
    executor=multiprocess_executor.configured({"max_concurrent": 1}),
)
