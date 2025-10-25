"""
Dagster definitions for Clinical-Enriched Drug Discovery.

This module defines all Dagster assets, jobs, and resources.
"""

from dagster import (
    Definitions,
    load_assets_from_package_module,
)

from . import assets
from .schedules import (
    clinical_extraction_job,
    complete_pipeline_job,
    daily_clinical_extraction,
    monthly_pipeline_run,
    weekly_data_refresh,
    weekly_data_refresh_job,
)
from .sensors import (
    new_clinical_data_sensor,
    primekg_update_sensor,
)

# Load all assets from the assets package
all_assets = load_assets_from_package_module(assets)

# Define Dagster configuration with schedules, sensors, and jobs
defs = Definitions(
    assets=all_assets,
    schedules=[
        daily_clinical_extraction,
        weekly_data_refresh,
        monthly_pipeline_run,
    ],
    sensors=[
        new_clinical_data_sensor,
        primekg_update_sensor,
    ],
    jobs=[
        clinical_extraction_job,
        weekly_data_refresh_job,
        complete_pipeline_job,
    ],
)
