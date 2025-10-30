"""
Schedules for Clinical-Enriched Drug Discovery pipeline.
"""

from dagster import (
    AssetSelection,
    ScheduleDefinition,
    define_asset_job,
)


# Job 1: Daily clinical extraction (DISABLED)
# clinical_extraction_job = define_asset_job(
#     name="clinical_extraction_job",
#     selection=AssetSelection.groups("clinical_extraction"),
#     description="Extract drug-disease pairs from clinical notes using NER",
# )

# Schedule: Run clinical extraction daily at 8 AM (DISABLED)
# daily_clinical_extraction = ScheduleDefinition(
#     name="daily_clinical_extraction",
#     job=clinical_extraction_job,
#     cron_schedule="0 8 * * *",  # Every day at 8:00 AM
#     description="Daily extraction of clinical drug-disease pairs",
# )


# Job 2: Weekly full data refresh
weekly_data_refresh_job = define_asset_job(
    name="weekly_data_refresh",
    selection=AssetSelection.groups("data_loading"),
    description="Weekly refresh of PrimeKG data from Harvard Dataverse",
)

# Schedule: Run data loading weekly on Sundays at 2 AM
weekly_data_refresh = ScheduleDefinition(
    name="weekly_data_refresh",
    job=weekly_data_refresh_job,
    cron_schedule="0 2 * * 0",  # Every Sunday at 2:00 AM
    description="Weekly refresh of PrimeKG knowledge graph data",
)


# Job 3: Complete pipeline run
complete_pipeline_job = define_asset_job(
    name="complete_pipeline",
    selection=AssetSelection.all(),
    description="Run the complete clinical drug discovery pipeline from start to finish",
)

# Schedule: Run complete pipeline monthly on 1st at midnight
monthly_pipeline_run = ScheduleDefinition(
    name="monthly_pipeline_run",
    job=complete_pipeline_job,
    cron_schedule="0 0 1 * *",  # 1st of every month at midnight
    description="Monthly complete pipeline execution",
)


# Example: Hourly schedule (commented out - uncomment to use)
# hourly_clinical_check = ScheduleDefinition(
#     name="hourly_clinical_check",
#     job=clinical_extraction_job,
#     cron_schedule="0 * * * *",  # Every hour
#     description="Hourly clinical data extraction",
# )
