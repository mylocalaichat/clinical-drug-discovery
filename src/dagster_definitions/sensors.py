"""
Sensors for Clinical-Enriched Drug Discovery pipeline.

Sensors trigger jobs based on events or conditions rather than time schedules.
"""

from dagster import (
    AssetSelection,
    RunRequest,
    SensorEvaluationContext,
    sensor,
)
from pathlib import Path


@sensor(
    name="new_clinical_data_sensor",
    target=AssetSelection.groups("clinical_extraction"),
    minimum_interval_seconds=300,  # Check every 5 minutes
)
def new_clinical_data_sensor(context: SensorEvaluationContext):
    """
    Trigger clinical extraction when new clinical notes file is detected.

    Checks for a 'trigger' file in the data directory. When found, it triggers
    the clinical extraction pipeline and removes the trigger file.
    """
    trigger_file = Path("data/01_raw/clinical_notes_trigger.txt")

    if trigger_file.exists():
        context.log.info(f"New clinical data detected! Trigger file: {trigger_file}")

        # Read any metadata from the trigger file (optional)
        try:
            metadata = trigger_file.read_text().strip()
            context.log.info(f"Trigger metadata: {metadata}")
        except Exception as e:
            context.log.warning(f"Could not read trigger file metadata: {e}")
            metadata = "unknown"

        # Remove trigger file to avoid duplicate runs
        trigger_file.unlink()

        # Trigger the run
        yield RunRequest(
            run_key=f"clinical_data_{context.cursor or 0}",
            tags={
                "source": "new_clinical_data_sensor",
                "trigger_metadata": metadata,
            }
        )

        # Update cursor to track this run
        context.update_cursor(str(int(context.cursor or 0) + 1))
    else:
        context.log.debug("No new clinical data detected")


@sensor(
    name="primekg_update_sensor",
    target=AssetSelection.groups("data_loading"),
    minimum_interval_seconds=3600,  # Check every hour
)
def primekg_update_sensor(context: SensorEvaluationContext):
    """
    Trigger PrimeKG data refresh when update flag is set.

    Example usage:
        touch data/01_raw/primekg/update_trigger.flag
    """
    trigger_file = Path("data/01_raw/primekg/update_trigger.flag")

    if trigger_file.exists():
        context.log.info("PrimeKG update triggered!")

        # Remove trigger file
        trigger_file.unlink()

        # Trigger the data loading pipeline
        yield RunRequest(
            run_key=f"primekg_update_{context.cursor or 0}",
            tags={"source": "primekg_update_sensor"}
        )

        context.update_cursor(str(int(context.cursor or 0) + 1))


# Example: More advanced sensor with file monitoring
# Uncomment to use with watchdog or similar file monitoring library

# @sensor(
#     name="data_folder_monitor",
#     target=AssetSelection.groups("data_loading"),
#     minimum_interval_seconds=60,
# )
# def data_folder_monitor(context: SensorEvaluationContext):
#     """
#     Monitor a folder for new .csv files and trigger processing.
#     """
#     data_folder = Path("data/01_raw/incoming")
#
#     # Get list of CSV files
#     csv_files = list(data_folder.glob("*.csv"))
#
#     # Compare with last known state (stored in cursor)
#     last_count = int(context.cursor or 0)
#     current_count = len(csv_files)
#
#     if current_count > last_count:
#         context.log.info(f"New files detected! {current_count - last_count} new CSV files")
#
#         yield RunRequest(
#             run_key=f"new_files_{current_count}",
#             tags={
#                 "file_count": str(current_count),
#                 "new_files": str(current_count - last_count)
#             }
#         )
#
#         context.update_cursor(str(current_count))
