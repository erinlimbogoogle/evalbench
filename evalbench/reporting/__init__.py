from .csv import CsvReporter
from .bqstore import BigQueryReporter
from .report import Reporter
from .gcs_artifact import GcsReporter


def get_reporters(reporting_config, job_id, run_time) -> list[Reporter]:
    reporters: list[Reporter] = []
    if not reporting_config:
        reporting_config = {}

    # Always ensure CsvReporter is enabled and runs first so local viewers have run data immediately
    csv_config = reporting_config.get("csv", {})
    reporters.append(CsvReporter(csv_config, job_id, run_time))

    if "bigquery" in reporting_config:
        reporters.append(
            BigQueryReporter(reporting_config["bigquery"], job_id, run_time)
        )
    if "gcs_artifacts" in reporting_config:
        reporters.append(GcsReporter(
            reporting_config["gcs_artifacts"], job_id, run_time))

    return reporters
