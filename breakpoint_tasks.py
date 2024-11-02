"""Entrypoint tasks which complete either a segement of work or the full pipeline.

Entrypoint tasks which complete either a segement of work or the full pipeline in which
ExecuteSupplementalTasksWithCluster executes the entire cluster with outputs for tools. Many of
these tasks will both start a cluster in Coiled before spinning it down after finishing work.

License: BSD
"""

import luigi

import cluster_tasks
import const
import export_tasks
import preprocess_climate_tasks
import preprocess_combine_tasks
import sim_tasks
import stats_tasks
import training_tasks


class SampleClimatePreprocessTask(cluster_tasks.EndClusterTask):
    """Task which completes through gneerating a subset of climate summaries.

    Task which completes through gneerating a small sample of climate summaries as preprocessed
    geotiffs that go through geohashing as a testing step. This will end the cluster on completion.
    """

    def get_prereq(self):
        """Get the tasks that need to be completed.

        Returns:
            Single task (PreprocessClimateGeotiffsTask) that is required to reach this breakpoint.
        """
        return preprocess_climate_tasks.PreprocessClimateGeotiffsTask(
            dataset_name='sample',
            conditions=['observations'],
            years=const.YEARS[0:2]
        )

    def get_task_name(self):
        """Get a machine friendly name for this task.

        Returns:
            Machine-friendly name.
        """
        return 'end_sample_climate_preprocess'


class FullClimatePreprocessTask(cluster_tasks.EndClusterTask):
    """Task which completes through gneerating a subset of climate summaries.

    Task which completes through gneerating a small sample of climate summaries as preprocessed
    geotiffs that go through geohashing as a testing step. This will end the cluster on completion.
    """

    def get_prereq(self):
        """Get the tasks that need to be completed.

        Returns:
            Single task (PreprocessClimateGeotiffsTask) that is required to reach this breakpoint.
        """
        return preprocess_climate_tasks.PreprocessClimateGeotiffsTask(
            dataset_name='historic',
            conditions=['observations'],
            years=const.YEARS
        )

    def get_task_name(self):
        """Get a machine friendly name for this task.

        Returns:
            Machine-friendly name.
        """
        return 'end_full_climate_preprocess'


class RunThroughPreprocessTask(cluster_tasks.EndClusterTask):
    """Run all preprocessing tasks on historic data.

    Breakpoint which runs up until training (preprocessing tasks). This will end the cluster on
    completion. Note that this only includes historic acutals not future values predicted by third
    party models.
    """

    def get_prereq(self):
        """Get the tasks that need to be completed.

        Returns:
            Single task (CombineHistoricPreprocessTask) that is required to reach this breakpoint.
        """
        return preprocess_combine_tasks.CombineHistoricPreprocessTask()

    def get_task_name(self):
        """Get a machine friendly name for this task.

        Returns:
            Machine-friendly name.
        """
        return 'end_preprocess'


class RunThroughPreprocessFutureTask(cluster_tasks.EndClusterTask):
    """Run all preprocessing tasks on future data.

    Breakpoint which runs up until training (preprocessing tasks). This will end the cluster on
    completion. Note that this only includes future values predicted by third party models and not
    historic acutals.
    """

    def get_prereq(self):
        """Get the tasks that need to be completed.

        Returns:
            Tasks for 2030_SSP245 and 2050_SSP245.
        """
        return {
            '2030_SSP245': preprocess_combine_tasks.ReformatFuturePreprocessTask(
                condition='2030_SSP245'
            ),
            '2050_SSP245': preprocess_combine_tasks.ReformatFuturePreprocessTask(
                condition='2050_SSP245'
            )
        }

    def get_task_name(self):
        """Get a machine friendly name for this task.

        Returns:
            Machine-friendly name.
        """
        return 'end_preprocess_future'


class RunThroughSweepTask(cluster_tasks.EndClusterTask):
    """Run through a model sweep with all dependent tasks.

    Run through a model sweep with all dependent tasks, closing the cluster afterwards upon
    completion.
    """

    def get_prereq(self):
        """Get the tasks that need to be completed.

        Returns:
            Single task (SweepTask).
        """
        return training_tasks.SweepTask()

    def get_task_name(self):
        """Get a machine friendly name for this task.

        Returns:
            Machine-friendly name.
        """
        return 'end_sweep'


class RunThroughExtendedSweepTask(cluster_tasks.EndClusterTask):
    """Execute extended sweep.

    Execute extended sweep for the set of meta-parameter combinations less likely to be successful
    before terminating the cluster.
    """

    def get_prereq(self):
        """Get the dependent task.

        Returns:
            Single dependent task (SweepExtendedTask).
        """
        return training_tasks.SweepExtendedTask()

    def get_task_name(self):
        """Get a machine friendly name for this task.

        Returns:
            Machine-friendly name.
        """
        return 'end_sweep_ext'


class RunThroughHistoricProject(cluster_tasks.EndClusterTask):
    """Task which executes through backwards projection.

    Breakpoint which preprocesses, trains models, and makes projections backwards into historic
    data. This is largely useful for testing prior to making future predictions.  Will terminate
    cluster.
    """

    def get_prereq(self):
        """Get the dependent task.

        Returns:
            Single dependent task (ProjectHistoricTask).
        """
        return sim_tasks.ProjectHistoricTask()

    def get_task_name(self):
        """Get a machine friendly name for this task.

        Returns:
            Machine-friendly name.
        """
        return 'end_project_historic'


class RunThroughSimTask(cluster_tasks.EndClusterTask):
    """Task which runs through simulation of future outcomes.

    Breakpoint task which completes the data pipeline through Monte Carlo (includes preprocessing,
    model training, etc). This is largely useful for running the pipeline for raw predictions prior
    to generation of outputs for tools and visualizations. Will terminate cluster.
    """

    def get_prereq(self):
        """Get the dependent task.

        Returns:
            Single dependent task (CombineSimulationsTask).
        """
        return sim_tasks.CombineSimulationsTask()

    def get_task_name(self):
        """Get a machine friendly name for this task.

        Returns:
            Machine-friendly name.
        """
        return 'end_sim'


class RunThroughHistTask(cluster_tasks.EndClusterTask):
    """Run the pipeline to the point of generating future prediction system-wide histograms.

    Run the pipeline through model training and simulation before summarizing results as system-wide
    histograms which can be used in tooling. Will terminate cluster. Unlike
    ExecuteSupplementalTasks, this covers only one tool.
    """

    def get_prereq(self):
        """Get the dependent task.

        Returns:
            Single dependent task (HistExportTask).
        """
        return export_tasks.HistExportTask()

    def get_task_name(self):
        """Get a machine friendly name for this task.

        Returns:
            Machine-friendly name.
        """
        return 'end_hist'


class ExecuteSupplementalTasks(luigi.Task):
    """Execute the pipeline through supplemental tasks.

    Execute the pipeline and report on supplemental data outputs which can be used in the paper and'
    tools. Will not terminate the cluster and this task is largely useful for debugging the pipeline
    in its entirety.
    """

    def requires(self):
        return {
            'stats': stats_tasks.CombineStatsTask(),
            'climate': export_tasks.ClimateExportTask(),
            'sweep': export_tasks.SweepExportTask(),
            'hist': export_tasks.HistExportTask(),
            'summary': export_tasks.SummaryExportTask(),
            'combined': export_tasks.CombinedTasksRecordTask(),
            'rates': export_tasks.ExportClaimsRatesTask(),
            'ratesHold': export_tasks.ExportClaimsRatesHoldYearTask()
        }

    def output(self):
        """Indicate where a status report file should be written.

        Returns:
            Local target for the status file.
        """
        return luigi.LocalTarget(const.get_file_location('break_supplemental.txt'))

    def run(self):
        """Execute the pipeline through the supplemental tasks."""
        with self.output().open('w') as f:
            f.write('done')


class ExecuteSignificantLongTask(luigi.Task):
    """Execute the pipeline to determine the frequency of significant results with long geohashes.

    Execute the pipeline to determine the frequency of significant results with long geohashes
    (5 character). This is largely useful for debugging or answering follow up questions from the
    pipeline. Does not terminate cluster.
    """

    def requires(self):
        """Get the task that needs to be completed for this breakpoint to be reached.

        Returns:
            Single dependency (DeterminePercentSignificantLongTask).
        """
        return stats_tasks.DeterminePercentSignificantLongTask()

    def output(self):
        """Indicate where a status report file should be written.

        Returns:
            Local target for the status file.
        """
        return luigi.LocalTarget(const.get_file_location('long_sig.txt'))

    def run(self):
        """Execute the pipeline through the significant long geohash task."""
        with self.output().open('w') as f:
            f.write('done')


class ExecuteSupplementalTasksWithCluster(cluster_tasks.EndClusterTask):
    """Execute the entire data pipeline before terminating the cluster."""

    def get_prereq(self):
        """Get the task that needs to be completed for this breakpoint to be reached.

        Returns:
            Single dependency (ExecuteSupplementalTasks).
        """
        return ExecuteSupplementalTasks()

    def get_task_name(self):
        """Get a machine friendly name for this task.

        Returns:
            Machine-friendly name.
        """
        return 'end_supplemental'


class ExecuteAllWithCluster(cluster_tasks.EndClusterTask):
    """Execute the entire data pipeline with extended sweep before terminating the cluster."""

    def get_prereq(self):
        """Get the task that needs to be completed for this breakpoint to be reached.

        Returns:
            Multiple dependency.
        """
        return [ExecuteSupplementalTasks(), training_tasks.SweepExtendedTask()]

    def get_task_name(self):
        """Get a machine friendly name for this task.

        Returns:
            Machine-friendly name.
        """
        return 'end_supplemental_all'
