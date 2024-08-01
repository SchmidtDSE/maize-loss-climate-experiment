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

    def get_prereq(self):
        return preprocess_climate_tasks.PreprocessClimateGeotiffsTask(
            dataset_name='sample',
            conditions=['observations'],
            years=const.YEARS[0:2]
        )

    def get_task_name(self):
        return 'end_sample_climate_preprocess'


class FullClimatePreprocessTask(cluster_tasks.EndClusterTask):

    def get_prereq(self):
        return preprocess_climate_tasks.PreprocessClimateGeotiffsTask(
            dataset_name='historic',
            conditions=['observations'],
            years=const.YEARS
        )

    def get_task_name(self):
        return 'end_full_climate_preprocess'


class RunThroughPreprocessTask(cluster_tasks.EndClusterTask):

    def get_prereq(self):
        return preprocess_combine_tasks.CombineHistoricPreprocessTask()

    def get_task_name(self):
        return 'end_preprocess'


class RunThroughPreprocessFutureTask(cluster_tasks.EndClusterTask):

    def get_prereq(self):
        return {
            '2030_SSP245': preprocess_combine_tasks.ReformatFuturePreprocessTask(condition='2030_SSP245'),
            '2050_SSP245': preprocess_combine_tasks.ReformatFuturePreprocessTask(condition='2050_SSP245')
        }

    def get_task_name(self):
        return 'end_preprocess_future'


class RunThroughSweepTask(cluster_tasks.EndClusterTask):

    def get_prereq(self):
        return training_tasks.SweepTask()

    def get_task_name(self):
        return 'end_sweep'


class RunThroughExtendedSweepTask(cluster_tasks.EndClusterTask):

    def get_prereq(self):
        return training_tasks.SweepExtendedTask()

    def get_task_name(self):
        return 'end_sweep_ext'


class RunThroughHistoricProject(cluster_tasks.EndClusterTask):

    def get_prereq(self):
        return sim_tasks.ProjectHistoricTask()

    def get_task_name(self):
        return 'end_project_historic'


class RunThroughSimTask(cluster_tasks.EndClusterTask):

    def get_prereq(self):
        return sim_tasks.CombineSimulationsTasks()

    def get_task_name(self):
        return 'end_sim'


class RunThroughHistTask(cluster_tasks.EndClusterTask):

    def get_prereq(self):
        return export_tasks.HistExportTask()

    def get_task_name(self):
        return 'end_hist'


class ExecuteSupplementalTasks(luigi.Task):

    def requires(self):
        return {
            'text': stats_tasks.CombineStatsTask(),
            'climate': export_tasks.ClimateExportTask(),
            'sweep': export_tasks.SweepExportTask(),
            'hist': export_tasks.HistExportTask(),
            'summary': export_tasks.SummaryExportTask(),
            'combined': export_tasks.CombinedTasksRecordTask(),
            'rates': export_tasks.ExportClaimsRatesTask()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location('break_supplemental.txt'))

    def run(self):
        with self.output().open('w') as f:
            f.write('done')


class ExecuteSupplementalTasksWithCluster(cluster_tasks.EndClusterTask):

    def get_prereq(self):
        return ExecuteSupplementalTasks()

    def get_task_name(self):
        return 'end_supplemental'
