"""Tasks to generate statistics for the manuscript.

Tasks to generate statistics for the manuscript where "main" results refer to 4 character geohashes
but simulations of 5 character geohashes may also be provided in extension or "long" simulations.
In this case, these are approximated results where sample sizes are manipulated to simulate 5
character geohashes but the actual 5 character geohashes are not used.

License:
    BSD
"""
import csv
import json
import statistics

import luigi
import toolz.itertoolz

import const
import export_tasks
import selection_tasks
import sim_tasks

NEURONS_PER_LAYER = [512, 256, 128, 64, 32, 8]


def format_percent(target):
    """Format a number as a precent.

    Args:
        target: Number to format.

    Returns:
        Formatted string.
    """
    return '%.1f\\%%' % (target * 100)


def format_severity(target):
    """Format a loss severity.

    Args:
        target: Loss where 0.15 is 15%.

    Returns:
        Formatted string.
    """
    return format_percent(target * -1 - 0.25)


class ExportModelInfoTask(luigi.Task):
    """Task to export information about the chosen model."""

    def requires(self):
        """Require that the sweep have concluded and the preferred model selected.

        Returns:
            SelectConfigurationTask
        """
        return selection_tasks.SelectConfigurationTask()

    def output(self):
        """Get the location where the selected model information should be written.

        Returns:
            LocalTarget where JSON should be written.
        """
        return luigi.LocalTarget(const.get_file_location('stats_model.json'))

    def run(self):
        """Gather model information and output in expected format."""
        with self.input().open() as f:
            source = json.load(f)

        chosen = source['constrained']

        if chosen['allowCount']:
            count_info_str = 'included'
        else:
            count_info_str = 'excluded'

        num_layers = chosen['layers']
        num_layers_int = int(num_layers)
        neurons = NEURONS_PER_LAYER[-num_layers_int:]
        neurons_strs = map(lambda x: '%d neurons' % x, neurons)
        neurons_str = ', '.join(neurons_strs)

        output_record = {
            'numLayers': '%d' % num_layers,
            'layersDescription': neurons_str,
            'dropout': chosen['dropout'],
            'l2': chosen['l2Reg'],
            'countInfoStr': count_info_str,
            'trainMeanMae': format_percent(chosen['trainMean']),
            'trainStdMae': format_percent(chosen['trainStd']),
            'validationMeanMae': format_percent(chosen['validMean']),
            'validationStdMae': format_percent(chosen['validStd']),
            'testMeanMae': format_percent(chosen['testMean']),
            'testStdMae': format_percent(chosen['testStd'])
        }

        with self.output().open('w') as f:
            json.dump(output_record, f)


class ExportPosthocTestTask(luigi.Task):
    """Export information about posthoc tests after selecting the perferred model."""

    def requires(self):
        """Require that the posthoc tests have been completed.

        Returns:
            PostHocTestRawDataRetrainCountTask, PostHocTestRawDataTemporalCountTask,
            PostHocTestRawDataRandomCountTask, and PostHocTestRawDataSpatialCountTask.
        """
        return {
            'retrain': selection_tasks.PostHocTestRawDataRetrainCountTask(),
            'temporal': selection_tasks.PostHocTestRawDataTemporalCountTask(),
            'random': selection_tasks.PostHocTestRawDataRandomCountTask(),
            'spatial': selection_tasks.PostHocTestRawDataSpatialCountTask()
        }

    def output(self):
        """Indicate where posthoc test information should be written.

        Returns:
            LocalTarget where the combined post-hoc results should be written.
        """
        return luigi.LocalTarget(const.get_file_location('stats_posthoc.json'))

    def run(self):
        """Assemble post-hoc test information."""
        temporal_record = self._summarize_post_hoc('temporal')
        random_record = self._summarize_post_hoc('random')
        spatial_record = self._summarize_post_hoc('spatial')
        retrain_record = self._summarize_post_hoc('retrain')

        output_record = {
            'temporalMeanMae': format_percent(temporal_record['meanMae']),
            'temporalMeanMdae': format_percent(temporal_record['meanMdae']),
            'temporalStdMae': format_percent(temporal_record['stdMae']),
            'temporalStdMdae': format_percent(temporal_record['stdMdae']),
            'temporalCount': round(temporal_record['count']),
            'temporalPercent': format_percent(temporal_record['percent']),
            'spatialMeanMae': format_percent(random_record['meanMae']),
            'spatialMeanMdae': format_percent(random_record['meanMdae']),
            'spatialStdMae': format_percent(random_record['stdMae']),
            'spatialStdMdae': format_percent(random_record['stdMdae']),
            'spatialCount': round(random_record['count']),
            'spatialPercent': format_percent(random_record['percent']),
            'randomMeanMae': format_percent(spatial_record['meanMae']),
            'randomMeanMdae': format_percent(spatial_record['meanMdae']),
            'randomStdMae': format_percent(spatial_record['stdMae']),
            'randomStdMdae': format_percent(spatial_record['stdMdae']),
            'randomCount': round(spatial_record['count']),
            'randomPercent': format_percent(spatial_record['percent']),
            'retrainMeanMae': format_percent(retrain_record['meanMae']),
            'retrainMeanMdae': format_percent(retrain_record['meanMdae']),
            'retrainStdMae': format_percent(retrain_record['stdMae']),
            'retrainStdMdae': format_percent(retrain_record['stdMdae']),
            'retrainCount': round(retrain_record['count']),
            'retrainPercent': format_percent(retrain_record['percent'])
        }

        with self.output().open('w') as f:
            json.dump(output_record, f)

    def _summarize_post_hoc(self, name):
        """Summarize a single post-hoc test.

        Args:
            name: The name of the test to summarize.

        Returns:
            Primitives-only dict describing the post-hoc test.
        """
        mean_running = 0
        std_running = 0
        count_running = 0
        test_running = 0
        means = []
        stds = []

        with self.input()[name].open() as f:
            rows = csv.DictReader(f)

            for row in rows:
                set_assignment = row['setAssign']
                abs_error_mean = abs(float(row['meanResidual']))
                abs_error_std = abs(float(row['stdResidual']))
                count = float(row['yieldObservations']) / const.RESOLUTION_SCALER

                mean_running += count * abs_error_mean
                std_running += count * abs_error_std
                means.append(abs_error_mean)
                stds.append(abs_error_std)
                count_running += count
                test_running += (count if set_assignment == 'test' else 0)

        return {
            'meanMae': mean_running / count_running,
            'stdMae': std_running / count_running,
            'meanMdae': statistics.median(means),
            'stdMdae': statistics.median(stds),
            'count': test_running,
            'percent': test_running / count_running
        }


class DeterminePercentSignificantTemplateTask(luigi.Task):
    """Task template to gather information about frequency of statistically significant results.

    Abstract base class (template class) for a Luigi task which gathers information about frequency
    of statistically significant results from a simulation.
    """

    def requires(self):
        """Require a simulation result.

        Returns:
            Luigi task whose output will be summarized.
        """
        return self.get_target()

    def output(self):
        """Get the location where the significant result rate should be written.

        Returns:
            LocalTarget where a summary of significance should be written.
        """
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        """Get information about significance."""

        with self.input().open() as f:
            records = csv.DictReader(f)
            records_mpci = filter(
                lambda x: abs(float(x['lossThreshold']) - 0.25) < 0.0001,
                records
            )

            records_2050 = list(filter(lambda x: '2050' in x['condition'], records_mpci))
            total_count = sum(map(lambda x: float(x['num']), records_2050))

            sig_records = filter(lambda x: int(x['pMeets0.05n']) == 1, records_2050)
            sig_geohashes = set(map(lambda x: x['geohash'], sig_records))

            sig_records_geohash = list(filter(
                lambda x: x['geohash'] in sig_geohashes,
                records_2050
            ))
            sig_count = sum(map(lambda x: float(x['num']), sig_records_geohash))

        percent = sig_count / total_count
        output_record = {'percentSignificant': format_percent(percent)}

        with self.output().open('w') as f:
            json.dump(output_record, f)

    def get_filename(self):
        """Get the filename in the workspace at which signifiance results should be written.

        Returns:
            String filename (not full path) where JSON will be written.
        """
        raise NotImplementedError('Use implementor.')

    def get_target(self):
        """Get the task whose output should be examined for significance information.

        Returns:
            Luigi task.
        """
        raise NotImplementedError('Use implementor.')


class DeterminePercentSignificantTask(DeterminePercentSignificantTemplateTask):
    """Task to determine signifiance from the main simulations on 4 char geohashes."""

    def get_filename(self):
        """Get the filename in the workspace at which signifiance results should be written.

        Returns:
            String filename (not full path) where JSON will be written.
        """
        return 'stats_significant.json'

    def get_target(self):
        """Get the task whose output should be examined for significance information.

        Returns:
            Luigi task.
        """
        return export_tasks.SummaryExportTask()


class DeterminePercentSignificantLongTask(DeterminePercentSignificantTemplateTask):
    """Determine signifiance from extended simulations on 5 character geohashes."""

    def get_filename(self):
        """Get the filename in the workspace at which signifiance results should be written.

        Returns:
            String filename (not full path) where JSON will be written.
        """
        return 'stats_significant_5char.json'

    def get_target(self):
        """Get the task whose output should be examined for significance information.

        Returns:
            Luigi task.
        """
        return export_tasks.SummaryExportLongTask()


class ExtractSimStatsTask(luigi.Task):
    """Extract information for the paper from main simulations."""

    def requires(self):
        """Require that simulation results are available.

        Returns:
            CombineSimulationsTasks
        """
        return sim_tasks.CombineSimulationsTasks()

    def output(self):
        """Indicate where simulation result statistics should be written.

        Returns:
            LocalTarget at which statistics should be written as JSON.
        """
        return luigi.LocalTarget(const.get_file_location('stats_sim.json'))

    def run(self):
        """Extract summary statistics for the simulations."""
        with self.input().open() as f:
            records = csv.DictReader(f)
            records_allowed = filter(
                lambda x: export_tasks.is_record_in_scope(x, 0.25),
                records
            )
            simplified_records = map(
                lambda x: self._simplify_record(x),
                records_allowed
            )
            reduced_records = toolz.itertoolz.reduceby(
                lambda x: self._get_record_key(x),
                lambda a, b: self._combine_records(a, b),
                simplified_records
            )

        output_record = {
            'counterfactualMean2030': format_percent(
                reduced_records['counterfactual2030']['mean']
            ),
            'counterfactualProbability2030': format_percent(
                reduced_records['counterfactual2030']['probability']
            ),
            'counterfactualSeverity2030': format_severity(
                reduced_records['counterfactual2030']['severity']
            ),
            'experimentalMean2030': format_percent(
                reduced_records['experimental2030']['mean']
            ),
            'experimentalProbability2030': format_percent(
                reduced_records['experimental2030']['probability']
            ),
            'experimentalSeverity2030': format_severity(
                reduced_records['experimental2030']['severity']
            ),
            'counterfactualMean2050': format_percent(
                reduced_records['counterfactual2050']['mean']
            ),
            'counterfactualProbability2050': format_percent(
                reduced_records['counterfactual2050']['probability']
            ),
            'counterfactualSeverity2050': format_severity(
                reduced_records['counterfactual2050']['severity']
            ),
            'experimentalMean2050': format_percent(
                reduced_records['experimental2050']['mean']
            ),
            'experimentalProbability2050': format_percent(
                reduced_records['experimental2050']['probability']
            ),
            'experimentalSeverity2050': format_severity(
                reduced_records['experimental2050']['severity']
            )
        }

        with self.output().open('w') as f:
            json.dump(output_record, f)

    def _simplify_record(self, record):
        """Simplify / standardize an input record, parsing attributes as numbers where appropriate

        Args:
            record: Raw dictionary to parse.

        Returns:
            Dictionary after parsing.
        """
        is_counterfactual = '_counterfactual' in record['series']
        year_series = int(record['series'].split('_')[0])

        return {
            'isCounterfactual': is_counterfactual,
            'year': year_series,
            'num': float(record['num']),
            'mean': float(record['predictedChange']),
            'probability': float(record['predictedClaims']),
            'severity': float(record['predictedLoss'])
        }

    def _get_record_key(self, record):
        """Generate a key identifying a year within a series type from which a record comes from.

        Args:
            record: The record for which a key is desired.

        Returns:
            String indicating the series type (experimental, counterfactual) and year (like 2024)
            that the record is from or represents.
        """
        prefix = 'counterfactual' if record['isCounterfactual'] else 'experimental'
        year = record['year']
        return '%s%d' % (prefix, year)

    def _combine_records(self, a, b):
        """Combine the samples between two simulation outcomes by pooling.

        Args:
            a: The first sample to pool.
            b: The second sample to pool.

        Returns:
            Dictionary representing the pooled samples.
        """
        assert self._get_record_key(a) == self._get_record_key(b)

        a_num = float(a['num'])
        b_num = float(b['num'])

        def get_weighted_avg(a_val, b_val, ignore_zero):
            if ignore_zero:
                if a_val == 0:
                    return b_val
                elif b_val == 0:
                    return a_val

            return (a_val * a_num + b_val * b_num) / (a_num + b_num)

        return {
            'isCounterfactual': a['isCounterfactual'],
            'year': a['year'],
            'num': a_num + b_num,
            'mean': get_weighted_avg(a['mean'], b['mean'], False),
            'probability': get_weighted_avg(a['probability'], b['probability'], False),
            'severity': get_weighted_avg(a['severity'], b['severity'], True)
        }


class SummarizeEquivalentStdTask(luigi.Task):
    """Summarize the standard deviation equivalent threshold

    Summarize the standard deviation equivalent threshold to a 25% below average based loss
    threshold.
    """

    def requires(self):
        """Require that the calculation of equivalent thresholds be completed.

        Returns:
            DetermineEquivalentStdTask
        """
        return sim_tasks.DetermineEquivalentStdTask()

    def output(self):
        """Determine where this threshold information should be written.

        Returns:
            LocalTarget at which the statistics should be written in JSON.
        """
        return luigi.LocalTarget(const.get_file_location('stats_equivalent.json'))

    def run(self):
        """Summarize the statistics."""
        with self.input().open('r') as f_in:
            with self.output().open('w') as f_out:
                source = json.load(f_in)
                json.dump({
                    'equivalentStd': '%.2f' % source['0.25']
                }, f_out)


class FindDivergentAphAndClaimsRate(luigi.Task):
    """Determine how often APH and claims both increase."""

    def requires(self):
        """Require that simulation results are available.

        Returns:
            CombineSimulationsTasks
        """
        return sim_tasks.CombineSimulationsTasks()
    
    def output(self):
        """Determine where the resulting statistics should be written.
        
        Returns:
            LocalTarget at which the JSON should be written.
        """
        return luigi.LocalTarget(const.get_file_location('divergent_aph_claims.json'))
    
    def run(self):
        """Calculate the rate of APH overall increase but increased claims."""
        with self.input().open('r') as f:
            all_data = csv.DictReader(f)
            right_baseline = filter(lambda x: x['offsetBaseline'] == 'always', all_data)
            right_condition = filter(lambda x: x['condition'] == '2050_SSP245', right_baseline)
            right_threshold = filter(
                lambda x: abs(float(x['threshold']) - 0.25) < 0.00001,
                right_condition
            )
            right_mult = filter(lambda x: int(float(x['stdMult'])) == 1, right_threshold)
            right_geohash = filter(lambda x: int(x['geohashSimSize']) == 4, right_mult)
            parsed = map(lambda x: self._parse_record(x), right_geohash)
            in_scope = list(parsed)
        
        # Determine geohashes with increased yield
        records_grouped_by_geohash = toolz.itertoolz.reduceby(
            lambda x: x['geohash'],
            lambda a, b: self._combine_means(a, b),
            in_scope
        ).values()

        geohash_summaries_increasing_yield = filter(
            lambda x: x['predictedChange'] >= 0,
            records_grouped_by_geohash
        )

        geohashes_increasing_yield = set(map(
            lambda x: x['geohash'],
            geohash_summaries_increasing_yield
        ))

        # Determine instances geohashes in which claims increase
        records_with_increase_risk = filter(
            lambda x: x['predictedClaims'] > x['baselineClaims'],
            in_scope
        )
        instances_with_increase_risk = set(map(
            lambda x: x['geohash'],
            records_with_increase_risk
        ))

        # Determine statistic
        geohashes_with_dual_increase = instances_with_increase_risk.intersection(
            geohashes_increasing_yield
        )
        
        rate = len(geohashes_with_dual_increase) / len(instances_with_increase_risk)

        # Output
        with self.output().open('w') as f:
            json.dump({'dualIncreasePercent2050': format_percent(rate)}, f)
    
    def _parse_record(self, target):
        """Parse a raw input record from the simulation results.
        
        Args:
            target: The record to parse (primitives-only dictionary).
        
        Returns:
            Parsed record.
        """
        return {
            'geohash': target['geohash'],
            'num': float(target['num']),
            'baselineChange': float(target['baselineChange']),
            'predictedChange': float(target['predictedChange']),
            'baselineClaims': float(target['baselineClaims']),
            'predictedClaims': float(target['predictedClaims'])
        }
    
    def _combine_means(self, a, b):
        """Pool yield change means.
        
        Args:
            a: The first record to pool.
            b: The second record to pool.
        
        Returns:
            Record after pooling samples.
        """
        assert a['geohash'] == b['geohash']
        new_count = a['num'] + b['num']

        def combine_key(key):
            pool_sum = a['num'] * a[key] + b['num'] * b[key]
            return pool_sum / new_count
        
        return {
            'geohash': a['geohash'],
            'num': new_count,
            'baselineChange': combine_key('baselineChange'),
            'predictedChange': combine_key('predictedChange')
        }


class CombineStatsTask(luigi.Task):
    """Create a combined statistical output as a JSON document."""

    def requires(self):
        """Require other statistical tasks have been completed.

        Returns:
            Various statistical tasks that feed into the combined JSON output.
        """
        return {
            'model': ExportModelInfoTask(),
            'posthoc': ExportPosthocTestTask(),
            'significance': DeterminePercentSignificantTask(),
            'sim': ExtractSimStatsTask(),
            'std': SummarizeEquivalentStdTask(),
            'dual': FindDivergentAphAndClaimsRate()
        }

    def output(self):
        """Indicate where the combined statistical output should be written.

        Returns:
            LocalTarget at which the JSON will be written.
        """
        return luigi.LocalTarget(const.get_file_location('stats.json'))

    def run(self):
        """Combine outputs."""
        model_inputs = self._get_subfile('model')
        posthoc_inputs = self._get_subfile('posthoc')
        significance_inputs = self._get_subfile('significance')
        sim_inputs = self._get_subfile('sim')
        std_inputs = self._get_subfile('std')
        dual_inputs = self._get_subfile('dual')

        output_record = {
            'numLayers': model_inputs['numLayers'],
            'layersDescription': model_inputs['layersDescription'],
            'dropout': model_inputs['dropout'],
            'l2': model_inputs['l2'],
            'countInfoStr': model_inputs['countInfoStr'],
            'trainMeanMae': model_inputs['trainMeanMae'],
            'trainStdMae': model_inputs['trainStdMae'],
            'validationMeanMae': model_inputs['validationMeanMae'],
            'validationStdMae': model_inputs['validationStdMae'],
            'testMeanMae': model_inputs['testMeanMae'],
            'testStdMae': model_inputs['testStdMae'],
            'retrainMeanMae': posthoc_inputs['retrainMeanMae'],
            'retrainMeanMdae': posthoc_inputs['retrainMeanMdae'],
            'retrainStdMae': posthoc_inputs['retrainStdMae'],
            'retrainStdMdae': posthoc_inputs['retrainStdMdae'],
            'retrainCount': posthoc_inputs['retrainCount'],
            'retrainPercent': posthoc_inputs['retrainPercent'],
            'temporalMeanMae': posthoc_inputs['temporalMeanMae'],
            'temporalMeanMdae': posthoc_inputs['temporalMeanMdae'],
            'temporalStdMae': posthoc_inputs['temporalStdMae'],
            'temporalStdMdae': posthoc_inputs['temporalStdMdae'],
            'temporalCount': posthoc_inputs['temporalCount'],
            'temporalPercent': posthoc_inputs['temporalPercent'],
            'spatialMeanMae': posthoc_inputs['spatialMeanMae'],
            'spatialMeanMdae': posthoc_inputs['spatialMeanMdae'],
            'spatialStdMae': posthoc_inputs['spatialStdMae'],
            'spatialStdMdae': posthoc_inputs['spatialStdMdae'],
            'spatialCount': posthoc_inputs['spatialCount'],
            'spatialPercent': posthoc_inputs['spatialPercent'],
            'randomMeanMae': posthoc_inputs['randomMeanMae'],
            'randomMeanMdae': posthoc_inputs['randomMeanMdae'],
            'randomStdMae': posthoc_inputs['randomStdMae'],
            'randomStdMdae': posthoc_inputs['randomStdMdae'],
            'randomCount': posthoc_inputs['randomCount'],
            'randomPercent': posthoc_inputs['randomPercent'],
            'percentSignificant': significance_inputs['percentSignificant'],
            'counterfactualMean2030': sim_inputs['counterfactualMean2030'],
            'counterfactualProbability2030': sim_inputs['counterfactualProbability2030'],
            'counterfactualSeverity2030': sim_inputs['counterfactualSeverity2030'],
            'experimentalMean2030': sim_inputs['experimentalMean2030'],
            'experimentalProbability2030': sim_inputs['experimentalProbability2030'],
            'experimentalSeverity2030': sim_inputs['experimentalSeverity2030'],
            'counterfactualMean2050': sim_inputs['counterfactualMean2050'],
            'counterfactualProbability2050': sim_inputs['counterfactualProbability2050'],
            'counterfactualSeverity2050': sim_inputs['counterfactualSeverity2050'],
            'experimentalMean2050': sim_inputs['experimentalMean2050'],
            'experimentalProbability2050': sim_inputs['experimentalProbability2050'],
            'experimentalSeverity2050': sim_inputs['experimentalSeverity2050'],
            'equivalentStd': std_inputs['equivalentStd'],
            'dualIncreasePercent2050': dual_inputs['dualIncreasePercent2050']
        }

        with self.output().open('w') as f:
            json.dump(output_record, f)

    def _get_subfile(self, key):
        """Load one of the prerequisite statistical summary outputs.

        Args:
            key: Name of the input task.

        Returns:
            Results of that summary task.
        """
        with self.input()[key].open() as f:
            result = json.load(f)

        return result
