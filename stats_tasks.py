import csv
import json
import statistics

import luigi
import toolz.itertoolz

import const
import export_tasks
import selection_tasks
import sim_tasks

NEURONS_PER_LAYER = [256, 128, 64, 32, 16, 8]


def format_percent(target):
    return '%.1f\\%%' % (target * 100)


def format_severity(target):
    return format_percent(target * -1 - 0.25)


class ExportModelInfoTask(luigi.Task):

    def requires(self):
        return selection_tasks.SelectConfigurationTask()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('stats_model.json'))

    def run(self):
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

    def requires(self):
        return {
            'temporal': selection_tasks.PostHocTestRawDataTemporalCountTask(),
            'random': selection_tasks.PostHocTestRawDataRandomCountTask(),
            'spatial': selection_tasks.PostHocTestRawDataSpatialCountTask()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location('stats_posthoc.json'))

    def run(self):
        temporal_record = self._summarize_post_hoc('temporal')
        random_record = self._summarize_post_hoc('random')
        spatial_record = self._summarize_post_hoc('spatial')

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
            'randomPercent': format_percent(spatial_record['percent'])
        }

        with self.output().open('w') as f:
            json.dump(output_record, f)

    def _summarize_post_hoc(self, name):
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


class DeterminePercentSignificantTask(luigi.Task):

    def requires(self):
        return export_tasks.SummaryExportTask()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('stats_significant.json'))

    def run(self):
        
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

            sig_records_geohash = list(filter(lambda x: x['geohash'] in sig_geohashes, records_2050))
            sig_count = sum(map(lambda x: float(x['num']), sig_records_geohash))

        percent = sig_count / total_count
        output_record = {'percentSignificant': format_percent(percent)}

        with self.output().open('w') as f:
            json.dump(output_record, f)


class ExtractSimStatsTask(luigi.Task):

    def requires(self):
        return sim_tasks.CombineSimulationsTasks()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('stats_sim.json'))

    def run(self):
        with self.input().open() as f:
            records = csv.DictReader(f)
            records_allowed = filter(
                lambda x: export_tasks.is_default_config_record(x, 0.25),
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
            'counterfactualMean2030': format_percent(reduced_records['counterfactual2030']['mean']),
            'counterfactualProbability2030': format_percent(reduced_records['counterfactual2030']['probability']),
            'counterfactualSeverity2030': format_severity(reduced_records['counterfactual2030']['severity']),
            'experimentalMean2030': format_percent(reduced_records['experimental2030']['mean']),
            'experimentalProbability2030': format_percent(reduced_records['experimental2030']['probability']),
            'experimentalSeverity2030': format_severity(reduced_records['experimental2030']['severity']),
            'counterfactualMean2050': format_percent(reduced_records['counterfactual2050']['mean']),
            'counterfactualProbability2050': format_percent(reduced_records['counterfactual2050']['probability']),
            'counterfactualSeverity2050': format_severity(reduced_records['counterfactual2050']['severity']),
            'experimentalMean2050': format_percent(reduced_records['experimental2050']['mean']),
            'experimentalProbability2050': format_percent(reduced_records['experimental2050']['probability']),
            'experimentalSeverity2050': format_severity(reduced_records['experimental2050']['severity'])
        }

        with self.output().open('w') as f:
            json.dump(output_record, f)

    def _simplify_record(self, record):
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
        prefix = 'counterfactual' if record['isCounterfactual'] else 'experimental'
        year = record['year']
        return '%s%d' % (prefix, year)

    def _combine_records(self, a, b):
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

    def requires(self):
        return sim_tasks.DetermineEquivalentStdTask()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('stats_equivalent.json'))

    def run(self):
        with self.input().open('r') as f_in:
            with self.output().open('w') as f_out:
                source = json.load(f_in)
                json.dump({
                    'equivalentStd': '%.2f' % source['0.25']
                }, f_out)


class CombineStatsTask(luigi.Task):

    def requires(self):
        return {
            'model': ExportModelInfoTask(),
            'posthoc': ExportPosthocTestTask(),
            'significance': DeterminePercentSignificantTask(),
            'sim': ExtractSimStatsTask(),
            'std': SummarizeEquivalentStdTask()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location('stats.json'))

    def run(self):
        model_inputs = self._get_subfile('model')
        posthoc_inputs = self._get_subfile('posthoc')
        significance_inputs = self._get_subfile('significance')
        sim_inputs = self._get_subfile('sim')
        std_inputs = self._get_subfile('std')

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
            'equivalentStd': std_inputs['equivalentStd']
        }

        with self.output().open('w') as f:
            json.dump(output_record, f)

    def _get_subfile(self, key):
        with self.input()[key].open() as f:
            result = json.load(f)

        return result
