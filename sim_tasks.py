"""Tasks that run the Monte Carlo simulations.

License:
    BSD
"""
import csv
import functools
import itertools
import json
import random
import statistics

import keras
import luigi
import pandas
import toolz

import cluster_tasks
import const
import normalize_tasks
import selection_tasks
import training_tasks

BINS = list(map(lambda x: x - 100, range(0, 205, 5)))
BIN_FIELDS = ['bin%d' % x for x in BINS]
OUTPUT_FIELDS = [
    'offsetBaseline',
    'unitSize',
    'geohash',
    'year',
    'condition',
    'threshold',
    'thresholdStd',
    'stdMult',
    'geohashSimSize',
    'num',
    'predictedChange',
    'baselineChange',
    'adaptedChange',
    'predictedClaims',
    'baselineClaims',
    'adaptedClaims',
    'predictedLoss',
    'baselineLoss',
    'adaptedLoss',
    'predictedClaimsStd',
    'baselineClaimsStd',
    'adaptedClaimsStd',
    'predictedLossStd',
    'baselineLossStd',
    'adaptedLossStd',
    'p',
    'pAdapted'
] + BIN_FIELDS
NUM_ARGS = 4
STD_MULT = [1.0]
THRESHOLDS = [0.25, 0.15]
GEOHASH_SIZE = [4, 5]
SIM_PARTIAL_AVG = 0


class Task:
    """Task defining a simulation to execute."""

    def __init__(self, geohash, year, condition, original_mean, original_std, original_skew,
        original_kurtosis, projected_mean, projected_std, projected_skew, projected_kurtosis,
        num_observations):
        """Create a new task record.

        Args:
            geohash: The name of the geohash to be simulated.
            year: The year for which the geohash is to be simulated.
            condition: The name of the condition for which the geohash should be simulated like
                2050_SSP245.
            original_mean: The original mean of yield deltas (in the counterfactual or baseline) in
                this geohash.
            original_std: The original standard deviation of yield deltas (in the counterfactual or
                baseline) in this geohash.
            original_skew: The original skew of yield deltas (in the counterfactual or baseline) in
                this geohash.
            original_kurtosis: The original skew of yield deltas (in the counterfactual or baseline)
                in this geohash.
            projected_mean: The experimental (neural network predicted) mean of yield deltas in this
                geohash.
            projected_std: The experimental (neural network predicted) standard deviation of yield
                deltas in this geohash.
            projected_skew: The experimental (neural network predicted) skew of yield deltas in this
                geohash.
            projected_kurtosis: The experimental (neural network predicted) kurtosis of yield deltas
                in this geohash.
            num_observations: The sample size or number of observations (pixels) in this geohash.
        """
        self._geohash = geohash
        self._year = year
        self._condition = condition
        self._original_mean = original_mean
        self._original_std = original_std
        self._original_skew = original_skew
        self._original_kurtosis = original_kurtosis
        self._projected_mean = projected_mean
        self._projected_std = projected_std
        self._projected_skew = projected_skew
        self._projected_kurtosis = projected_kurtosis
        self._num_observations = num_observations

    def get_geohash(self):
        """Get the name of the geohash to simulate.

        Returns:
            The name of the geohash to be simulated.
        """
        return self._geohash

    def get_year(self):
        """Get the year for which the geohash should be simulated.

        Returns:
            The year for which the geohash is to be simulated.
        """
        return self._year

    def get_condition(self):
        """Get the condition in which the geohash should be simulated.

        Returns:
            The name of the condition for which the geohash should be simulated like 2050_SSP245.
        """
        return self._condition

    def get_original_mean(self):
        """Get the original yield delta mean in the baseline or counterfactual.

        Returns:
            The original mean of yield deltas (in the counterfactual or baseline) in this geohash.
        """
        return self._original_mean

    def get_original_std(self):
        """Get the original yield delta standard deviation in the baseline or counterfactual.

        Returns:
            The original standard deviation of yield deltas (in the counterfactual or baseline) in
            this geohash.
        """
        return self._original_std

    def get_original_skew(self):
        """Get the original yield delta skew in the baseline or counterfactual.

        Returns:
            The original skew of yield deltas (in the counterfactual or baseline) in this geohash.
        """
        return self._original_skew

    def get_original_kurtosis(self):
        """Get the original yield delta kurtosis in the baseline or counterfactual.

        Returns:
            The original kurtosis of yield deltas (in the counterfactual or baseline) in this
            geohash.
        """
        return self._original_kurtosis

    def get_projected_mean(self):
        """Get the yield delta mean in the experimental or predicted series.

        Returns:
            The experimental (neural network predicted) mean of yield deltas in this geohash.
        """
        return self._projected_mean

    def get_projected_std(self):
        """Get the yield delta standard deivation in the experimental or predicted series.

        Returns:
            The experimental (neural network predicted) standard deviation of yield deltas in this
            geohash.
        """
        return self._projected_std

    def get_projected_skew(self):
        """Get the yield delta skew in the experimental or predicted series.

        Returns:
            The experimental (neural network predicted) skew of yield deltas in this geohash.
        """
        return self._projected_skew

    def get_projected_kurtosis(self):
        """Get the yield delta kurtosis in the experimental or predicted series.

        Returns:
            The experimental (neural network predicted) kurtosis of yield deltas in this geohash.
        """
        return self._projected_kurtosis

    def get_num_observations(self):
        """Get the number of observations (pixels) for this simulation.

        Returns:
            The sample size or number of observations (pixels) in this geohash.
        """
        return self._num_observations


def run_simulation(task, deltas, threshold, std_mult, geohash_sim_size, offset_baseline,
    unit_sizes, std_thresholds):
    """Run a single geohash simulation.

    Run a single geohash simulation from within a self-contained function that can run in
    distribution meaning it has its own imports and can be exported to other machines.

    Args:
        task: The task describing the simulation to execute.
        deltas: The model residuals as yield deltas to sample (should be a dictionary with mean and
            std mapping to list of numbers).
        threshold: The loss / claims threshold as a precent like 0.15 for 15% below average.
        std_mult: The amount by which to muiltiply the standard deviation in simulation. A value of
            1 leaves things as is from the model outputs.
        geohash_sim_size: The size of the geohash (number of characters like 4) to simulate.
        offset_baseline: Should be 'always' or 'negative' or 'never' describing when the APH should
            be calculated by sampling the original distribution. Always means that the original
            distribution is always sampled, never means APH is assumed to be baseline yield, and
            negative means sampling only when the predicted average is lower than the original.
        unit_sizes: List of insured unit sizes to sample as pixels. See `refine/unit_size.json` for
            more details.
        std_thresholds: The loss / claims threshold as standard deviations like 2.11 for 2.11
            standard deviations below average.

    Returns:
        Dictionary with OUTPUT_FIELDS describing simulation results.
    """

    import math
    import random

    import distribution_struct
    import distribution_util

    import scipy.stats
    import toolz.itertoolz

    def draw_number(mean, std, skew, kurtosis):
        distribution = distribution_util.find_beta_distribution(mean, std, skew, kurtosis)

        return scipy.stats.beta.rvs(
            distribution['a'],
            distribution['b'],
            loc=distribution['loc'],
            scale=distribution['scale'],
            size=1
        )[0]

    std_threshold = std_thresholds['%.2f' % threshold]

    mean_deltas = deltas['mean']
    std_deltas = deltas['std']
    skew_deltas = deltas['skew']
    kurtosis_deltas = deltas['kurtosis']

    original_mean = task.get_original_mean()
    original_std = task.get_original_std()
    projected_mean = task.get_projected_mean()
    projected_std = task.get_projected_std()
    projected_skew = task.get_projected_skew()
    projected_kurtosis = task.get_projected_kurtosis()
    num_observations = task.get_num_observations() / const.RESOLUTION_SCALER

    assert projected_std >= 0

    if geohash_sim_size == 5:
        num_observations = round(num_observations / 32)
        unit_size_multiplier = 1 / 32
    else:
        unit_size_multiplier = 1

    if num_observations == 0:
        return None

    predicted_deltas = []
    baseline_deltas = []
    adapted_deltas = []
    pixels_remaining = num_observations
    while pixels_remaining > 0:

        unit_size = random.choice(unit_sizes) * unit_size_multiplier
        unit_size_scaled = math.ceil(unit_size / const.RESOLUTION_SCALER)
        pixels_remaining = pixels_remaining - unit_size_scaled

        predicted_yield_acc = distribution_struct.WelfordAccumulator()
        adapted_yield_acc = distribution_struct.WelfordAccumulator()

        for pixel_i in range(0, unit_size_scaled):

            mean_delta = random.choice(mean_deltas) * -1
            std_delta = random.choice(std_deltas) * -1
            skew_delta = random.choice(skew_deltas) * -1
            kurtosis_delta = random.choice(kurtosis_deltas) * -1

            sim_mean = projected_mean + mean_delta
            sim_std = projected_std * std_mult + std_delta
            sim_skew = projected_skew + skew_delta
            sim_kurtosis = projected_kurtosis + kurtosis_delta

            if sim_std <= 0:
                sim_std = 0.0001

            predicted_yield = draw_number(sim_mean, sim_std, sim_skew, sim_kurtosis)
            adapted_yield = predicted_yield + sim_std

            predicted_yield_acc.add(predicted_yield)
            adapted_yield_acc.add(adapted_yield)

        def execute_offset_baseline(value, offset):
            half_way = (value + offset) / 2
            effective_offset = half_way * SIM_PARTIAL_AVG + offset * (1 - SIM_PARTIAL_AVG)
            return value - effective_offset

        if offset_baseline == 'always':
            predicted_delta = execute_offset_baseline(
                predicted_yield_acc.get_mean(),
                original_mean
            )
            adapted_delta = execute_offset_baseline(
                adapted_yield_acc.get_mean(),
                original_mean
            )
        elif offset_baseline == 'negative' and original_mean < 0:
            predicted_delta = execute_offset_baseline(
                predicted_yield_acc.get_mean(),
                original_mean
            )
            adapted_delta = execute_offset_baseline(
                adapted_yield_acc.get_mean(),
                original_mean
            )
        else:
            predicted_delta = predicted_yield_acc.get_mean()
            adapted_delta = adapted_yield_acc.get_mean()

        baseline_deltas.append(original_mean)
        predicted_deltas.append(predicted_delta)
        adapted_deltas.append(adapted_delta)

    def get_claims_rate(target, inner_threshold=threshold):
        neg_threshold = inner_threshold * -1
        claims = filter(lambda x: x <= neg_threshold, target)
        num_claims = sum(map(lambda x: 1, claims))
        return num_claims / len(target)

    def get_loss_level(target, inner_threshold=threshold):
        neg_threshold = inner_threshold * -1
        claims = list(filter(lambda x: x <= neg_threshold, target))
        if len(claims) > 0:
            return statistics.mean(claims)
        else:
            return 0

    def get_claims_rate_std(target):
        converted_threshold = original_std * std_threshold
        return get_claims_rate(target, inner_threshold=converted_threshold)

    def get_loss_level_std(target):
        converted_threshold = original_std * std_threshold
        return get_loss_level(target, inner_threshold=converted_threshold)

    def get_change(target):
        if len(target) > 0:
            return statistics.mean(target)
        else:
            return 0

    baseline_change = get_change(baseline_deltas)
    predicted_change = get_change(predicted_deltas)
    adapted_change = get_change(adapted_deltas)

    baseline_claims_rate = get_claims_rate(baseline_deltas)
    predicted_claims_rate = get_claims_rate(predicted_deltas)
    adapted_claims_rate = get_claims_rate(adapted_deltas)

    baseline_loss = get_loss_level(baseline_deltas)
    predicted_loss = get_loss_level(predicted_deltas)
    adapted_loss = get_loss_level(adapted_deltas)

    baseline_claims_rate_std = get_claims_rate_std(baseline_deltas)
    predicted_claims_rate_std = get_claims_rate_std(predicted_deltas)
    adapted_claims_rate_std = get_claims_rate_std(adapted_deltas)

    baseline_loss_std = get_loss_level_std(baseline_deltas)
    predicted_loss_std = get_loss_level_std(predicted_deltas)
    adapted_loss_std = get_loss_level_std(adapted_deltas)

    p_baseline = scipy.stats.mannwhitneyu(predicted_deltas, baseline_deltas)[1]
    p_adapted = scipy.stats.mannwhitneyu(predicted_deltas, adapted_deltas)[1]

    # This would ideally be a structured object.
    ret_dict = {
        'offsetBaseline': offset_baseline,
        'unitSize': unit_size,
        'geohash': task.get_geohash(),
        'year': task.get_year(),
        'condition': task.get_condition(),
        'threshold': threshold,
        'thresholdStd': std_threshold,
        'stdMult': std_mult,
        'geohashSimSize': geohash_sim_size,
        'num': num_observations,
        'predictedChange': predicted_change,
        'baselineChange': baseline_change,
        'adaptedChange': adapted_change,
        'predictedClaims': predicted_claims_rate,
        'baselineClaims': baseline_claims_rate,
        'adaptedClaims': adapted_claims_rate,
        'predictedLoss': predicted_loss,
        'baselineLoss': baseline_loss,
        'adaptedLoss': adapted_loss,
        'predictedClaimsStd': predicted_claims_rate_std,
        'baselineClaimsStd': baseline_claims_rate_std,
        'adaptedClaimsStd': adapted_claims_rate_std,
        'predictedLossStd': predicted_loss_std,
        'baselineLossStd': baseline_loss_std,
        'adaptedLossStd': adapted_loss_std,
        'p': p_baseline,
        'pAdapted': p_adapted
    }

    def cap_delta(target):
        if target > 1:
            return 1
        elif target < -1:
            return -1
        else:
            return target

    def project_delta(target):
        percent = target * 100
        percent_rounded = round(percent / 5) * 5
        return percent_rounded

    predicted_deltas_cap = map(cap_delta, predicted_deltas)
    predicted_deltas_projected = map(project_delta, predicted_deltas_cap)
    deltas_tuples = map(lambda x: (x, 1), predicted_deltas_projected)
    deltas_count = dict(toolz.itertoolz.reduceby(
        lambda x: x[0],
        lambda a, b: (a[0], a[1] + b[1]),
        deltas_tuples
    ).values())

    for bin_num in BINS:
        ret_dict['bin%d' % bin_num] = deltas_count.get(bin_num, 0)

    return ret_dict


def parse_record(record_raw):
    """Parse a single raw tuple record describing a simulation task as a Task object.

    Args:
        record_raw: The raw tuple record to parse.

    Returns:
        The record after parsing into a Task object.
    """
    geohash = str(record_raw[0])
    year = int(record_raw[1])
    condition = str(record_raw[2])
    original_mean = float(record_raw[3])
    original_std = float(record_raw[4])
    projected_mean = float(record_raw[5])
    projected_std = float(record_raw[6])
    num_observations = int(record_raw[7])

    return Task(
        geohash,
        year,
        condition,
        original_mean,
        original_std,
        projected_mean,
        projected_std,
        num_observations
    )


def parse_record_dict(record_raw):
    """Parse a single raw dict record describing a simulation task as a Task object.

    Args:
        record_raw: The raw dict record to parse.

    Returns:
        The record after parsing into a Task object.
    """
    geohash = str(record_raw['geohash'])
    year = int(record_raw['year'])
    condition = str(record_raw['condition'])
    original_mean = float(record_raw['originalYieldMean'])
    original_std = float(record_raw['originalYieldStd'])
    original_skew = float(record_raw['originalYieldSkew'])
    original_kurtosis = float(record_raw['originalYieldKurtosis'])
    projected_mean = float(record_raw['projectedYieldMean'])
    projected_std = float(record_raw['projectedYieldStd'])
    projected_skew = float(record_raw['projectedYieldSkew'])
    projected_kurtosis = float(record_raw['projectedYieldKurtosis'])
    num_observations = int(record_raw['numObservations'])

    return Task(
        geohash,
        year,
        condition,
        original_mean,
        original_std,
        original_skew,
        original_kurtosis,
        projected_mean,
        projected_std,
        projected_skew,
        projected_kurtosis,
        num_observations
    )


def run_simulation_set(tasks, deltas, threshold, std_mult, geohash_sim_size, offset_baseline,
    unit_sizes, std_thresholds):
    """Run a set of simulations.

    Args:
        tasks: The tasks describing the simulations to execute.
        deltas: The model residuals as yield deltas to sample (should be a dictionary with mean and
            std mapping to list of numbers).
        threshold: The loss / claims threshold as a precent like 0.15 for 15% below average.
        std_mult: The amount by which to muiltiply the standard deviation in simulation. A value of
            1 leaves things as is from the model outputs.
        geohash_sim_size: The size of the geohash (number of characters like 4) to simulate.
        offset_baseline: Should be 'always' or 'negative' or 'never' describing when the APH should
            be calculated by sampling the original distribution. Always means that the original
            distribution is always sampled, never means APH is assumed to be baseline yield, and
            negative means sampling only when the predicted average is lower than the original.
        unit_sizes: List of insured unit sizes to sample as pixels. See `refine/unit_size.json` for
            more details.
        std_thresholds: The loss / claims threshold as standard deviations like 2.11 for 2.11
            standard deviations below average.

    Returns:
        List of dictionaries with OUTPUT_FIELDS describing simulation results.
    """
    results_all = map(
        lambda x: run_simulation(
            x,
            deltas,
            threshold,
            std_mult,
            geohash_sim_size,
            offset_baseline,
            unit_sizes,
            std_thresholds
        ),
        tasks
    )
    results_valid = filter(lambda x: x is not None, results_all)
    results_realized = list(results_valid)
    return results_realized


class NormalizeRefHistoricTrainingFrameTask(luigi.Task):
    """Get the historic reference data.

    Prepare the historic reference data (not used for model training but for baselines in the
    simulations). This specifically filters normalized rows prepared in a prior step and checks the
    output fields are expected.
    """

    def requires(self):
        """Indicate which task whose output to filter.

        Returns:
            NormalizeHistoricTrainingFrameTask
        """
        return normalize_tasks.NormalizeHistoricTrainingFrameTask()

    def output(self):
        """Indicate where the filtered data should be written.

        Returns:
            LocalTarget at which filtered data should be written.
        """
        return luigi.LocalTarget(const.get_file_location('ref_historic_normalized.csv'))

    def run(self):
        """Run the filter and field check."""
        with self.input().open('r') as f_in:
            reader = csv.DictReader(f_in)
            allowed_rows = filter(lambda x: int(x['year']) in const.FUTURE_REF_YEARS, reader)

            with self.output().open('w') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=const.TRAINING_FRAME_ATTRS)
                writer.writeheader()
                writer.writerows(allowed_rows)


class NormalizeRefHistoricEarlyTask(luigi.Task):
    """Get the historic reference data for the first 10 years.

    Prepare the historic reference data (not used for model training but for baselines in the
    simulations). This specifically filters normalized rows prepared in a prior step and checks the
    output fields are expected.
    """

    def requires(self):
        """Indicate which task whose output to filter.

        Returns:
            NormalizeHistoricTrainingFrameTask
        """
        return normalize_tasks.NormalizeHistoricTrainingFrameTask()

    def output(self):
        """Indicate where the filtered data should be written.

        Returns:
            LocalTarget at which filtered data should be written.
        """
        return luigi.LocalTarget(const.get_file_location('ref_historic_normalized.csv'))

    def run(self):
        """Run the filter and field check."""
        with self.input().open('r') as f_in:
            reader = csv.DictReader(f_in)
            allowed_rows = filter(lambda x: int(x['year']) <= 2008, reader)

            with self.output().open('w') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=const.TRAINING_FRAME_ATTRS)
                writer.writeheader()
                writer.writerows(allowed_rows)


class NormalizeRefHistoricLateTask(luigi.Task):
    """Get the historic reference data for the final 9 years.

    Prepare the historic reference data (not used for model training but for baselines in the
    simulations). This specifically filters normalized rows prepared in a prior step and checks the
    output fields are expected.
    """

    def requires(self):
        """Indicate which task whose output to filter.

        Returns:
            NormalizeHistoricTrainingFrameTask
        """
        return normalize_tasks.NormalizeHistoricTrainingFrameTask()

    def output(self):
        """Indicate where the filtered data should be written.

        Returns:
            LocalTarget at which filtered data should be written.
        """
        return luigi.LocalTarget(const.get_file_location('ref_historic_normalized.csv'))

    def run(self):
        """Run the filter and field check."""
        with self.input().open('r') as f_in:
            reader = csv.DictReader(f_in)
            allowed_rows = filter(lambda x: int(x['year']) > 2008, reader)

            with self.output().open('w') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=const.TRAINING_FRAME_ATTRS)
                writer.writeheader()
                writer.writerows(allowed_rows)


class GetNumObservationsTask(luigi.Task):
    """Get the number of yield observations per geohash and year."""

    def requires(self):
        """Indicate that this task requires historic data.

        Returns:
            NormalizeRefHistoricTrainingFrameTask
        """
        return NormalizeRefHistoricTrainingFrameTask()

    def output(self):
        """Get the location at which the observations should be written.

        Returns:
            LocalTarget at which the CSV file should be written.
        """
        return luigi.LocalTarget(const.get_file_location('observation_counts.csv'))

    def run(self):
        """Get the number of yield observations per year / geohash."""
        with self.input().open('r') as f_in:
            with self.output().open('w') as f_out:
                rows = csv.DictReader(f_in)
                standard_rows = map(lambda x: self._standardize_row(x), rows)

                writer = csv.DictWriter(f_out, fieldnames=['geohash', 'year', 'yieldObservations'])
                writer.writeheader()
                writer.writerows(standard_rows)

    def _standardize_row(self, target):
        """Ensure an input dataset fits an expected format.

        Args:
            target: The row to standardize and from which to parse numeric data.

        Returns:
            Standardized row.
        """
        return {
            'geohash': target['geohash'],
            'year': int(target['year']),
            'yieldObservations': int(target['yieldObservations'])
        }


class ProjectTaskTemplate(luigi.Task):
    """Project yield information into the future."""

    def requires(self):
        """Indicate the prerequisite tasks.

        Indicate that model and selected sweep configuration are required along with a frame
        containing input data.

        Returns:
            TrainFullModel and SelectConfigurationTask as well as the target task.
        """
        return {
            'model': selection_tasks.TrainFullModel(),
            'target': self.get_target_task(),
            'configuration': selection_tasks.SelectConfigurationTask(),
            'dist': normalize_tasks.GetInputDistributionsTask()
        }

    def output(self):
        """Determine where the projections should be written.

        Returns:
            LocalTarget at which the results should be written.
        """
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        """Use the trained model to project yield information into the future."""
        target_frame = pandas.read_csv(self.input()['target'].path)

        with self.input()['configuration'].open('r') as f:
            configuration = json.load(f)['constrained']

        with self.input()['dist'].open('r') as f:
            rows = csv.DictReader(f)

            distributions = {}

            for row in rows:
                distributions[row['field']] = {
                    'mean': float(row['mean']),
                    'std': float(row['std'])
                }

        model = keras.models.load_model(self.input()['model'].path)

        additional_block = configuration['block']
        allow_count = configuration['allowCount'].lower() == 'true'

        target_frame['joinYear'] = target_frame['year']
        target_frame['simYear'] = target_frame['year'] - 2007 + self.get_base_year()
        target_frame['year'] = target_frame['simYear']

        input_attrs = training_tasks.get_input_attrs(additional_block, allow_count)
        inputs = target_frame[input_attrs]

        outputs = model.predict(inputs)

        def process_output(name, index):
            name_capitalized = name.capitalize()
            dist = distributions['yield%s' % name_capitalized]
            raw_values = outputs[:, index]

            if const.JIT_UNNORM_YIELD:
                transformed = raw_values * dist['std'] + dist['mean']
            else:
                transformed = raw_values

            target_frame['predicted%s' % name_capitalized] = transformed

        process_output('mean', 0)
        process_output('std', 1)
        process_output('skew', 2)
        process_output('kurtosis', 3)

        target_frame[[
            'geohash',
            'simYear',
            'joinYear',
            'predictedMean',
            'predictedStd',
            'predictedSkew',
            'predictedKurtosis',
            'yieldObservations'
        ]].to_csv(self.output().path)

    def get_target_task(self):
        """Get the task whose output should be used as model inputs.

        Returns:
            Luigi task.
        """
        raise NotImplementedError('Use implementor.')

    def get_base_year(self):
        """Get the "center" year of the series to be predicted.

        Returns:
            Integer year like 2007, 2030, or 2050.
        """
        raise NotImplementedError('Use implementor.')

    def get_filename(self):
        """Get the filename at which the projections should be written.

        Returns:
            Filename (not full path) as string.
        """
        raise NotImplementedError('Use implementor.')


class InterpretProjectTaskTemplate(luigi.Task):
    """Convert projections into a format expected by simulation tasks."""

    def requires(self):
        """Get the prerequisite tasks.

        Indicate that distribution information is needed and the task whose output is to be
        interpreted.

        Returns:
            GetInputDistributionsTask and the target task.
        """
        return {
            'target': self.get_target_task(),
            'dist': normalize_tasks.GetInputDistributionsTask()
        }

    def output(self):
        """Indicate where the reformatted data should be written.

        Returns:
            LocalTarget at which outputs should be written.
        """
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        """Execute the reformatting."""
        with self.input()['dist'].open('r') as f:
            rows = csv.DictReader(f)

            distributions = {}

            for row in rows:
                distributions[row['field']] = {
                    'mean': float(row['mean']),
                    'std': float(row['std'])
                }

        with self.output().open('w') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=[
                'geohash',
                'simYear',
                'joinYear',
                'predictedMean',
                'predictedStd',
                'predictedSkew',
                'predictedKurtosis',
                'yieldObservations'
            ])
            writer.writeheader()

            with self.input()['target'].open('r') as f_in:
                reader = csv.DictReader(f_in)
                standardized_rows = map(lambda x: self._standardize_row(x), reader)
                updated_rows = map(lambda x: self._update_row(x, distributions), standardized_rows)
                writer.writerows(updated_rows)

    def get_target_task(self):
        """Get the task whose output should be reformatted.

        Returns:
            Luigi task.
        """
        raise NotImplementedError('Must use implementor.')

    def get_filename(self):
        """Get the filename at which the reformatted data should be written.

        Returns:
            String filename (but not full path).
        """
        raise NotImplementedError('Must use implementor.')

    def _standardize_row(self, target):
        """Ensure an input row has expected fields and types.

        Args:
            target: The record to standardize.

        Returns:
            The record after standardization.
        """
        return {
            'geohash': target['geohash'],
            'simYear': int(target['simYear']),
            'joinYear': int(target['joinYear']),
            'predictedMean': float(target['predictedMean']),
            'predictedStd': float(target['predictedStd']),
            'predictedSkew': float(target['predictedSkew']),
            'predictedKurtosis': float(target['predictedKurtosis']),
            'yieldObservations': int(target['yieldObservations'])
        }

    def _update_row(self, row, distributions):
        """Unapply normalization to return to original human-interpretable values.

        Args:
            row: The row to update.
            distributions: The distribution information to use to unnoramlize data.

        Returns:
            Row after unnormalization.
        """

        def interpret(target, dist):
            if self._get_requires_interpretation():
                reverse_z = target * dist['std'] + dist['mean']
                return reverse_z
            else:
                return target

        def update_field(field):
            field_capitalized = field.capitalize()
            dist = distributions['yield%s' % field_capitalized]
            target = row['predicted%s' % field_capitalized]
            interpreted = interpret(target, dist)
            row['predicted%s' % field_capitalized] = interpreted

        update_field('mean')
        update_field('std')
        update_field('skew')
        update_field('kurtosis')

        return row

    def _get_requires_interpretation(self):
        """Determine if z score interpretation should be applied.

        Returns:
            True if z scores are being used as inputs and should be inverted or False otherwise.
        """
        return const.NORM_YIELD_FIELDS and not const.JIT_UNNORM_YIELD


class MakeSimulationTasksTemplate(luigi.Task):
    """Task to generate simulation task information."""

    def requires(self):
        """Indicate which tasks provide input data.

        Args:
            GetNumObservationsTask as well as the baseline and projection data tasks.
        """
        return {
            'baseline': self.get_baseline_task(),
            'projection': self.get_projection_task(),
            'numObservations': GetNumObservationsTask()
        }

    def output(self):
        """Determine where the simulation task information should be written.

        Returns:
            LocalTarget at which the simulation task information should be written.
        """
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        """Build the simulation tasks."""
        baseline_indexed = self._index_input('baseline', 0)
        projection_indexed = self._index_input(
            'projection',
            self._get_offset_years()
        )

        baseline_keys = set(baseline_indexed.keys())
        projection_keys = set(projection_indexed.keys())
        keys = baseline_keys.intersection(projection_keys)

        assert len(keys) > 0

        with self.input()['numObservations'].open('r') as f:
            rows = csv.DictReader(f)
            keyed = map(
                lambda x: ('%s.%s' % (x['geohash'], x['year']), int(x['yieldObservations'])),
                rows
            )
            observation_counts_indexed = dict(keyed)

        def get_num_observations(geohash, join_year):
            key = '%s.%d' % (geohash, join_year)
            return observation_counts_indexed.get(key, 0)

        def get_output_row(key):
            baseline_record = baseline_indexed[key]
            projection_record = projection_indexed[key]

            geohash = projection_record['geohash']
            join_year = projection_record['joinYear']

            return {
                'geohash': geohash,
                'year': projection_record['simYear'],
                'condition': self.get_condition(),
                'originalYieldMean': baseline_record['predictedMean'],
                'originalYieldStd': baseline_record['predictedStd'],
                'originalYieldSkew': baseline_record['predictedSkew'],
                'originalYieldKurtosis': baseline_record['predictedKurtosis'],
                'projectedYieldMean': projection_record['predictedMean'],
                'projectedYieldStd': projection_record['predictedStd'],
                'projectedYieldSkew': projection_record['predictedSkew'],
                'projectedYieldKurtosis': projection_record['predictedKurtosis'],
                'numObservations': get_num_observations(geohash, join_year)
            }

        output_rows_all = map(get_output_row, keys)
        output_rows = filter(lambda x: x['numObservations'] > 0, output_rows_all)

        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'geohash',
                'year',
                'condition',
                'originalYieldMean',
                'originalYieldStd',
                'originalYieldSkew',
                'originalYieldKurtosis',
                'projectedYieldMean',
                'projectedYieldStd',
                'projectedYieldSkew',
                'projectedYieldKurtosis',
                'numObservations'
            ])
            writer.writeheader()
            writer.writerows(output_rows)

    def get_filename(self):
        """Get the filename at which the task information should be written.

        Returns:
            Filename but not full path as string.
        """
        raise NotImplementedError('Use implementor.')

    def get_baseline_task(self):
        """Get the task whose output describes geohash-level yield baselines.

        Returns:
            Luigi task.
        """
        raise NotImplementedError('Use implementor.')

    def get_projection_task(self):
        """Get the task whose output describes geohash-level yield predictions.

        Returns:
            Luigi task.
        """
        raise NotImplementedError('Use implementor.')

    def get_condition(self):
        """Get the condition in which the predicted data were made.

        Returns:
            Name of condition as string like 2050_SSP245.
        """
        raise NotImplementedError('Use implementor.')

    def _index_input(self, name, year_offset):
        """Index one of the datasets by geohash and year.

        Args:
            name: The name of the dataset to index. This is one of the inputs in requires.
            year_offset: Offset to apply to join year.

        Returns:
            The dataset after indexing.
        """
        indexed = {}

        with self.input()[name].open('r') as f:
            rows_raw = csv.DictReader(f)
            rows = map(lambda x: self._parse_row(x), rows_raw)

            for row in rows:
                join_year = row['joinYear'] + year_offset
                row['joinYear'] = join_year
                key = '%s.%d' % (row['geohash'], join_year)
                indexed[key] = row

        return indexed

    def _parse_row(self, row):
        """Parse a raw input row.

        Args:
            row: The raw row to parse.

        Returns:
            The row with expected fields and data types.
        """
        return {
            'geohash': row['geohash'],
            'simYear': int(row['simYear']),
            'joinYear': int(row['joinYear']),
            'predictedMean': float(row['predictedMean']),
            'predictedStd': float(row['predictedStd']),
            'predictedSkew': float(row['predictedSkew']),
            'predictedKurtosis': float(row['predictedKurtosis']),
            'yieldObservations': int(row['yieldObservations'])
        }

    def _get_offset_years(self):
        """Get the number of years to offset the projection series.

        Returns:
            Number of years to offset.
        """
        return 0


class CheckUnitSizes(luigi.Task):
    """Check that the unit sizes input dataset is available."""

    def output(self):
        """Get the location at which the data are expected."""
        return luigi.LocalTarget(const.get_file_location('unit_sizes_2023.csv'))

    def run(self):
        """Execute the test (this should not run as the input file should alerady be present)."""
        raise RuntimeError('Expected unit_sizes.csv to be provided.')


class ExecuteSimulationTasksTemplate(luigi.Task):
    """Abstract base class (template class) for executing a set of simulations."""

    def requires(self):
        """Get the tasks whose outputs are required for running simulations.

        Returns:
            StartClusterTask, CheckUnitSizes, DetermineEquivalentStdTask, task to generate
            simulation task information and task to generate model residuals.
        """
        return {
            'tasks': self.get_tasks_task(),
            'deltas': self.get_deltas_task(),
            'cluster': cluster_tasks.StartClusterTask(),
            'unitSizes': CheckUnitSizes(),
            'stdThresholds': DetermineEquivalentStdTask()
        }

    def output(self):
        """Get the location at which the simulation outputs should be written.

        Returns:
            LocalTarget at which to write simulation outputs.
        """
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        """Run a set of simulations."""
        with self.input()['unitSizes'].open('r') as f:
            reader = csv.DictReader(f)
            values_pixels_str = map(lambda x: x['coveragePixels'], reader)
            values_pixels_int = map(lambda x: int(x), values_pixels_str)
            unit_sizes = list(values_pixels_int)

        with self.input()['tasks'].open('r') as f:
            rows = csv.DictReader(f)
            tasks = [parse_record_dict(x) for x in rows]

        job_shuffles = list(range(0, 100))
        input_records_grouped = toolz.itertoolz.groupby(
            lambda x: random.choice(job_shuffles),
            tasks
        )

        tasks_with_variations = list(
            itertools.product(
                input_records_grouped.values(),
                THRESHOLDS,
                STD_MULT,
                GEOHASH_SIZE,
                ['always', 'never']
            )
        )

        cluster = cluster_tasks.get_cluster()
        cluster.adapt(minimum=20, maximum=500)
        client = cluster.get_client()

        with self.input()['deltas'].open('r') as f:
            rows = csv.DictReader(f)
            test_rows = filter(lambda x: x['setAssign'] == 'test', rows)
            rows_mean_std_linear_str = map(
                lambda x: (
                    x['meanResidual'],
                    x['stdResidual'],
                    x['skewResidual'],
                    x['kurtosisResidual']
                ),
                test_rows
            )
            rows_mean_std_linear = map(
                lambda x: (float(x[0]), float(x[1]), float(x[2]), float(x[3])),
                rows_mean_std_linear_str
            )
            unzipped = list(zip(*rows_mean_std_linear))
            deltas = {
                'mean': unzipped[0],
                'std': unzipped[1],
                'skew': unzipped[2],
                'kurtosis': unzipped[3]
            }

        with self.input()['stdThresholds'].open('r') as f:
            std_thresholds = json.load(f)

        outputs_all = client.map(
            lambda x: run_simulation_set(
                x[0],
                deltas,
                x[1],
                x[2],
                x[3],
                x[4],
                unit_sizes,
                std_thresholds
            ),
            tasks_with_variations
        )
        outputs_realized = map(lambda x: x.result(), outputs_all)

        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()

            for output_set in outputs_realized:
                writer.writerows(output_set)
                f.flush()

    def get_tasks_task(self):
        """Get the simulation task information generation task.

        Returns:
            Luigi task.
        """
        raise NotImplementedError('Use implementor.')

    def get_filename(self):
        """Get the filename at which the simulation outputs should be written.

        Returns:
            String filename (not path).
        """
        raise NotImplementedError('Use implementor.')

    def get_deltas_task(self):
        """Get the task whose output are the model residuals.

        Returns:
            Luigi task.
        """
        raise NotImplementedError('Use implementor.')


class ProjectHistoricTask(luigi.Task):
    """Task in which historic data are retroactively predicted for reference."""

    def requires(self):
        """Get the task whose outputs should be used as inputs to the neural network.

        Returns:
            NormalizeRefHistoricTrainingFrameTask
        """
        return {
            'target': NormalizeRefHistoricTrainingFrameTask()
        }

    def output(self):
        """Determine where the retroactive predictions should be written.

        Returns:
            LocalTarget at which the predictions should be written.
        """
        return luigi.LocalTarget(const.get_file_location('historic_project_dist.csv'))

    def run(self):
        """Project into the historic dataset."""
        target_frame = pandas.read_csv(self.input()['target'].path)

        target_frame['joinYear'] = target_frame['year']
        target_frame['simYear'] = 2007
        target_frame['year'] = target_frame['simYear']

        target_frame['predictedMean'] = target_frame['yieldMean']
        target_frame['predictedStd'] = target_frame['yieldStd']
        target_frame['predictedSkew'] = target_frame['yieldSkew']
        target_frame['predictedKurtosis'] = target_frame['yieldKurtosis']

        target_frame[[
            'geohash',
            'simYear',
            'joinYear',
            'predictedMean',
            'predictedStd',
            'predictedSkew',
            'predictedKurtosis',
            'yieldObservations'
        ]].to_csv(self.output().path)


class ProjectHistoricEarlyTask(luigi.Task):
    """Task in which historic data are retroactively predicted for reference."""

    def requires(self):
        """Get the task whose outputs should be used as inputs to the neural network.

        Returns:
            NormalizeRefHistoricTrainingFrameTask
        """
        return {
            'target': NormalizeRefHistoricEarlyTask()
        }

    def output(self):
        """Determine where the retroactive predictions should be written.

        Returns:
            LocalTarget at which the predictions should be written.
        """
        return luigi.LocalTarget(const.get_file_location('historic_project_dist_early.csv'))

    def run(self):
        """Project into the historic dataset."""
        target_frame = pandas.read_csv(self.input()['target'].path)

        target_frame['joinYear'] = target_frame['year']
        target_frame['simYear'] = 2000
        target_frame['year'] = target_frame['simYear']

        target_frame['predictedMean'] = target_frame['yieldMean']
        target_frame['predictedStd'] = target_frame['yieldStd']
        target_frame['predictedSkew'] = target_frame['yieldSkew']
        target_frame['predictedKurtosis'] = target_frame['yieldKurtosis']

        target_frame[[
            'geohash',
            'simYear',
            'joinYear',
            'predictedMean',
            'predictedStd',
            'predictedSkew',
            'predictedKurtosis',
            'yieldObservations'
        ]].to_csv(self.output().path)


class ProjectHistoricLateTask(luigi.Task):
    """Task in which historic data are retroactively predicted for reference."""

    def requires(self):
        """Get the task whose outputs should be used as inputs to the neural network.

        Returns:
            NormalizeRefHistoricTrainingFrameTask
        """
        return {
            'target': NormalizeRefHistoricLateTask()
        }

    def output(self):
        """Determine where the retroactive predictions should be written.

        Returns:
            LocalTarget at which the predictions should be written.
        """
        return luigi.LocalTarget(const.get_file_location('historic_project_dist_late.csv'))

    def run(self):
        """Project into the historic dataset."""
        target_frame = pandas.read_csv(self.input()['target'].path)

        target_frame['joinYear'] = target_frame['year']
        target_frame['simYear'] = 2010
        target_frame['year'] = target_frame['simYear']

        target_frame['predictedMean'] = target_frame['yieldMean']
        target_frame['predictedStd'] = target_frame['yieldStd']
        target_frame['predictedSkew'] = target_frame['yieldSkew']
        target_frame['predictedKurtosis'] = target_frame['yieldKurtosis']

        target_frame[[
            'geohash',
            'simYear',
            'joinYear',
            'predictedMean',
            'predictedStd',
            'predictedSkew',
            'predictedKurtosis',
            'yieldObservations'
        ]].to_csv(self.output().path)


class ProjectHistoricModelTask(ProjectTaskTemplate):
    """Retroactively project historic yield."""

    def get_target_task(self):
        """Get the task whose output should be used as model inputs.

        Returns:
            Luigi task.
        """
        return NormalizeRefHistoricTrainingFrameTask()

    def get_base_year(self):
        """Get the "center" year of the series to be predicted.

        Returns:
            Integer year like 2007, 2030, or 2050.
        """
        return 2007

    def get_filename(self):
        """Get the filename at which the projections should be written.

        Returns:
            Filename (not full path) as string.
        """
        return 'historic_project_dist_model.csv'


class Project2030Task(ProjectTaskTemplate):
    """Project yield for 2030."""

    def get_target_task(self):
        """Get the task whose output should be used as model inputs.

        Returns:
            Luigi task.
        """
        return normalize_tasks.NormalizeFutureTrainingFrameTask(condition='2030_SSP245')

    def get_base_year(self):
        """Get the "center" year of the series to be predicted.

        Returns:
            Integer year like 2007, 2030, or 2050.
        """
        return 2030

    def get_filename(self):
        """Get the filename at which the projections should be written.

        Returns:
            Filename (not full path) as string.
        """
        return '2030_project_dist.csv'


class Project2030HoldYearTask(ProjectTaskTemplate):
    """Project yield for 2030 without advancing the year."""

    def get_target_task(self):
        """Get the task whose output should be used as model inputs.

        Returns:
            Luigi task.
        """
        return normalize_tasks.NormalizeFutureTrainingFrameTask(condition='2030_SSP245')

    def get_base_year(self):
        """Get the "center" year of the series to be predicted.

        Returns:
            Integer year like 2007, 2030, or 2050.
        """
        return 2007

    def get_filename(self):
        """Get the filename at which the projections should be written.

        Returns:
            Filename (not full path) as string.
        """
        return '2030_project_dist_no_year.csv'


class Project2030CounterfactualTask(ProjectTaskTemplate):
    """Project yield for 2030 without additional warming."""

    def get_target_task(self):
        """Get the task whose output should be used as model inputs.

        Returns:
            Luigi task.
        """
        return NormalizeRefHistoricTrainingFrameTask()

    def get_base_year(self):
        """Get the "center" year of the series to be predicted.

        Returns:
            Integer year like 2007, 2030, or 2050.
        """
        return 2030

    def get_filename(self):
        """Get the filename at which the projections should be written.

        Returns:
            Filename (not full path) as string.
        """
        return '2030_project_dist_counterfactual.csv'


class Project2050Task(ProjectTaskTemplate):
    """Project yield for 2050."""

    def get_target_task(self):
        """Get the task whose output should be used as model inputs.

        Returns:
            Luigi task.
        """
        return normalize_tasks.NormalizeFutureTrainingFrameTask(condition='2050_SSP245')

    def get_base_year(self):
        """Get the "center" year of the series to be predicted.

        Returns:
            Integer year like 2007, 2030, or 2050.
        """
        return 2050

    def get_filename(self):
        """Get the filename at which the projections should be written.

        Returns:
            Filename (not full path) as string.
        """
        return '2050_project_dist.csv'


class Project2050HoldYearTask(ProjectTaskTemplate):
    """Project yield for 2050 without changing the year."""

    def get_target_task(self):
        """Get the task whose output should be used as model inputs.

        Returns:
            Luigi task.
        """
        return normalize_tasks.NormalizeFutureTrainingFrameTask(condition='2050_SSP245')

    def get_base_year(self):
        """Get the "center" year of the series to be predicted.

        Returns:
            Integer year like 2007, 2030, or 2050.
        """
        return 2007

    def get_filename(self):
        """Get the filename at which the projections should be written.

        Returns:
            Filename (not full path) as string.
        """
        return '2050_project_dist_hold_year.csv'


class Project2050CounterfactualTask(ProjectTaskTemplate):
    """Project yield for 2050 without additional warming."""

    def get_target_task(self):
        """Get the task whose output should be used as model inputs.

        Returns:
            Luigi task.
        """
        return NormalizeRefHistoricTrainingFrameTask()

    def get_base_year(self):
        """Get the "center" year of the series to be predicted.

        Returns:
            Integer year like 2007, 2030, or 2050.
        """
        return 2050

    def get_filename(self):
        """Get the filename at which the projections should be written.

        Returns:
            Filename (not full path) as string.
        """
        return '2050_project_dist_counterfactual.csv'


class InterpretProjectHistoricTask(InterpretProjectTaskTemplate):
    """Interpret retroactive historic projections."""

    def get_target_task(self):
        """Get the task whose output should be reformatted.

        Returns:
            Luigi task.
        """
        return ProjectHistoricTask()

    def get_filename(self):
        """Get the filename at which the reformatted data should be written.

        Returns:
            String filename (but not full path).
        """
        return 'historic_project_dist_interpret.csv'

    def _get_requires_interpretation(self):
        """Indicate that, if there is JIT, it has to apply to historic norms.

        Indicate that, if there is JIT, it has to apply to historic norms as the model will not run
        on those inputs to apply the JIT z score transformation.

        Returns:
            True if JIT active and false otherwise.
        """
        return const.JIT_UNNORM_YIELD


class InterpretProjectHistoricEarlyTask(InterpretProjectTaskTemplate):
    """Interpret retroactive historic projections with early years."""

    def get_target_task(self):
        """Get the task whose output should be reformatted.

        Returns:
            Luigi task.
        """
        return ProjectHistoricEarlyTask()

    def get_filename(self):
        """Get the filename at which the reformatted data should be written.

        Returns:
            String filename (but not full path).
        """
        return 'historic_project_dist_interpret_early.csv'

    def _get_requires_interpretation(self):
        """Indicate that, if there is JIT, it has to apply to historic norms.

        Indicate that, if there is JIT, it has to apply to historic norms as the model will not run
        on those inputs to apply the JIT z score transformation.

        Returns:
            True if JIT active and false otherwise.
        """
        return const.JIT_UNNORM_YIELD


class InterpretProjectHistoricLateTask(InterpretProjectTaskTemplate):
    """Interpret retroactive historic projections with late years."""

    def get_target_task(self):
        """Get the task whose output should be reformatted.

        Returns:
            Luigi task.
        """
        return ProjectHistoricLateTask()

    def get_filename(self):
        """Get the filename at which the reformatted data should be written.

        Returns:
            String filename (but not full path).
        """
        return 'historic_project_dist_interpret_late.csv'

    def _get_requires_interpretation(self):
        """Indicate that, if there is JIT, it has to apply to historic norms.

        Indicate that, if there is JIT, it has to apply to historic norms as the model will not run
        on those inputs to apply the JIT z score transformation.

        Returns:
            True if JIT active and false otherwise.
        """
        return const.JIT_UNNORM_YIELD


class InterpretProject2030Task(InterpretProjectTaskTemplate):
    """Interpret 2030 projections."""

    def get_target_task(self):
        """Get the task whose output should be reformatted.

        Returns:
            Luigi task.
        """
        return Project2030Task()

    def get_filename(self):
        """Get the filename at which the reformatted data should be written.

        Returns:
            String filename (but not full path).
        """
        return '2030_project_dist_interpret.csv'


class InterpretProject2030HoldYearTask(InterpretProjectTaskTemplate):
    """Interpret 2030 projections without changing the year."""

    def get_target_task(self):
        """Get the task whose output should be reformatted.

        Returns:
            Luigi task.
        """
        return Project2030HoldYearTask()

    def get_filename(self):
        """Get the filename at which the reformatted data should be written.

        Returns:
            String filename (but not full path).
        """
        return '2030_project_dist_interpret_hold_year.csv'


class InterpretProject2030CounterfactualTask(InterpretProjectTaskTemplate):
    """Interpret 2030 projections without further warming."""

    def get_target_task(self):
        """Get the task whose output should be reformatted.

        Returns:
            Luigi task.
        """
        return Project2030CounterfactualTask()

    def get_filename(self):
        """Get the filename at which the reformatted data should be written.

        Returns:
            String filename (but not full path).
        """
        return '2030_project_dist_counterfactual_interpret.csv'


class InterpretProject2050Task(InterpretProjectTaskTemplate):
    """Interpret 2050 projections."""

    def get_target_task(self):
        """Get the task whose output should be reformatted.

        Returns:
            Luigi task.
        """
        return Project2050Task()

    def get_filename(self):
        """Get the filename at which the reformatted data should be written.

        Returns:
            String filename (but not full path).
        """
        return '2050_project_dist_interpret.csv'


class InterpretProject2050HoldYearTask(InterpretProjectTaskTemplate):
    """Interpret 2050 projections without changing years."""

    def get_target_task(self):
        """Get the task whose output should be reformatted.

        Returns:
            Luigi task.
        """
        return Project2050HoldYearTask()

    def get_filename(self):
        """Get the filename at which the reformatted data should be written.

        Returns:
            String filename (but not full path).
        """
        return '2050_project_dist_interpret_hold_year.csv'


class InterpretProject2050CounterfactualTask(InterpretProjectTaskTemplate):
    """Interpret 2030 projections with no futher warming."""

    def get_target_task(self):
        """Get the task whose output should be reformatted.

        Returns:
            Luigi task.
        """
        return Project2050CounterfactualTask()

    def get_filename(self):
        """Get the filename at which the reformatted data should be written.

        Returns:
            String filename (but not full path).
        """
        return '2050_project_dist_counterfactual_interpret.csv'


class MakeSimulationTasksHistoricTask(MakeSimulationTasksTemplate):
    """Make simulation tasks for retroactive prediction."""

    def get_filename(self):
        """Get the filename at which the task information should be written.

        Returns:
            Filename but not full path as string.
        """
        return 'current_sim_tasks.csv'

    def get_baseline_task(self):
        """Get the task whose output describes geohash-level yield baselines.

        Returns:
            Luigi task.
        """
        return InterpretProjectHistoricTask()

    def get_projection_task(self):
        """Get the task whose output describes geohash-level yield predictions.

        Returns:
            Luigi task.
        """
        return InterpretProjectHistoricTask()

    def get_condition(self):
        """Get the condition in which the predicted data were made.

        Returns:
            Name of condition as string like 2050_SSP245.
        """
        return 'historic'


class MakeSimulationTasks2010Task(MakeSimulationTasksTemplate):
    """Make simulation tasks for 2010 prediction."""

    def get_filename(self):
        """Get the filename at which the task information should be written.

        Returns:
            Filename but not full path as string.
        """
        return '2010_sim_tasks.csv'

    def get_baseline_task(self):
        """Get the task whose output describes geohash-level yield baselines.

        Returns:
            Luigi task.
        """
        return InterpretProjectHistoricEarlyTask()

    def get_projection_task(self):
        """Get the task whose output describes geohash-level yield predictions.

        Returns:
            Luigi task.
        """
        return InterpretProjectHistoricLateTask()

    def get_condition(self):
        """Get the condition in which the predicted data were made.

        Returns:
            Name of condition as string like 2050_SSP245.
        """
        return '2010_Historic'

    def _get_offset_years(self):
        """Indicate that we are joining by offseting the join year by 10.

        Returns:
            Offset to apply to the InterpretProjectHistoricLateTask output.
        """
        return -10


class MakeSimulationTasks2030Task(MakeSimulationTasksTemplate):
    """Make simulation tasks for 2030 prediction."""

    def get_filename(self):
        """Get the filename at which the task information should be written.

        Returns:
            Filename but not full path as string.
        """
        return '2030_sim_tasks.csv'

    def get_baseline_task(self):
        """Get the task whose output describes geohash-level yield baselines.

        Returns:
            Luigi task.
        """
        return InterpretProjectHistoricTask()

    def get_projection_task(self):
        """Get the task whose output describes geohash-level yield predictions.

        Returns:
            Luigi task.
        """
        return InterpretProject2030Task()

    def get_condition(self):
        """Get the condition in which the predicted data were made.

        Returns:
            Name of condition as string like 2050_SSP245.
        """
        return '2030_SSP245'


class MakeSimulationTasks2030HoldYearTask(MakeSimulationTasksTemplate):
    """Make simulation tasks for 2030 prediction without incrementing years."""

    def get_filename(self):
        """Get the filename at which the task information should be written.

        Returns:
            Filename but not full path as string.
        """
        return '2030_sim_tasks_hold_year.csv'

    def get_baseline_task(self):
        """Get the task whose output describes geohash-level yield baselines.

        Returns:
            Luigi task.
        """
        return InterpretProjectHistoricTask()

    def get_projection_task(self):
        """Get the task whose output describes geohash-level yield predictions.

        Returns:
            Luigi task.
        """
        return InterpretProject2030HoldYearTask()

    def get_condition(self):
        """Get the condition in which the predicted data were made.

        Returns:
            Name of condition as string like 2050_SSP245.
        """
        return '2030_SSP245'


class MakeSimulationTasks2050Task(MakeSimulationTasksTemplate):
    """Make simulation tasks for 2050 prediction."""

    def get_filename(self):
        """Get the filename at which the task information should be written.

        Returns:
            Filename but not full path as string.
        """
        return '2050_sim_tasks.csv'

    def get_baseline_task(self):
        """Get the task whose output describes geohash-level yield baselines.

        Returns:
            Luigi task.
        """
        return InterpretProject2030Task()

    def get_projection_task(self):
        """Get the task whose output describes geohash-level yield predictions.

        Returns:
            Luigi task.
        """
        return InterpretProject2050Task()

    def get_condition(self):
        """Get the condition in which the predicted data were made.

        Returns:
            Name of condition as string like 2050_SSP245.
        """
        return '2050_SSP245'


class MakeSimulationTasks2050HoldYearTask(MakeSimulationTasksTemplate):
    """Make simulation tasks for 2050 prediction without changing years."""

    def get_filename(self):
        """Get the filename at which the task information should be written.

        Returns:
            Filename but not full path as string.
        """
        return '2050_sim_tasks_hold_year.csv'

    def get_baseline_task(self):
        """Get the task whose output describes geohash-level yield baselines.

        Returns:
            Luigi task.
        """
        return InterpretProject2030HoldYearTask()

    def get_projection_task(self):
        """Get the task whose output describes geohash-level yield predictions.

        Returns:
            Luigi task.
        """
        return InterpretProject2050HoldYearTask()

    def get_condition(self):
        """Get the condition in which the predicted data were made.

        Returns:
            Name of condition as string like 2050_SSP245.
        """
        return '2050_SSP245'


class MakeSimulationTasks2030CounterfactualTask(MakeSimulationTasksTemplate):
    """Make simulation tasks for 2030 prediction without further warming."""

    def get_filename(self):
        """Get the filename at which the task information should be written.

        Returns:
            Filename but not full path as string.
        """
        return '2030_sim_tasks_counterfactual.csv'

    def get_baseline_task(self):
        """Get the task whose output describes geohash-level yield baselines.

        Returns:
            Luigi task.
        """
        return InterpretProjectHistoricTask()

    def get_projection_task(self):
        """Get the task whose output describes geohash-level yield predictions.

        Returns:
            Luigi task.
        """
        return InterpretProject2030CounterfactualTask()

    def get_condition(self):
        """Get the condition in which the predicted data were made.

        Returns:
            Name of condition as string like 2050_SSP245.
        """
        return '2030_SSP245'


class MakeSimulationTasks2050CounterfactualTask(MakeSimulationTasksTemplate):
    """Make simulation tasks for 2050 prediction without further warming."""

    def get_filename(self):
        """Get the filename at which the task information should be written.

        Returns:
            Filename but not full path as string.
        """
        return '2050_sim_tasks_counterfactual.csv'

    def get_baseline_task(self):
        """Get the task whose output describes geohash-level yield baselines.

        Returns:
            Luigi task.
        """
        return InterpretProject2030CounterfactualTask()

    def get_projection_task(self):
        """Get the task whose output describes geohash-level yield predictions.

        Returns:
            Luigi task.
        """
        return InterpretProject2050CounterfactualTask()

    def get_condition(self):
        """Get the condition in which the predicted data were made.

        Returns:
            Name of condition as string like 2050_SSP245.
        """
        return '2030_SSP245'


class ExecuteSimulationTasksHistoricPredictedTask(ExecuteSimulationTasksTemplate):
    """Execute simulation for historic retroactive projection."""

    def get_tasks_task(self):
        """Get the simulation task information generation task.

        Returns:
            Luigi task.
        """
        return MakeSimulationTasksHistoricTask()

    def get_filename(self):
        """Get the filename at which the simulation outputs should be written.

        Returns:
            String filename (not path).
        """
        return 'historic_sim.csv'

    def get_deltas_task(self):
        """Get the task whose output are the model residuals.

        Returns:
            Luigi task.
        """
        return selection_tasks.PostHocTestRawDataTemporalResidualsTask()

    def get_sample_model_residuals(self):
        """Determine if model residuals should be sampled and applied to simulation outputs.

        Returns:
            True if model residuals should be sampled and false otherwise.
        """
        return False  # This one is not predicted


class ExecuteSimulationTasks2010PredictedTask(ExecuteSimulationTasksTemplate):
    """Execute simulation for 2010 projection."""

    def get_tasks_task(self):
        """Get the simulation task information generation task.

        Returns:
            Luigi task.
        """
        return MakeSimulationTasks2010Task()

    def get_filename(self):
        """Get the filename at which the simulation outputs should be written.

        Returns:
            String filename (not path).
        """
        return '2010_sim.csv'

    def get_deltas_task(self):
        """Get the task whose output are the model residuals.

        Returns:
            Luigi task.
        """
        return selection_tasks.PostHocTestRawDataTemporalResidualsTask()


class ExecuteSimulationTasks2030PredictedTask(ExecuteSimulationTasksTemplate):
    """Execute simulation for 2030 projection."""

    def get_tasks_task(self):
        """Get the simulation task information generation task.

        Returns:
            Luigi task.
        """
        return MakeSimulationTasks2030Task()

    def get_filename(self):
        """Get the filename at which the simulation outputs should be written.

        Returns:
            String filename (not path).
        """
        return '2030_sim.csv'

    def get_deltas_task(self):
        """Get the task whose output are the model residuals.

        Returns:
            Luigi task.
        """
        return selection_tasks.PostHocTestRawDataTemporalResidualsTask()


class ExecuteSimulationTasks2030PredictedHoldYearTask(ExecuteSimulationTasksTemplate):
    """Execute simulation for 2030 projection without incrementing years."""

    def get_tasks_task(self):
        """Get the simulation task information generation task.

        Returns:
            Luigi task.
        """
        return MakeSimulationTasks2030HoldYearTask()

    def get_filename(self):
        """Get the filename at which the simulation outputs should be written.

        Returns:
            String filename (not path).
        """
        return '2030_sim_hold_year.csv'

    def get_deltas_task(self):
        """Get the task whose output are the model residuals.

        Returns:
            Luigi task.
        """
        return selection_tasks.PostHocTestRawDataTemporalResidualsTask()


class ExecuteSimulationTasks2050PredictedTask(ExecuteSimulationTasksTemplate):
    """Execute simulation for 2050 projection."""

    def get_tasks_task(self):
        """Get the simulation task information generation task.

        Returns:
            Luigi task.
        """
        return MakeSimulationTasks2050Task()

    def get_filename(self):
        """Get the filename at which the simulation outputs should be written.

        Returns:
            String filename (not path).
        """
        return '2050_sim.csv'

    def get_deltas_task(self):
        """Get the task whose output are the model residuals.

        Returns:
            Luigi task.
        """
        return selection_tasks.PostHocTestRawDataTemporalResidualsTask()


class ExecuteSimulationTasks2050PredictedHoldYearTask(ExecuteSimulationTasksTemplate):
    """Execute simulation for 2050 projection without changing years."""

    def get_tasks_task(self):
        """Get the simulation task information generation task.

        Returns:
            Luigi task.
        """
        return MakeSimulationTasks2050HoldYearTask()

    def get_filename(self):
        """Get the filename at which the simulation outputs should be written.

        Returns:
            String filename (not path).
        """
        return '2050_sim_hold_year.csv'

    def get_deltas_task(self):
        """Get the task whose output are the model residuals.

        Returns:
            Luigi task.
        """
        return selection_tasks.PostHocTestRawDataTemporalResidualsTask()


class ExecuteSimulationTasks2030Counterfactual(ExecuteSimulationTasksTemplate):
    """Execute simulation for 2030 projection without further warming."""

    def get_tasks_task(self):
        """Get the simulation task information generation task.

        Returns:
            Luigi task.
        """
        return MakeSimulationTasks2030CounterfactualTask()

    def get_filename(self):
        """Get the filename at which the simulation outputs should be written.

        Returns:
            String filename (not path).
        """
        return '2030_sim_counterfactual.csv'

    def get_deltas_task(self):
        """Get the task whose output are the model residuals.

        Returns:
            Luigi task.
        """
        return selection_tasks.PostHocTestRawDataTemporalResidualsTask()


class ExecuteSimulationTasks2050Counterfactual(ExecuteSimulationTasksTemplate):
    """Execute simulation for 2050 projection without further warming."""

    def get_tasks_task(self):
        """Get the simulation task information generation task.

        Returns:
            Luigi task.
        """
        return MakeSimulationTasks2050CounterfactualTask()

    def get_filename(self):
        """Get the filename at which the simulation outputs should be written.

        Returns:
            String filename (not path).
        """
        return '2050_sim_counterfactual.csv'

    def get_deltas_task(self):
        """Get the task whose output are the model residuals.

        Returns:
            Luigi task.
        """
        return selection_tasks.PostHocTestRawDataTemporalResidualsTask()


class CombineSimulationsTaskTemplate(luigi.Task):
    """Combine all simulations into a single data file."""

    def requires(self):
        """Get the listing of all simulations to be concatenated.

        Returns:
            All simulations to be combined.
        """
        raise NotImplementedError('Use implementor.')

    def output(self):
        """Get the location at which concatenated outputs should be written.

        Returns:
            LocalTarget at which concatenated outputs should be written.
        """
        return NotImplementedError('Use implementor.')

    def _write_out(self, label, writer):
        """Write out a set of simulation results.

        Args:
            label: The label for the series as a string matching the input dictionary.
            writer: The writer through which the simulations should be written.
        """
        with self.input()[label].open('r') as f:
            reader = csv.DictReader(f)
            rows = map(lambda x: self._add_series(label, x), reader)
            writer.writerows(rows)

    def _add_series(self, series, row):
        """Add a series label to an output row.

        Args:
            series: Series label as a string.
            row: The row in which the series label should be added.

        Returns:
            Input row after adding series.
        """
        row['series'] = series
        return row


class CombineSimulationsTask(CombineSimulationsTaskTemplate):
    """Combine all simulations into a single data file."""

    def requires(self):
        """Get the listing of all simulations to be concatenated.

        Returns:
            All simulations to be combined.
        """
        return {
            'historic': ExecuteSimulationTasksHistoricPredictedTask(),
            '2010': ExecuteSimulationTasks2010PredictedTask(),
            '2030': ExecuteSimulationTasks2030PredictedTask(),
            '2030_counterfactual': ExecuteSimulationTasks2030Counterfactual(),
            '2050': ExecuteSimulationTasks2050PredictedTask(),
            '2050_counterfactual': ExecuteSimulationTasks2050Counterfactual()
        }

    def output(self):
        """Get the location at which concatenated outputs should be written.

        Returns:
            LocalTarget at which concatenated outputs should be written.
        """
        return luigi.LocalTarget(const.get_file_location('sim_combined.csv'))

    def run(self):
        """Combine simulation outputs."""
        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['series'] + OUTPUT_FIELDS)
            writer.writeheader()
            self._write_out('historic', writer)
            self._write_out('2010', writer)
            self._write_out('2030', writer)
            self._write_out('2030_counterfactual', writer)
            self._write_out('2050', writer)
            self._write_out('2050_counterfactual', writer)


class CombineSimulationsHoldYearTask(CombineSimulationsTaskTemplate):
    """Combine all simulations into a single data file without incrementing year."""

    def requires(self):
        """Get the listing of all simulations to be concatenated.

        Returns:
            All simulations to be combined.
        """
        return {
            'historic': ExecuteSimulationTasksHistoricPredictedTask(),
            '2030': ExecuteSimulationTasks2030PredictedHoldYearTask(),
            '2030_counterfactual': ExecuteSimulationTasks2030Counterfactual(),
            '2050': ExecuteSimulationTasks2050PredictedHoldYearTask(),
            '2050_counterfactual': ExecuteSimulationTasks2050Counterfactual()
        }

    def output(self):
        """Get the location at which concatenated outputs should be written.

        Returns:
            LocalTarget at which concatenated outputs should be written.
        """
        return luigi.LocalTarget(const.get_file_location('sim_combined_hold_year.csv'))

    def run(self):
        """Combine simulation outputs."""
        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['series'] + OUTPUT_FIELDS)
            writer.writeheader()
            self._write_out('historic', writer)
            self._write_out('2030', writer)
            self._write_out('2030_counterfactual', writer)
            self._write_out('2050', writer)
            self._write_out('2050_counterfactual', writer)


class DetermineEquivalentStdTask(luigi.Task):
    """Get an equivalent standard deviation-based threshold.

    Determine a standard deviation-based threshold with equivalent system-wide aggregate coverage
    as an average-based threshold.
    """

    def requires(self):
        """Get the projections or projection-like frame in which to calculate this equivalency.

        Returns:
            InterpretProjectHistoricTask
        """
        return InterpretProjectHistoricTask()

    def output(self):
        """Get the location at which the calculation results should be written.

        Returns:
            LocalTarget at which to write the calculation results.
        """
        return luigi.LocalTarget(const.get_file_location('stats_equivalent_raw.json'))

    def run(self):
        """Calculate the standard deviation threshold."""
        def execute_threshold(level):
            with self.input().open() as f:
                reader = csv.DictReader(f)
                equivalencies = map(lambda x: {
                    'equivalent': self._get_equivalent_std(x, level),
                    'num': int(x['yieldObservations'])
                }, reader)

                equivalencies_valid = filter(
                    lambda x: x['equivalent'] is not None,
                    equivalencies
                )

                def combine(a, b):
                    total = a['num'] + b['num']
                    unnorm = a['equivalent'] * a['num'] + b['equivalent'] * b['num']
                    return {
                        'equivalent': unnorm / total,
                        'num': total
                    }

                overall_equivalency = functools.reduce(combine, equivalencies_valid)['equivalent']

                return overall_equivalency

        with self.output().open('w') as f:
            json.dump({
                '0.25': execute_threshold(0.25),
                '0.15': execute_threshold(0.15)
            }, f)

    def _get_equivalent_std(self, target, level):
        """Get the equivalent standard deviation threshold for a single record.

        Args:
            target: Record for which a standard deviation threshold should be generated.
            level: The average-based level for which a standard deviation-based equivalent should be
                found.

        Returns:
            Equivalent standard deviation threshold.
        """
        predicted_std = float(target['predictedStd'])

        if predicted_std > 0:
            return level / predicted_std
        else:
            return None


class DetermineEquivalentStdExtendedTask(luigi.Task):
    """Determine the standard deviation equivalent for a series of average-based thresholds."""

    def requires(self):
        """Get the task from which a standard devivation equivalent threshold should be calculated.

        Returns:
            InterpretProjectHistoricTask
        """
        return InterpretProjectHistoricTask()

    def output(self):
        """Determine where information about the equivalent thresholds should be written.

        Returns:
            LocalTarget at which the equivalent thresholds should be written as JSON.
        """
        return luigi.LocalTarget(const.get_file_location('stats_equivalent_raw_extended.json'))

    def run(self):
        """Find a set of equivalent standard deviation-based thresholds."""

        def execute_threshold(level):
            with self.input().open() as f:
                reader = csv.DictReader(f)
                equivalencies = map(lambda x: {
                    'equivalent': self._get_equivalent_std(x, level),
                    'num': int(x['yieldObservations'])
                }, reader)

                equivalencies_valid = filter(
                    lambda x: x['equivalent'] is not None,
                    equivalencies
                )

                def combine(a, b):
                    total = a['num'] + b['num']
                    unnorm = a['equivalent'] * a['num'] + b['equivalent'] * b['num']
                    return {
                        'equivalent': unnorm / total,
                        'num': total
                    }

                overall_equivalency = functools.reduce(combine, equivalencies_valid)['equivalent']

                return overall_equivalency

        with self.output().open('w') as f:
            ret_dict = {}
            for threshold in range(15, 55, 5):
                ret_dict[threshold / 100] = execute_threshold(threshold / 100)
            json.dump(ret_dict, f)

    def _get_equivalent_std(self, target, level):
        """Find an individual equivalent standard deviation-based threshold.

        Args:
            target: The record for which the standard deviation-based threshold should be
                determined.
            level: The average-based threshold for which an equivalency should be calculated like
                0.15 for 15% below average.

        Returns:
            Equivalent number of standard deviations below average.
        """
        predicted_std = float(target['predictedStd'])

        if predicted_std > 0:
            return level / predicted_std
        else:
            return None
