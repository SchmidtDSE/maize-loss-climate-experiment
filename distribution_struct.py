"""Generic mathematical utilities.

Generic mathematical utilities useful in representing and working with distributions. However,
unlike data_struct, these are not task-specific.

License:
    BSD
"""
import math


class Distribution:
    """Record describing a distribution of real numbers."""

    def __init__(self, mean, std, count, dist_min=None, dist_max=None, skew=None, kurtosis=None):
        """Create a new distribution structure.

        Args:
            mean: The average of the distribution.
            std: The standard deviation of the distribution.
            count: The number of observations in the sample.
            dist_min: The minimum value if known or None if not known.
            dist_max: The maximum value if known or None if not known.
            skew: Measure of skew for how uncentered the distribution is or None if not known.
            kurtosis: Measure of the kurtosis (tail shape) for the distribution or None if not
                known.
        """
        self._mean = mean
        self._std = std
        self._count = count
        self._dist_min = dist_min
        self._dist_max = dist_max
        self._skew = skew
        self._kurtosis = kurtosis

    def get_mean(self):
        """Get the distribution mean.

        Returns:
            The average of the distribution.
        """
        return self._mean

    def get_std(self):
        """Get the distribution standard deviation.

        Returns:
            The standard deviation of the distribution.
        """
        return self._std

    def get_count(self):
        """Get the sample size.

        Returns:
            The number of observations in the sample.
        """
        return self._count

    def get_min(self):
        """Get the minimum value of the distribution or sample if known.

        Returns:
            The minimum value if known or None if not known.
        """
        return self._dist_min

    def get_max(self):
        """Get the maximum value of the distribution or sample if known.

        Returns:
            The maximum value if known or None if not known.
        """
        return self._dist_max

    def get_skew(self):
        """Get a measure of distribution skew.

        Returns:
            Measure of skew for how uncentered the distribution is or None if not known.
        """
        return self._skew

    def get_kurtosis(self):
        """Get a measure of distribution kurtosis.

        Returns:
            Measure of the kurtosis (tail shape) for the distribution or None if not known.
        """
        return self._kurtosis
    
    def get_is_approx_normal(self):
        """Determine if the distribution is approximately normal.

        Returns:
            True if approximately normal and False otherwise.
        """
        return abs(self.get_skew()) > 2 and abs(self.get_kurtosis()) < 7

    def combine(self, other, allow_multiple_shapes=False):
        """Combine the samples from two different distributions.

        Args:
            other: The distribution to add to this one.
            allow_multiple_shapes: Flag indicating if combining multiple shapes is OK. If False,
                will raise an exception if either self or other are not approximately normal.

        Returns:
            Combined distributions.
        """
        new_count = self.get_count() + other.get_count()

        self_count = self.get_count()
        other_count = other.get_count()

        def get_weighted_average(self_val, other_val):
            self_weighted = self_val * self_count
            other_weighted = other_val * other_count
            pooled_weighted = self_weighted + other_weighted
            return pooled_weighted / (self_count + other_count)

        new_mean = get_weighted_average(self.get_mean(), other.get_mean())

        self_mean_weight = self.get_mean() * self.get_count()
        other_mean_weight = other.get_mean() * other.get_count()
        new_mean = (self_mean_weight + other_mean_weight) / new_count

        def get_variance_piece(target):
            return (target.get_count() - 1) * (target.get_std() ** 2)

        self_variance_piece = get_variance_piece(self)
        other_variance_piece = get_variance_piece(other)
        combined_variance_pieces = self_variance_piece + other_variance_piece
        combined_counts_adj = new_count - 2
        new_std = math.sqrt(combined_variance_pieces / combined_counts_adj)

        self_min = self.get_min()
        other_min = other.get_min()
        if self_min is None or other_min is None:
            new_min = None
        else:
            new_min = min([self_min, other_min])

        self_max = self.get_min()
        other_max = other.get_min()
        if self_max is None or other_max is None:
            new_max = None
        else:
            new_max = max([self_max, other_max])

        no_skew_info = self.get_skew() is None and other.get_skew() is None
        no_kurtosis_info = self.get_kurtosis() is None and other.get_kurtosis() is None
        no_sampling = no_skew_info and no_kurtosis_info
        partial_info = no_skew_info != no_kurtosis_info

        if no_sampling:
            new_skew = None
            new_kurtosis = None
        elif partial_info:
            raise RuntimeError('Cant combine where skew or kurtosis specified but not both.')
        else:
            self_approx_normal = self.get_is_approx_normal()
            other_approx_normal = other.get_is_approx_normal()
            both_normal = self_approx_normal and other_approx_normal

            if not both_normal:
                if allow_multiple_shapes:
                    print('Encountered multiple distribution shapes.')
                else:
                    raise RuntimeError('Encountered multiple distribution shapes.')

            new_skew = new_mean = get_weighted_average(
                self.get_skew(),
                other.get_skew()
            )

            new_kurtosis = new_mean = get_weighted_average(
                self.get_kurtosis(),
                other.get_kurtosis()
            )

        return Distribution(
            new_mean,
            new_std,
            new_count,
            new_min,
            new_max,
            new_skew,
            new_kurtosis
        )


class WelfordAccumulator:
    """Implementor of a memory-efficient and numerically stable Welford Accumulator.

    Structure to calculate mean and standard deivation over a large distribution with memory
    efficiency.
    """

    def __init__(self):
        """Create an accumulator with an empty sample."""
        self._count = 0
        self._mean = 0
        self._delta_accumulator = 0

    def add(self, value):
        """Add a new value to this sample.

        Args:
            value: The value to add to the accumulator.
        """
        pre_delta = value - self._mean

        self._count += 1
        self._mean += pre_delta / self._count

        post_delta = value - self._mean

        self._delta_accumulator += post_delta * post_delta

    def get_mean(self):
        """Get the average of the sample provided to this accumulator.

        Returns:
            Mean of all values provided via the add method.
        """
        return self._mean

    def get_std(self):
        """Get the standard deviation of the sample provided tot his accumulator.

        Returns:
            Standard deviation of all values provided via the add method.
        """
        return math.sqrt(self._delta_accumulator / (self._count - 1))
