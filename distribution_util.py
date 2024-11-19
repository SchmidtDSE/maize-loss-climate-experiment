import math

import numpy
import scipy.optimize
import scipy.stats


def find_beta_distribution(mean, std, skew, kurtosis):

    if not math.isfinite(mean):
        raise RuntimeError('Passed invalid mean %s.' % str(mean))

    if not math.isfinite(std):
        raise RuntimeError('Passed invalid std %s.' % str(std))

    if not math.isfinite(skew):
        raise RuntimeError('Passed invalid skew %s.' % str(skew))

    if not math.isfinite(kurtosis):
        raise RuntimeError('Passed invalid kurtosis %s.' % str(kurtosis))

    def get_errors(params):
        a = params[0]
        b = params[1]
        loc = params[2]
        scale = params[3]

        params_str = ','.join(map(lambda x: str(x), params))

        if numpy.any(numpy.isnan(params)):
            return numpy.inf

        if scale < 0:
            raise RuntimeError(
                'Invalid scale encountered within %s.' % params_str
            )

        try:
            samples = scipy.stats.beta.rvs(a, b, loc=loc, scale=scale)
        except Exception as e:
            template_vals = (params_str, str(e))
            raise RuntimeError(
                'Failed to build sample from %s due to %s.' % template_vals
            )

        candidate_mean = numpy.mean(samples)
        candidate_std = numpy.std(samples)
        candidate_skew = scipy.stats.skew(samples)
        candidate_kurtosis = scipy.stats.kurtosis(samples)

        errors = [
            candidate_mean - mean,
            candidate_std - std,
            candidate_skew - skew,
            candidate_kurtosis - kurtosis
        ]
        errors_squared = map(lambda x: x ** 2, errors)
        return sum(errors_squared)

    params = [2, 2, mean - std, 2 * std]

    bounds = [(0.1, 100), (0.1, 100), (None, None), (0.0001, None)]
    optimized = scipy.optimize.minimize(
        get_errors,
        params,
        bounds=bounds,
        method='L-BFGS-B'
    )

    return dict(zip(['a', 'b', 'loc', 'scale'], optimized.x))
