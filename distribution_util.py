import numpy
import scipy.optimize
import scipy.stats


def find_beta_distribution(mean, std, skew, kurtosis):

    def get_errors(params):
        samples = scipy.stas.beta.rvs(a, b, loc=loc, scale=1000)
        
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