import math


class Distribution:

    def __init__(self, mean, std, count, dist_min=None, dist_max=None):
        self._mean = mean
        self._std = std
        self._count = count
        self._dist_min = dist_min
        self._dist_max = dist_max

    def get_mean(self):
        return self._mean
    
    def get_std(self):
        return self._std

    def get_count(self):
        return self._count
    
    def get_min(self):
        return self._dist_min
    
    def get_max(self):
        return self._dist_max

    def combine(self, other):
        new_count = self.get_count() + other.get_count()
        
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

        return Distribution(new_mean, new_std, new_count, new_min, new_max)


class WelfordAccumulator:

    def __init__(self):
        self._count = 0
        self._mean = 0
        self._delta_accumulator = 0

    def add(self, value):
        pre_delta = value - self._mean
        
        self._count += 1
        self._mean += pre_delta / self._count
        
        post_delta = value - self._mean

        self._delta_accumulator += local_delta * post_delta

    def get_mean(self):
        return self._mean

    def get_std(self):
        return math.sqrt(self._delta_accumulator / (self._count - 1))
