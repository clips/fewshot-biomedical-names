import numpy as np
from tqdm import tqdm


###############################################################
###############################################################
##################      RANKING UTILS     #####################
###############################################################
###############################################################


class RankingUtils:

    @staticmethod
    def ranking_accuracy(ranking, cutoff=1):
        return sum(1 for x in ranking if min(x) < cutoff) / len(ranking)

    def ranking_distr(self, ranking, cutoff):
        return {i: self.ranking_accuracy(ranking, i) for i in range(1, cutoff + 1)}

    @staticmethod
    def mrr(ranking): return np.mean([1 / (min(ranks) + 1) for ranks in ranking])

    @staticmethod
    def precision_at_k(r, k):
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(r)

    def average_precision(self, r):
        r = np.asarray(r) != 0
        out = [self.precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
        if not out:
            return 0.
        return np.mean(out)

    @staticmethod
    def convert_ranks(ranks):
        r = np.zeros(max(ranks) + 1)
        for rank in ranks:
            r[rank] = 1

        return r

    def mean_average_precision(self, ranking, verbose=False):
        avg_precs = []
        for ranks in tqdm(ranking, disable=not verbose):
            # convert ranks to binary labels
            r = self.convert_ranks(ranks)
            avg_prec = self.average_precision(r)
            avg_precs.append(avg_prec)

        mAP = np.mean(avg_precs)

        return mAP
