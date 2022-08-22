from collections import OrderedDict
from collections import namedtuple
from itertools import product


class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


params = OrderedDict(
    lr=[.01, .001],
    batch_size=[100, 1000]
)

runs = RunBuilder.get_runs(params)
print(runs)
