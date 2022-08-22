from collections import namedtuple
from itertools import product


# 超参参数化
class RunBuilder():
    @staticmethod
    # 输入是字典 输出是字典集的笛卡尔积
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
