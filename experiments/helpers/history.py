"""
    Keeping Track of the experiment history
"""

import pandas as pd
import os
import time
import copy


class ExpHistory:
    def __init__(self):
        self.experiments = pd.read_csv('experiment_history.csv')
        self.user = os.environ['USER']

    def report_experiment(self, params):
        self.experiments = self.experiments.append(params, ignore_index=True)
        self.experiments.to_csv('experiment_history.csv', index=False)

    def copy_from_record(self, index):
        now = time.time()
        localtime = time.asctime(time.localtime(time.time()))
        new_copy = copy.deepcopy(self.experiments.loc[index])
        new_copy.user = self.user
        new_copy.localtime = localtime
        new_copy.timestamp = now
        return new_copy


history = ExpHistory()
