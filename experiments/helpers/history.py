"""
    Keeping Track of the experiment history
"""

import pandas as pd
import os
import time
import copy

HIST_FILE_NAME = 'experiment_history.csv'


class ExpHistory:

    def __init__(self, file_name):
        self.file_name = file_name
        self.experiments = pd.read_csv(file_name)
        self.user = os.environ['USER']

    def report_experiment(self, params):
        self.experiments = self.experiments.append(params, ignore_index=True)
        self.experiments.to_csv(self.file_name, index=False)

    def copy_from_record(self, index):
        now = time.time()
        localtime = time.asctime(time.localtime(time.time()))
        new_copy = copy.deepcopy(self.experiments.loc[index])
        new_copy.user = self.user
        new_copy.localtime = localtime
        new_copy.timestamp = now
        return new_copy


history = ExpHistory(HIST_FILE_NAME)
