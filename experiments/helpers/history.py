"""
    A simple tool to help you keep track of the experiment history
"""

from helpers.os_utils import os_info
import pandas as pd
import os
import time
import copy


class ExpHistory:

    def __init__(self, file_name):
        self.file_name = file_name
        self.experiments = pd.read_csv(file_name)
        self.user = os.environ.get('USER', os.environ.get('USERNAME', 'anonymous'))
        
    def report_experiment(self, record):
        """
        Append the given hyper-parameter record to the history, and write the entire history to disk.
        :param record: the new hyper-parameter set
        """
        self.experiments = self.experiments.append(record, ignore_index=True)
        self.experiments.to_csv(self.file_name, index=False)

    def copy_from_record(self, index):
        """
        creates a copy of the record identified by parameter 'index', with current user and time values
        :param index: the row number of the row to be copied
        :return: a copy of that row
        """
        now = int(time.time())
        localtime = time.asctime(time.localtime(now))
        new_copy = copy.deepcopy(self.experiments.loc[index])
        new_copy.user = self.user
        new_copy.localtime = localtime
        new_copy.timestamp = now
        return new_copy

    def suggest_from_history(self):
        """
        creates a copy of the most recent experiment hyper-parameter record on the given platform, or if none such
        is found, the most recent record from any platform.
        :return: a deep copy of the matching history record.
        """
        rt = os_info()
        this_node = rt['node']
        h = self.experiments
        index = h[h['node'] == this_node].tail(1).index
        if len(index) == 1:
            sugg_index = index[0]
        else:
            sugg_index = self.experiments.shape[0] - 1

        acopy = self.copy_from_record(sugg_index)
        acopy.os = rt['os']
        acopy.node = rt['node']
        acopy.machine = rt['machine']
        acopy.cuda = rt['cuda']
        return acopy
