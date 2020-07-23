import csv
import random
import copy
import numpy as np
import math

class MemoryBuffer:
    def __init__(self, param_set):
        self.buffer = []
        self.current = {}
        self.mamory_size = param_set['mamory_size']
        self.path = 'data/' + param_set['path']

        self.keys = param_set['memory_keys']
        self.fine_keys = self.keys + ['next_obs', 'return']

        self.device = param_set['device']
        self.lamda_return = param_set['lamda_return']
        if self.lamda_return:
            self.gamma = param_set['gamma']
            self.lamda = param_set['lamda']
            self.fine_keys += ['advangtage']

        self.max_seq_len = param_set['max_seq_len']
        self.fill = param_set['memory_fill']
        if self.fill:
            self.fine_keys += ['mask']

    def end_trajectory(self):
        for agent_id in self.current.keys():

            for key in self.keys:
                self.current[agent_id][key] = np.array(self.current[agent_id][key])

            len_trajectory = self.current[agent_id]['next_obs'].shape[0]

            self.current[agent_id]['next_obs'] = copy.deepcopy(self.current[agent_id]['observation'][1:])
            self.current[agent_id]['observation'] = self.current[agent_id]['observation'][:-1]

            if 'avail_action' in self.keys:
                self.current[agent_id]['next_avail_action'] = copy.deepcopy(self.current[agent_id]['avail_action'][1:])
                self.current[agent_id]['avail_action']= self.current[agent_id]['avail_action'][:-1]

            if self.lamda_return:
                advangtage = np.zeros_like(self.current[agent_id]['reward'])
                returns = np.zeros_like(self.current[agent_id]['reward'])
                deltas = np.zeros_like(self.current[agent_id]['reward'])
                returns[-1] = self.current[agent_id]['reward'][-1]
                deltas[-1] = self.current[agent_id]['reward'][-1] - self.current[agent_id]['value'][-1]
                advangtage[-1] = deltas[-1]
                for i in range(len_trajectory - 2, -1, -1):
                    returns[i] = self.current[agent_id]['reward'][i] + self.gamma * returns[i+1]
                    deltas[i] = self.current[agent_id]['reward'][i] + self.gamma * self.current[agent_id]['value'][i+1] - self.current[agent_id]['value'][i]
                    advangtage[i] = deltas[i] + self.gamma * self.lamda * advangtage[i+1]
                self.current[agent_id]['return'] = returns
                self.current[agent_id]['advangtage'] = advangtage
            else:
                returns = np.zeros_like(self.current[agent_id]['reward'])
                returns[-1] = self.current[agent_id]['reward'][-1]
                for i in range(len_trajectory - 2, -1, -1):
                    returns[i] = self.current[agent_id]['reward'][i] + self.gamma * returns[i+1]
                self.current[agent_id]['return'] = returns

            if self.fill:
                fill_len = self.max_seq_len - len_trajectory
                for key in self.current[agent_id].keys():
                    fill_part = np.zeros((fill_len,) + self.current[agent_id][key].shape[1:])
                    self.current[agent_id][key] = np.concatenate([self.current[agent_id][key],fill_part])
                mask = np.ones_like(self.current[agent_id]['reward'])
                mask[len_trajectory:] = 0
                self.current[agent_id]['mask'] = mask

        self.buffer.append(self.current)
        if len(self.buffer) > self.mamory_size:
            self.buffer = self.buffer[1:]
        self.current = {}

    def append(self, experience:{}):
        agent_id = experience['id']
        if not experience['id'] in self.current.keys():
            self.current[id] = {key:[] for key in self.keys}
        for key in experience.keys():
            self.current[agent_id][key].append(experience[id])


    def sample(self, idList:[], batchSize:int,):
        """
        :param idList:
        :param batchSize: n random trajectory
        :return: batch
        """
        batch = {key:[[] for _ in idList] for key in self.fine_keys}

        for item in range(batchSize):
            b_id = random.randint(0, len(self.buffer)-1)
            for index, d_id in enumerate(idList):
                for key in self.fine_keys:
                    batch[key][index] += [self.buffer[b_id][d_id][key],]

        if self.fill:
            for key in self.fine_keys:
                batch[key] = np.array(batch[key])
            batch.update({'shape': (len(idList), batchSize, self.max_seq_len)})
        else:
            for key in self.fine_keys:
                batch[key] = np.concatenate(batch[key])
            batch.update({'shape': (len(idList), batch['reward'].shape[1])})
        return batch

    def get_current(self, idList:[]):
        batch = {key:[None for _ in idList] for key in self.keys}
        for index, d_id in enumerate(idList):
            for key in self.keys:
                batch[key][index] = np.array(self.current[d_id][key])
        return batch

    def get_all_trajectories(self, idList:[]):
        batch = {key:[[] for _ in idList] for key in self.fine_keys}
        len_buffer = len(self.buffer)

        for b_id in range(len_buffer):
            for index, d_id in enumerate(idList):
                for key in self.fine_keys:
                    batch[key][index] += [self.buffer[b_id][d_id][key],]

        if self.fill:
            for key in self.fine_keys:
                batch[key] = np.array(batch[key])
            batch.update({'shape': (len(idList), len_buffer, self.max_seq_len)})
        else:
            for key in self.fine_keys:
                batch[key] = np.concatenate(batch[key])
            batch.update({'shape': (len(idList), batch['reward'].shape[1])})
        return batch

