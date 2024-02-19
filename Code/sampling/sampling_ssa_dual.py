import os
import math
import time
import glob
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import tqdm


class SpatialSimulatedAnnealing:
    def __init__(self, df, n_sample, cost_func,
                 max_iter=1000, threshold=None, max_reject=None):
        self._df = df
        self._origin_pnts = self._df[['X', 'Y']].values.tolist()
        self._muls = self._df[['MUL']].values.tolist()
        self.n_sample = n_sample
        self.cost_func = cost_func
        self.max_iter = max_iter
        self.threshold = threshold
        self.max_reject = max_reject

        # Initialize
        self._best_sample_idx, self.best_sample = self.initial_sample()
        # different idx set
        self._dif_idx = list(set(range(len(self._origin_pnts))) - set(self._best_sample_idx))
        # cost function value list
        cost_mul, cost_dis = self.cost_func(self._df, self._best_sample_idx)
        self.scores = [(cost_mul, cost_dis)]
        # self.scores_dis = []
        # optimized cost function value list
        self.best_scores = [(0, cost_mul, cost_dis)]

    def initial_sample(self):
        sample_idx = random.sample(range(len(self._origin_pnts)), self.n_sample)
        sample = np.array(self._origin_pnts)[sample_idx].tolist()
        return sample_idx, sample

    def replace(self, sample_idx):
        """
        Replace the worst sample in sampling set with random sample in difference set.
        :param sample_idx:
        :return:
        """
        # optimization of the minimum nni value
        idx_dis = np.argmin(nearest_distance_vector(np.array(self._origin_pnts)[sample_idx].tolist()))
        rand_d = random.randint(0, len(self._dif_idx) - 1)
        sample_idx[idx_dis] = self._dif_idx[rand_d]
        # optimization of the minimum mul value
        idx_mul = np.argmin(np.array(self._muls)[sample_idx])
        rand_d = random.randint(0, len(self._dif_idx) - 1)
        sample_idx[idx_mul] = self._dif_idx[rand_d]
        return sample_idx

    def simulated_annealing_sampling(self):
        reject_cnt = 0
        for iter_idx in tqdm.tqdm(range(self.max_iter),
                                  desc='Number of samples, %d' % self.n_sample):
            # Termination conditions
            if self.threshold and self.cost_func(self._df, self._best_sample_idx) <= self.threshold:
                break

            if self.max_reject and reject_cnt >= self.max_reject:
                break

            temp_sample_idx = self.replace(self._best_sample_idx.copy())

            temp_sample = np.array(self._origin_pnts)[temp_sample_idx].tolist()
            score_mul, score_dis = self.cost_func(self._df, self._best_sample_idx)
            self.scores.append((score_mul, score_dis))

            if score_mul < self.best_scores[-1][1]:
                p_mul = 1 # convertion probability for mul
            else:
                p_mul = int(1 / (1 + math.exp(score_mul - self.best_scores[-1][1])) > random.uniform(0, 1))

            if score_dis < self.best_scores[-1][2]:
                p_dis = 1 # convertion probability for distance
            else:
                p_dis = int(1 / (1 + math.exp(score_dis - self.best_scores[-1][2])) > random.uniform(0, 1))

            if p_mul and p_dis:
                reject_cnt = 0
                self._best_sample_idx = temp_sample_idx
                self.best_sample = np.array(self._origin_pnts)[self._best_sample_idx].tolist()
                self._dif_idx = list(set(range(len(self._origin_pnts))) - set(self._best_sample_idx)) # TODO: 这里应该放弃之前删掉过的样本点
                self.best_scores.append((iter_idx, score_mul, score_dis))
            else:
                reject_cnt += 1

    def write_file(self, path):
        with open(os.path.join(path, 'best_samples.txt'), 'w') as f:
            for sample in self.best_sample:
                f.write(str(sample[0]))
                f.write(',')
                f.write(str(sample[1]))
                f.write('\n')

        sorces = {'total_mul_scores': np.array(self.scores)[:, 0],
                  'total_dis_scores': np.array(self.scores)[:, 1],
                  'best_mul_iter_num': np.array(self.best_scores)[:, 0],
                  'best_mul_scores': np.array(self.best_scores)[:, 1],
                  'best_dis_scores': np.array(self.best_scores)[:, 2],
                  'best_sample_0': np.array(self.best_sample)[:, 0],
                  'best_sample_1': np.array(self.best_sample)[:, 1]}

        np.save(os.path.join(path, 'sampling_results.npy'), sorces)


    def visualize(self):
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes = axes.ravel()

        axes[0].plot(range(len(self.scores)), np.array(self.scores)[:, 0])
        axes[0].scatter(np.array(self.best_scores)[:, 0], np.array(self.best_scores)[:, 1], c='r')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Cost')
        axes[0].set_title('Cost MUL curve')

        axes[1].plot(range(len(self.scores)), np.array(self.scores)[:, 1])
        axes[1].scatter(np.array(self.best_scores)[:, 0], np.array(self.best_scores)[:, 2], c='r')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Cost')
        axes[1].set_title('Cost Distance curve')

        axes[2].scatter(np.array(self.best_sample)[:, 0], np.array(self.best_sample)[:, 1])
        axes[2].set_xlabel('Longitude')
        axes[2].set_ylabel('Latitude')
        axes[2].set_title('Best sampling local view')

        axes[3].scatter(np.array(self._origin_pnts)[:, 0], np.array(self._origin_pnts)[:, 1])
        axes[3].scatter(np.array(self.best_sample)[:, 0], np.array(self.best_sample)[:, 1], c='r')
        axes[3].set_xlabel('Longitude')
        axes[3].set_ylabel('Latitude')
        axes[3].set_title('Best sampling global view')
        plt.show()


def nearest_distance_vector(XY_list):
    # distance matrix
    dis_mat = distance.cdist(XY_list, XY_list, 'euclidean')
    # delete diag from matrix
    del_diag_dis_mat = dis_mat[~np.eye(dis_mat.shape[0], dtype=bool)]
    min_dist = del_diag_dis_mat.reshape(dis_mat.shape[0], -1).min(axis=1)
    return min_dist


def cost_func(df, selected_FID_list):
    dis_mat = nearest_distance_vector(df.loc[selected_FID_list][['X', 'Y']])
    n_sample = len(selected_FID_list)

    min_dist_sum = dis_mat.sum() 
    nni = (min_dist_sum / n_sample) / (1/(2*np.sqrt(n_sample)))

    mul = df.loc[selected_FID_list]['MUL'].sum() / n_sample

    cost_mul = 1 / mul
    cost_dis = 1 / nni
        
    return cost_mul, cost_dis


def main(path, output_folder):
    n_sample = 80
    max_iter = 5000

    df = pd.read_csv(path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    SSA = SpatialSimulatedAnnealing(df, n_sample, cost_func, max_iter)
    SSA.simulated_annealing_sampling()

    SSA.write_file(output_folder)
    SSA.visualize()


if __name__ == '__main__':
    grid_file = ''
    output_folder = ''
    main(grid_file, output_folder)