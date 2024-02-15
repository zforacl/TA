from TNLCFRS.parser.helper.metric import discoF1
import pickle
from library.ensemble import ensemble
from teachers_guide import teachers
import os
import numpy as np
from tqdm import tqdm
import json

fold = 'test'
WORDS = f'{fold}.words.pkl'
PREDICTION = f'{fold}.prediction.pkl'
GOLD = f'{fold}.gold.pkl'
source='lassy_0_inc_10_20_testsorted'
paths = list(teachers[source].keys())
weights = list(teachers[source].values())
len_threshold = float('inf')
N = 30
LOAD = False


words = [pickle.load(open(os.path.join(path,WORDS), 'rb')) for path in paths]
words = [list(map(' '.join, word)) for word in words]
reorders = [[word.index(gword) for gword in words[0]] for word in words]
trees = [pickle.load(open(os.path.join(path,PREDICTION), 'rb')) for path in paths]
trees = [[tree[i] for i in reorder] for tree, reorder in zip(trees, reorders)]
for tree in trees:
    for t in tree:
        n = max([c[-1] for c in t[0]+t[1]])
        if (0, n) not in t[0]:
            t[0].append((0, n))
trees = [[t for t, w in zip(tree, words[0]) if len(w.split())<len_threshold] for tree in trees]
gold = pickle.load(open(os.path.join(paths[0],GOLD), 'rb'))
gold = [t for t, w in zip(gold, words[0]) if len(w.split())<len_threshold]

if LOAD:
    f1_means = json.load(open('analysis_incremental_data/f1_means.json', 'r'))
    f1_stds = json.load(open('analysis_incremental_data/f1_stds.json', 'r'))
    f1c_means = json.load(open('analysis_incremental_data/f1c_means.json', 'r'))
    f1c_stds = json.load(open('analysis_incremental_data/f1c_stds.json', 'r'))
    f1d_means = json.load(open('analysis_incremental_data/f1d_means.json', 'r'))
    f1d_stds = json.load(open('analysis_incremental_data/f1d_stds.json', 'r'))
else:
    f1_means, f1_stds = [], []
    f1c_means, f1c_stds = [], []
    f1d_means, f1d_stds = [], []
    for k in tqdm(range(1, len(trees)+1)):
        f1s, f1cs, f1ds = [], [], []
        for i in (range(N)):
            ts_indexes = np.random.choice(len(trees), k, replace=False)
            ts = [trees[i] for i in ts_indexes]
            ws = [weights[i] for i in ts_indexes]
            avgs = ensemble(ts, beta=1, weights=ws, MAX_CANDID=40, parallel=True, progress_bar=False)
            avgs = [t if a is None else a for a,t in zip(avgs, ts[0])]
            f1_metric = discoF1()
            f1_metric(list(filter(lambda x: x is not None, avgs)), gold)
            f1_d, prec_d, recall_d = f1_metric.corpus_uf1_disco
            f1_c, prec_c, recall_c = f1_metric.corpus_uf1
            f1 = f1_metric.all_uf1 if type(f1_metric.all_uf1) is float else f1_metric.all_uf1[0]
            f1s.append(f1)
            f1cs.append(f1_c)
            f1ds.append(f1_d)

        f1_means.append(np.mean(f1s))
        f1_stds.append(np.std(f1s))
        f1c_means.append(np.mean(f1cs))
        f1c_stds.append(np.std(f1cs))
        f1d_means.append(np.mean(f1ds))
        f1d_stds.append(np.std(f1ds))

    json.dump(f1_means, open('analysis_incremental_data/f1_means.json', 'w'))
    json.dump(f1_stds, open('analysis_incremental_data/f1_stds.json', 'w'))
    json.dump(f1c_means, open('analysis_incremental_data/f1c_means.json', 'w'))
    json.dump(f1c_stds, open('analysis_incremental_data/f1c_stds.json', 'w'))
    json.dump(f1d_means, open('analysis_incremental_data/f1d_means.json', 'w'))
    json.dump(f1d_stds, open('analysis_incremental_data/f1d_stds.json', 'w'))

f1_means, f1_stds = np.array(f1_means), np.array(f1_stds)
f1c_means, f1c_stds = np.array(f1c_means), np.array(f1c_stds)
f1d_means, f1d_stds = np.array(f1d_means), np.array(f1d_stds)

f1_means *= 100
f1_stds *= 100
f1c_means *= 100
f1c_stds *= 100
f1d_means *= 100
f1d_stds *= 100

import matplotlib.pyplot as plt

def plot(ax, means, stds, color, label, marker):
    labels = [str(i) for i in range(1, len(means)+1)]
    ax.plot(labels, means, linestyle=':', color=color)
    ax.errorbar(labels, means, fmt='none', yerr=stds, capsize=5, elinewidth=1.5, capthick=1.5, color=color)
    style = ax.scatter(labels, means, marker=marker, s=100, label=label, color=color)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    return ax, style

top_range = 36, 56
but_range = 1,11
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7, 7), gridspec_kw={'height_ratios': [top_range[1]-top_range[0], but_range[1]-but_range[0]]})
ax, line1 = plot(axs[0], f1_means, f1_stds, color='#007e73', label='F1', marker='o')
ax, line2 = plot(ax, f1c_means, f1c_stds, color='#93af39', label='CF1', marker='^')
ax.set_ylim(*top_range)
axb, line3 = plot(axs[1], f1d_means, f1d_stds, color='#003f5c', label='DF1', marker='s')
axb.set_ylim(*but_range)
axb.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(labeltop=False)
ax.tick_params(labelsize=25)
ax.xaxis.set_ticks_position('none') 
axb.xaxis.tick_bottom()
axb.tick_params(labelsize=25)
d = .015
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((1-d, 1+d), (-d, +d), **kwargs)
ax.plot((-d, +d), (-d, +d), **kwargs)
kwargs.update(transform=axb.transAxes)
axb.plot((1-d, 1+d), (1-1.5*d, 1+d*2.5), **kwargs)
axb.plot((-d, +d), (1-1.5*d, 1+d*2.5), **kwargs)
plt.tight_layout()
grid_color = '#d3d3d3'
ax.grid(axis='x', color=grid_color)
axb.grid(axis='x', color=grid_color)
ax.legend([line1, line2, line3], [r'$F_1$', r'$CF_1$', r'$DF_1$'], loc='lower right', fontsize=25, bbox_to_anchor=(1, 0))
plt.xlabel('# teachers in the ensemble', fontsize=25)
plt.savefig('analysis_incremental.png', bbox_inches='tight')