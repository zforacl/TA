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
    f1s = json.load(open('best_to_worst_incremental_data/f1s.json', 'r'))
    f1cs = json.load(open('best_to_worst_incremental_data/f1cs.json', 'r'))
    f1ds = json.load(open('best_to_worst_incremental_data/f1ds.json', 'r'))
else:
    f1s = []
    f1cs = []
    f1ds = []
    for k in tqdm(range(1, len(trees)+1)):
        ts_indexes = np.random.choice(len(trees), k, replace=False)
        ts = trees[:k]
        ws = weights[:k]
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

    json.dump(f1s, open('best_to_worst_incremental_data/f1s.json', 'w'))
    json.dump(f1cs, open('best_to_worst_incremental_data/f1cs.json', 'w'))
    json.dump(f1ds, open('best_to_worst_incremental_data/f1ds.json', 'w'))


def evaluate_teacher(tree):
    f1_metric = discoF1()
    f1_metric(tree, gold)
    f1_d, prec_d, recall_d = f1_metric.corpus_uf1_disco
    f1_c, prec_c, recall_c = f1_metric.corpus_uf1
    f1 = f1_metric.all_uf1 if type(f1_metric.all_uf1) is float else f1_metric.all_uf1[0]
    return f1*100, f1_c*100, f1_d*100

tf1s, tf1cs, tf1ds = zip(*[evaluate_teacher(tree) for tree in trees])

f1s = np.array(f1s)
f1cs = np.array(f1cs)
f1ds = np.array(f1ds)

f1s *= 100
f1cs *= 100
f1ds *= 100

import matplotlib.pyplot as plt

def plot(ax, means, base, color, label, marker):
    labels = [str(i) for i in range(1, len(means)+1)]
    ax.plot(labels, means, linestyle=':', color=color)
    style = ax.scatter(labels, means, marker=marker, s=100, label=label, color=color)
    base_style = ax.scatter(labels, base, marker=marker, s=100, label=label, color=color, facecolors='none')
    # ax.spines['left'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    return ax, style, base_style

top_range = min(min(tf1s), min(tf1cs))-1, max(max(f1s), max(f1cs))+1
but_range = min(tf1ds)-1, max(f1ds)+1
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7, 7), gridspec_kw={'height_ratios': [top_range[1]-top_range[0], but_range[1]-but_range[0]]})
ax, _, line1 = plot(axs[0], f1s, tf1s, color='#007e73', label='F1', marker='o')
ax, _, line2 = plot(ax, f1cs, tf1cs, color='#93af39', label='CF1', marker='^')
ax.set_ylim(*top_range)
axb, _, line3 = plot(axs[1], f1ds, tf1ds, color='#003f5c', label='DF1', marker='s')
axb.set_ylim(*but_range)
ax.spines['bottom'].set_visible(False)
axb.spines['top'].set_visible(False)
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
ax.yaxis.tick_right()
axb.yaxis.tick_right()
ax.legend([line1, line2, line3], [',', ',', "  Teacher's performance"], loc='lower left', fontsize=22, bbox_to_anchor=(0, 0), ncol=3, handletextpad=-.6, columnspacing=-.6)
plt.xlabel('# teachers in the ensemble', fontsize=25)
plt.savefig('best_to_worst_incremental.png', bbox_inches='tight')