from TNLCFRS.parser.helper.metric import discoF1
import pickle
from library.ensemble import ensemble
from teachers_guide import teachers
import os
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


fold = 'test'
WORDS = f'{fold}.words.pkl'
PREDICTION = f'{fold}.prediction.pkl'
GOLD = f'{fold}.gold.pkl'
source='lassy_0'
TEACHERS_COUNT = 5
paths = list(teachers[source].keys())[:TEACHERS_COUNT]
weights = list(teachers[source].values())[:TEACHERS_COUNT]
TOP_t_TYPES = 10


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
gold = pickle.load(open(os.path.join(paths[0],GOLD), 'rb'))
avgs = ensemble(trees, beta=1, MAX_CANDID=40, weights=weights, parallel=True)

def recall(tree_corpus, gold, label, select_constituents):
    def flatten_list(nested_list):
        flat_list = []
        for element in nested_list:
            if isinstance(element, list):
                flat_list.extend(flatten_list(element))
            else:
                flat_list.append(element)
        return flat_list

    total, TP = 0, 0
    for tree_sent, gold_sent in zip(tree_corpus, gold):
        tree_sent, gold_sent = select_constituents(tree_sent), select_constituents(gold_sent)
        for gold_constituent in gold_sent:
            if label != gold_constituent[-1]:
                continue
            gold_constituent = sorted(flatten_list(gold_constituent[:-1]))
            if tuple(gold_constituent) in tree_sent:
                TP += 1
            total += 1
    return TP/total*100

def do_everything(select_constituents, y_label='Recall', ymax=100, bbox_to_anchor=(.9, 1), save_path='per_label_analysis.pdf'):
    labels = [ggg[-1] for g in gold for ggg in select_constituents(g)]
    labels_freq = dict(sorted(zip(*np.unique(labels, return_counts=True)), key=lambda x: x[1], reverse=True))

    selected_labels = list(labels_freq.keys())[:TOP_t_TYPES]
    teacher_mean_recalls, teacher_std_recalls, avg_recalls = [], [], []
    for label in selected_labels:
        teachers_recall = [recall(t, gold, label, select_constituents) for t in trees]
        teacher_mean_recalls.append(np.mean(teachers_recall))
        teacher_std_recalls.append(np.std(teachers_recall))
        avg_recall = recall(avgs, gold, label, select_constituents)
        avg_recalls.append(avg_recall)

    def my_round(num):
        if num < 10:
            return round(num, 1)
        else:
            return round(num)

    selected_labels = [l+f'\n{my_round(100*f/sum(labels_freq.values()))}%' for l,f in labels_freq.items()][:TOP_t_TYPES]
    fig, ax = plt.subplots(figsize=(15,7))
    bar_width = 0.35
    bar_positions_left = np.arange(TOP_t_TYPES)
    bar_positions_right = [x + bar_width for x in bar_positions_left]
    ax.bar(bar_positions_left, teacher_mean_recalls, width=bar_width, color='#007e73', yerr=teacher_std_recalls, capsize=7, error_kw={'elinewidth': 2.5}, label='Individuals', alpha=.3)
    ax.bar(bar_positions_right, avg_recalls, width=bar_width, color='#007e73', label='Ensembe', hatch='/')
    ax.set_ylim(0, ymax)
    ax.set_xlim(-.35, 9.7)
    ax.set_ylabel(y_label, fontsize=25)
    ax.set_xticks([r + bar_width / 2 for r in range(len(teacher_mean_recalls))])
    ax.set_xticklabels(selected_labels)
    ax.legend(fontsize=25, bbox_to_anchor=bbox_to_anchor)
    ax.tick_params(labelsize=23)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(save_path, bbox_inches='tight')

do_everything(lambda x: x[0]+x[1])
# do_everything(lambda x: x[1], y_label='Discontinuous recall', ymax=30, bbox_to_anchor=None, save_path='per_label_disco_analysis.pdf')