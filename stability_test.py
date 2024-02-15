from TNLCFRS.parser.helper.metric import LikelihoodMetric,  UF1, LossMetric, UAS, discoF1, printTree
import pickle
import os
from scipy.stats import binom

METRIC = 'discontinuous'
fold = 'test'
WORDS = f'{fold}.words.pkl'
PREDICTION = f'{fold}.prediction.pkl'
GOLD = f'{fold}.gold.pkl'
dataset='lassy_0'
teacher='TNLCFRS/log/dutch/tnlcfrs_nl/TN_LCFRS2024-01-27-00_55_50/'
student=dataset+'.ensemble.'
paths=[student,teacher]

words = [pickle.load(open((path+WORDS), 'rb')) for path in paths]
words = [list(map(' '.join, word)) for word in words]
reorders = [[word.index(gword) for gword in words[0]] for word in words]
trees = [pickle.load(open((path+PREDICTION), 'rb')) for path in paths]
trees = [[tree[i] for i in reorder] for tree, reorder in zip(trees, reorders)]
gold = pickle.load(open((paths[0]+GOLD), 'rb'))
gold = [t for a,t in zip(trees[0], gold) if a is not None]
trees = [[t for a,t in zip(trees[0], tree) if a is not None] for tree in trees]
for tree in trees:
    for t in tree:
        n = max([c[-1] for c in t[0]+t[1]])
        if (0, n) not in t[0]:
            t[0].append((0, n))

def metric(p, g):
    f1computer = discoF1()
    f1computer(p, g)
    f1_d, prec_d, recall_d = f1computer.corpus_uf1_disco
    f1_c, prec_c, recall_c = f1computer.corpus_uf1
    f1 = f1computer.all_uf1 if type(f1computer.all_uf1) is float else f1computer.all_uf1[0]
    metrics = {
        'overal': f1,
        'continuous': f1_c,
        'discontinuous': f1_d
    }
    return metrics[METRIC]

win, tie, loss = 0, 0, 0
for s, t, g in zip(*trees, gold):
    t, s, g = [t], [s], [g]
    t_metric = metric(t, g)
    s_metric = metric(s, g)
    tie += s_metric == t_metric
    win += s_metric > t_metric
    loss += s_metric < t_metric
n = win+tie+loss

print(win, tie, loss, '/', len(gold))

def NCS_test(p, o, n, alpha=.05):
    return p > binom.ppf(1-alpha, p+n, 0.5)

print(NCS_test(win, tie, loss))
print(NCS_test(win, tie, loss, alpha=.01))