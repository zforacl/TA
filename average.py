from TNLCFRS.parser.helper.metric import discoF1
import pickle
from library.ensemble import ensemble
from teachers_guide import teachers
import shutil
import os

name = 'ensemble'
fold = 'test'
WORDS = f'{fold}.words.pkl'
PREDICTION = f'{fold}.prediction.pkl'
GOLD = f'{fold}.gold.pkl'
source='lassy_0'
TEACHERS_COUNT = 5
paths = list(teachers[source].keys())[:TEACHERS_COUNT]
weights = list(teachers[source].values())[:TEACHERS_COUNT]


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
        for i in range(n):
            if (i, i+1) not in t[0]:
                t[0].append((i, i+1))

avgs = ensemble(trees, weights=weights, MAX_CANDID=40)
avgs = [t if a is None else a for a,t in zip(avgs, trees[0])]
pickle.dump(avgs, open(source+'.'+name+'.'+fold+'.prediction.pkl', 'wb'))
shutil.copyfile(os.path.join(paths[0],WORDS), source+'.'+name+'.'+fold+'.words.pkl')
shutil.copyfile(os.path.join(paths[0],GOLD), source+'.'+name+'.'+fold+'.gold.pkl')

gold = pickle.load(open(os.path.join(paths[0],GOLD), 'rb'))
co, disco = 0, 0
for g in gold:
    co += len(g[0])
    disco += len(g[1])
total = co+disco
print(co, disco, round(co/total,2), round(disco/total,2))
print(len(gold), end=' -> ')
gold = [t for a,t in zip(avgs, gold) if a is not None]
print(len(gold))

for path, tree, w in zip(paths, trees, weights):
    if w==0:
        continue
    tree = [t for a,t in zip(avgs, tree) if a is not None]
    f1_metric = discoF1()
    f1_metric(tree, gold)
    print(path)
    print(f1_metric)
    print(f1_metric.corpus_uf1[0])
    print('-'*30)
f1_metric = discoF1()
f1_metric(list(filter(lambda x: x is not None, avgs)), gold)
print('avg')
print(f1_metric)
print('-'*30)