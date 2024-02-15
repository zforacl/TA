from library.DP4F import words_count_in_constituent
from teachers_guide import teachers
import pickle
from tqdm import tqdm

def isin(constituent1, constituent2):
    if len(constituent1) <= 2:
        if len(constituent2) <= 2:
            return (constituent1[0] >= constituent2[0]) and (constituent1[1] <= constituent2[1])
        else:
            return any(isin(constituent1, constituent2[i:i+2]) for i in range(0, len(constituent2), 2))
    else:
        return all(isin(constituent1[i:i+2], constituent2) for i in range(0, len(constituent1), 2))

class Node:
    def __init__(self, label, span):
        self.label = label
        if words_count_in_constituent(span)==1:
            self.label = str(span[0])
        self.span = span
        self.children = []

    def attach_word(self, word):
        self.label += '='+word

    def add(self, span, label):
        assert isin(span, self.span)
        for child in self.children:
            if isin(span, child.span):
                return child.add(span, label)
                break
        else:
            node = Node(label, span)
            self.children.append(node)
            return node

    def __str__(self):
        if words_count_in_constituent(self.span) == 1:
            return self.label
        return f'({self.label}'+(' ' if len(self.children) else '')+' '.join(map(str, self.children))+')'

def build_tree(spans, words, include_labels=False):
    spans = sorted(spans, key=lambda x: words_count_in_constituent(x[:-1] if include_labels else x), reverse=True)
    if include_labels:
        spans, labels = [s[:-1] for s in spans], [s[-1] for s in spans]
    else:
        labels = [' ']+['.' if words_count_in_constituent(span)==1 else 'O' for span in spans[1:]]
    root = Node(labels[0], spans[0])
    for span, label in zip(spans[1:], labels[1:]):
        node = root.add(span, label)
        if words_count_in_constituent(span)==1:
            node.attach_word(words[span[0]])
    return root

fold = 'test'
WORDS = f'{fold}.words.pkl'
PREDICTION = f'{fold}.prediction.pkl'
GOLD = f'{fold}.gold.pkl'
dataset='lassy_0'
TEACHERS_COUNT = 5
teachers=list(teachers[dataset].keys())[:TEACHERS_COUNT]
teachers=[t if t.endswith('/') else t+'/' for t in teachers]
student=dataset+'.ensemble.'
paths=[student]+teachers
names = ['Ensemble']+[t.split('/')[-2] for t in teachers]

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

for name, tree in tqdm(zip(names, trees)):
    with open('discbrackets/'+name+'.discbracket', 'w') as f:
        for t, w in (zip(tree, words[0])):
            t = t[0]+t[1]
            f.write(str(build_tree(t, w.split()))+'\n')

with open('discbrackets/Gold.discbracket', 'w') as f:
    for t, w in (zip(gold, words[0])):
        cont, disco = t
        if len(disco):
            disco = [sum(d[0], [])+[d[-1]] for d in disco]
        t = cont+disco
        n = max(tt[-2] for tt in t)
        for i in range(n):
            t.append([i, i+1, '.'])
        f.write(str(build_tree(t, w.split(), include_labels=True))+'\n')
