from TNLCFRS.parser.helper.metric import LikelihoodMetric,  UF1, LossMetric, UAS, discoF1, printTree
import pickle

WORDS = '.words.pkl'
PREDICTION = '.pkl'
GOLD = '.gold.pkl'

pred_path = './lcfrs_4500/dutch/LCFRS_rank_full22022-11-07-20_32_40/'
gold_path = './lcfrs_4500/dutch/LCFRS_rank_full22022-11-07-20_32_33/'

pred_words = pickle.load(open(pred_path+WORDS, 'rb'))
pred_words = list(map(' '.join, pred_words))
gold_words = pickle.load(open(gold_path+WORDS, 'rb'))
gold_words = list(map(' '.join, gold_words))
pred_reorder = [pred_words.index(gword) for gword in gold_words]

pred = pickle.load(open(pred_path+PREDICTION, 'rb'))
pred = [pred[i] for i in pred_reorder]
gold = pickle.load(open(gold_path+GOLD, 'rb'))
print(len(pred))

# pred_gold = list(zip(pred, gold))
# pred_gold = [pg for pg in pred_gold if pg[0] is not None]
# pred, gold = zip(*pred_gold)
# print(len(pred))

f1_metric = discoF1()
f1_metric(pred, gold)

print(f1_metric)