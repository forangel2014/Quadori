## improve $p(ins|r_p)$
+ retrain on wikipedia corpus
+ fix A by select instance corresponding to the relation type, then generate B.

## clustering
+ hierarchical clustering

## DPP
+ DPP training

## new metric
+ average bleu score
+ entailment score(mean-mean, mean-max, mean-min)
+ diversity: counting the number of notional words

## STS

$p(sent = r_h + ins|r_p) = p(r_h|r_p)p(ins|r_p) = \sum_{t \in INS}p(r_h|t)p(t|r_p) * p(ins|r_p)$