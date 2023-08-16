import argparse
import logging
import re
from datetime import datetime
import os

import numpy as np
import torch
from nltk import bleu, meteor
from rouge_score.rouge_scorer import RougeScorer
from tqdm import tqdm
from src.distinct_n.distinct_n.metrics import distinct_n_corpus_level as distinct_n
from entailment_eval import EntailmentScorer
from inductor import BartInductor, CometInductor, LLMInductor
# import nltk
# nltk.download('wordnet')

FILES = {
    'amie-yago2': 'data/RE-datasets/AMIE-yago2.txt',
    'rules-yago2': 'data/RE-datasets/RuLES-yago2.txt',
    "openrule155": "data/OpenRule155.txt",
    'fewrel': 'data/RE/fewrel-5.txt',
    'semeval': 'data/RE/semeval-5.txt',
    'TREx': 'data/RE/trex-5.txt',
    'nyt10': 'data/RE/nyt10-5.txt',
    'google-re': 'data/RE/google-re-5.txt',
    'wiki80': 'data/RE/wiki80-5.txt',
}


scorer = RougeScorer(['rougeL'], use_stemmer=True)

def rouge(references, hypothesis):
    scores = []
    for reference in references:
        scores.append(
            scorer.score(
                reference, 
                hypothesis)['rougeL'][2]
        )
    
    return max(scores)


class RelationExtractionEvaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = 'cuda:' + self.args.device
        self.entailment_scorer = EntailmentScorer(self.device)
        if self.args.inductor == 'orion' or self.args.inductor == 'quadori':
            self.inductor = BartInductor(
                device=self.device,
                group_beam=self.args.group_beam,
                continue_pretrain_instance_generator=self.args.mlm_training,
                continue_pretrain_hypo_generator=self.args.bart_training,
                if_then=self.args.if_then,
                prompt=args.prompt,
                ssts=args.ssts,
                dpp=args.dpp,
                inductor=self.args.inductor
            )
        elif self.args.inductor == 'comet':
            self.inductor = CometInductor(self.device)
        elif self.args.inductor == 'llm':
            self.inductor = LLMInductor(self.args.model, self.device)

    def clean(self, text):
        segments = text.split('<mask>')
        if len(segments) == 3 and segments[2].startswith('.'):
            return '<mask>'.join(segments[:2]) + '<mask>.'
        else:
            return text
    
    def clean_references(self, texts):
        for i, text in enumerate(texts):
            if text.endswith(" ."):
                texts[i] = text.replace(" .", ".")
        
        return texts

    def self_bleu(self, hypothesis):
        bleus = []
        for i in range(len(hypothesis)):
            bleus.append(bleu(
                hypothesis[:i] + hypothesis[i + 1:],
                hypothesis[i],
                weights=(0.5, 0.5)))

        ret = np.mean(bleus)
        return ret
    
    def evaluate(self, task):
        with torch.no_grad():
            self.metrics = {
                "bleu-4": [],
                "bleu-3": [],
                "bleu-2": [],
                "bleu-1": [],
                "METEOR": [],
                "ROUGE-L": [],
                "entailment score(mean-mean)": [],
                "entailment score(mean-max)": [],
                "entailment score(mean-min)": [],
                "self-BLEU-2": [],
            }
            with open(FILES[task], 'r', encoding='utf-8') as file:
                data = file.readlines()
                with tqdm(total=len(data)) as pbar:
                    for row in data:
                        pbar.update(1)
                        row = row.strip().split('\t')
                        inputs, head, tail, relations = row[0], row[1], row[2], row[3]
                        inputs = inputs.strip()
                        
                        if relations.startswith('[') and relations.endswith(']'):
                            inputs = re.sub("<A>|<B>", "<mask>", inputs)
                            references = [relation.replace('<A>', '<mask>').replace('<B>', '<mask>').lower().strip() for relation in eval(relations)]
                        else:
                            references = [relations.replace('[X]', '<mask>').replace('[Y]', '<mask>').lower().strip()]
                        references = self.clean_references(references)
                        hypothesis = self.inductor.generate(inputs, k=10, topk=10)
                            
                        logger.info("***********Input************")
                        logger.info(inputs)
                        #logger.info("*********Hypothesis*********")
                        logger.info("*********Premises*********")
                        for i, hypo in enumerate(hypothesis):
                            hypothesis[i] = self.clean(hypo.lower().strip())
                            logger.info(hypo)

                        logger.info("****************************")
                        logger.info("*********References*********")
                        logger.info(references)
                        logger.info("****************************")
                        
                        if len(hypothesis) == 0:
                            for k in self.metrics.keys():
                                if k != 'self-BLEU-2':
                                    self.metrics[k].append(0.)

                        else:
                            entailment_scores = []
                            for hypo in hypothesis:
                                try:
                                    self.metrics['bleu-4'].append(
                                        bleu(
                                            [reference.split() for reference in references],
                                            hypo.split(),
                                            weights=(0.25, 0.25, 0.25, 0.25)
                                        )
                                    )
                                except Exception:
                                    logger.warning("Skip bleu-4 in example: {}".format(inputs))
                                    pass

                                try:
                                    self.metrics['bleu-3'].append(
                                        bleu(
                                            [reference.split() for reference in references],
                                            hypo.split(),
                                            weights=(1 / 3, ) * 3
                                        )
                                    )
                                except Exception:
                                    logger.warning("Skip bleu-3 in example: {}".format(inputs))
                                    pass

                                try:
                                    self.metrics['bleu-2'].append(
                                        bleu(
                                            [reference.split() for reference in references],
                                            hypo.split(),
                                            weights=(0.5, 0.5)
                                        )           
                                    )
                                except Exception:
                                    logger.warning("Skip bleu-2 in example: {}".format(inputs))
                                    pass

                                try:
                                    self.metrics['bleu-1'].append(
                                        bleu(
                                            [reference.split() for reference in references],
                                            hypo.split(),
                                            weights=(1.0, )
                                        )
                                    )
                                except Exception:
                                    logger.warning("Skip bleu-1 in example: {}".format(inputs))
                                    pass

                                try:
                                    self.metrics['METEOR'].append(
                                        meteor(
                                            references,
                                            hypo,
                                        )
                                    )
                                except:
                                    logger.warning("Skip METEOR in example: {}".format(inputs))
                                    pass
                                    

                                try:
                                    self.metrics['ROUGE-L'].append(
                                        rouge(
                                            references,
                                            hypo,
                                        )
                                    )
                                except:
                                    logger.warning("Skip ROUGE-L in example: {}".format(inputs))
                                    pass

                                
                                try:
                                    entailment_score = self.entailment_scorer.scoring(inputs, hypo)
                                    self.metrics['entailment score(mean-mean)'].append(entailment_score)
                                    entailment_scores.append(entailment_score)
                                except:
                                   logger.warning("Skip entailment score in example: {}".format(inputs))
                                   pass
                                
                            try:
                                self.metrics['entailment score(mean-max)'].append(max(entailment_scores))
                                logger.info("most entailed hypothesis: {}".format(hypothesis[np.argmax(entailment_scores)]))
                                self.metrics['entailment score(mean-min)'].append(min(entailment_scores))
                            except:
                                logger.warning("Skip entailment score in example: {}.".format(inputs))
                                pass

                            try:
                                self.metrics['self-BLEU-2'].append(
                                    self.self_bleu(
                                        hypothesis,
                                    )
                                )
                            except:
                                logger.warning("Skip self-bleu-2 in example: {}.".format(inputs))
                                pass
                        # break

            self.print(task, self.metrics)

    def eval_references(self, task):
        with torch.no_grad():
            entailment_score = []
            with open(FILES[task], 'r', encoding='utf-8') as file:
                data = file.readlines()
                with tqdm(total=len(data)) as pbar:
                    for row in data:
                        pbar.update(1)
                        row = row.strip().split('\t')
                        inputs, head, tail, relations = row[0], row[1], row[2], row[3]
                        inputs = inputs.strip()
                        
                        if relations.startswith('[') and relations.endswith(']'):
                            inputs = re.sub("<A>|<B>", "<mask>", inputs)
                            references = [relation.replace('<A>', '<mask>').replace('<B>', '<mask>').lower().strip() for relation in eval(relations)]
                        else:
                            references = [relations.replace('[X]', '<mask>').replace('[Y]', '<mask>').lower().strip()]
                        references = self.clean_references(references)
                            
                        logger.info("***********Input************")
                        logger.info(inputs)
                        logger.info("*********References*********")
                        logger.info(references)
                        logger.info("****************************")
                        
                        for ref in references:
                            entailment_score.append(self.entailment_scorer.scoring(inputs, ref))
            
            logger.info("reference entailment score: {}".format(str(np.mean(entailment_score))))

    def print(self, task, metrics):
        logger.info("Task: {}".format(str(task)))
        for k, v in metrics.items():
            logger.info("{}: {}".format(k, str(np.mean(v))))
        scores = [np.mean(metrics[k]) for k in ('bleu-4', 'bleu-3','bleu-2','bleu-1','METEOR','ROUGE-L')]
        logger.info("avg: {}".format(str(np.mean(scores))))

        logger.info("*******************************************************")
        logger.info("*******************************************************")
        logger.info("*******************************************************")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inductor", type=str, default='rule')
    parser.add_argument("--group_beam", type=bool, default=False)
    parser.add_argument("--mlm_training", type=bool, default=False)
    parser.add_argument("--bart_training", type=bool, default=False)
    parser.add_argument("--prompt", type=bool, default=False)
    parser.add_argument("--ssts", type=bool, default=False)
    parser.add_argument("--dpp", type=bool, default=False)
    parser.add_argument("--model", type=str, default="chatgpt")
    parser.add_argument("--if_then", type=bool, default=False)
    parser.add_argument("--task", type=str, default='openrule155')
    parser.add_argument("--log_dir", type=str, default='logs_final/')
    parser.add_argument("--log_name", type=str, default='default_log')
    parser.add_argument("--device", type=str, default='0')
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    logging.basicConfig(
        filename=args.log_dir+args.log_name+'.log',
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        filemode='w',
        level=logging.INFO)
    logger = logging.getLogger(__name__)


    def print_config(config):
        config = vars(config)
        logger.info("**************** MODEL CONFIGURATION ****************")
        for key in sorted(config.keys()):
            val = config[key]
            keystr = "{}".format(key) + (" " * (25 - len(key)))
            logger.info("{} -->   {}".format(keystr, val))
        logger.info("**************** MODEL CONFIGURATION ****************")
    

    print_config(args)
    evaluator = RelationExtractionEvaluator(args)
    evaluator.evaluate(args.task)
    #evaluator.eval_references(args.task)
