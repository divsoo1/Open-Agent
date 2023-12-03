import pandas as pd
from pathlib import Path
import re
import string
import nltk
from nltk.corpus import stopwords
import collections
from nltk.translate.bleu_score import sentence_bleu
from rouge_metric import PyRouge
import numpy as np


class ReferencedEvaluator:
    """ The container class for referenced evaluation
    """
    def __init__(self, tokenizer):
        self.bleu = sentence_bleu
        self.rouge = PyRouge(rouge_n=(1, 2), rouge_l=True)        
        self.metrics_store = pd.DataFrame(columns=["id"]).set_index("id", drop=True)
        self.tokenizer = tokenizer      
        nltk.download('stopwords')
        
    def tokenize(self, text):
        # Convert a string to a sequence of tokens
        text = self.normalize_answer(text)
        toks = self.tokenizer.encode(text)
        #print("\noriginal text: {}\n\nSequence of toks: {}\n\n".format(text, toks))
        return toks
    
    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_bleu(self, prediction, reference):
        references = [self.tokenize(reference)]
        prediction = self.tokenize(prediction)
        bleu1 = self.bleu(references, prediction, weights=(1,0,0,0))
        bleu2 = self.bleu(references, prediction, weights=(0.5,0.5,0,0))
        bleu3 = self.bleu(references, prediction, weights=(0.33,0.33,0.33,0))
        return bleu1, bleu2, bleu3
        
    def compute_rouge(self, prediction, reference):
        reference = [[[self.tokenize(reference)]]]
        prediction = [[self.tokenize(prediction)]]
        scores = self.rouge.evaluate_tokenized(prediction, reference)
        rouge1 = scores['rouge-1']['r']
        rouge2 = scores['rouge-2']['r']
        rougeL = scores['rouge-l']['r']
        rougeLsum = (rouge1 + rouge2 + rougeL) / 3.0
        return rouge1, rouge2, rougeL, rougeLsum


    def compute_f1(self, prediction, reference):
        gold_toks = self.tokenize(reference)
        pred_toks = self.tokenize(prediction)
        #print("\ngold_toks: {}\n\npre_toks: {}\n\n".format(gold_toks, pred_toks))
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return 0,0,0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = 0.0
        if (precision + recall != 0):
            f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1
    
    def evaluate_all_metrics(self, prediction, reference):

        bleu1, bleu2, bleu3 = self.compute_bleu(prediction, reference)
        rouge1, rouge2, rougeL, rougeLsum = self.compute_rouge(prediction, reference)
        precision, recall, f1 = self.compute_f1(prediction, reference)

        metrics_dict = {
            "bleu1": bleu1,
            "bleu2": bleu2,
            "bleu3": bleu3,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "rougeLsum": rougeLsum,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        self.metrics_store = self.metrics_store.append(metrics_dict, ignore_index=True)

        return metrics_dict