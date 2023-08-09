# coding:utf8
"""
This module computes evaluation metrics for DuReader dataset.
"""

import argparse
import itertools
import json
import sys
import importlib
import zipfile

from collections import Counter
from bleu import BLEUWithBonus
from rouge import RougeL

EMPTY = ''


def read_file(file_name, is_ref=False):
    line_num = 0
    with open(file_name, 'r', encoding='utf-8') as infile:
        line_num += 1
        obj = json.load(infile)
    return obj


def main(args):
    err = None
    metrics = {}
    bleu4, bleu2, rouge_l = 0.0, 0.0, 0.0
    alpha = args.ab
    beta = args.ab
    bleu_eval = BLEUWithBonus(4, alpha=alpha, beta=beta)
    rouge_eval = RougeL(alpha=alpha, beta=beta, gamma=1.2)
    try:
        pred_result = read_file(args.pred_file)
        ref_result = read_file(args.ref_file, is_ref=True)
        for pre, ref in zip(pred_result, ref_result):
            ref = [ref]
            bleu_eval.add_inst(
                pre,
                ref)
            rouge_eval.add_inst(
                pre,
                ref)
        bleu4 = bleu_eval.score()[-1]
        bleu2 = bleu_eval.score()[1]
        rouge_l = rouge_eval.score()
    except ValueError as ve:
        err = ve
    except AssertionError as ae:
        err = ae
    # too keep compatible to leaderboard evaluation.
    metrics['errorMsg'] = 'success' if err is None else err
    metrics['errorCode'] = 0 if err is None else 1
    metrics['data'] = [
            {'type': 'BOTH', 'name': 'ROUGE-L', 'value': round(rouge_l* 100, 2)},
            {'type': 'BOTH', 'name': 'BLEU-4', 'value': round(bleu4 * 100, 2)},
            {'type': 'BOTH', 'name': 'BLEU-4', 'value': round(bleu2 * 100, 2)}
            ]
    print(json.dumps(metrics, ensure_ascii=False).encode('utf8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', help='predict file')
    parser.add_argument('--ref_file', help='reference file')
    parser.add_argument('--ab', type=float, default=1.0,
            help='common value of alpha and beta')
    args = parser.parse_args()
    main(args)
