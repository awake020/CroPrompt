import os
from typing import List
import pandas as pd

def evaluate(preds: List[dict], golds: List[dict], save_name='tasks/predict.xlsx'):
    '''
    Evaluate intent acc and slot f1 for given pred and gold data

    :param pred: The predicted data
    :param gold: The gold data
    :return: The intent acc and slot f1
    '''
    num_sample = len(golds)

    # Compute intent acc if intent is provided
    if 'intent' in golds[0]:
        pred_intent = [pred['intent'] for pred in preds]
        gold_intent = [gold['intent'] for gold in golds]
        intent_acc = sum([1 if pred == gold else 0 for pred, gold in zip(
            pred_intent, gold_intent)]) / num_sample
    else:
        intent_acc = None

    # Compute slot f1 if slot is provided
    if 'slots' in golds[0]:
        pred_slots = [pred['slots'] for pred in preds]
        gold_slots = [gold['slots'] for gold in golds]
        slot_set = set([name for slots in gold_slots for name in slots])
        slot_f1 = []
        for slot_name in slot_set:
            preds_for_slot = [
                pred[slot_name] if slot_name in pred else None for pred in pred_slots]
            golds_for_slot = [
                gold[slot_name] if slot_name in gold else None for gold in gold_slots]

            proposal_made = [p != None for p in preds_for_slot]
            has_gold_label = [g != None for g in golds_for_slot]
            predict_corrects = [p == g for p, g in zip(
                preds_for_slot, golds_for_slot)]

            tp = sum([int(proposed and correct)
                     for proposed, correct in zip(proposal_made, predict_corrects)])

            num_predicted = sum(proposal_made)
            num_to_recall = sum(has_gold_label)

            precision = tp / num_predicted if num_predicted > 0 else 0
            recall = tp / num_to_recall if num_to_recall > 0 else 0

            f1 = 2 * precision * recall / \
                (precision + recall) if precision + recall > 0 else 0
            slot_f1.append(f1)

        preds_for_sample = [
            set([slot + value for slot, values in pred.items() for value in values]) for pred in pred_slots
        ]
        golds_for_sample = [
            set([slot + value for slot, values in gold.items() for value in values]) for gold in gold_slots
        ]
        correct_for_sample = [
            len(pred.intersection(gold)) for pred, gold in zip(preds_for_sample, golds_for_sample)
        ]
        num_pred_for_sample = map(len, preds_for_sample)
        num_gold_for_sample = map(len, golds_for_sample)
        recall_for_sample = [correct / (1e-8 + num_gold) for correct,
                             num_gold in zip(correct_for_sample, num_gold_for_sample)]
        precision_for_sample = [
            correct / (1e-8 + num_pred) for correct, num_pred in zip(correct_for_sample, num_pred_for_sample)]
        f1_score_for_sample = [
            2 * p * r / (1e-8 + p + r) for p, r in zip(precision_for_sample, recall_for_sample)]
        results = ([{
            'pred': pred,
            'gold': gold,
            'f1': f1,
            'precision': p,
            'recall': r,
            'intent': gold['intent']==pred['intent']
        } for pred, gold, f1, p, r in zip(preds, golds, f1_score_for_sample, precision_for_sample, recall_for_sample)])

        
        print('Overall Acc:', sum([abs(results[i]['f1'] - 1) < 0.001 and results[i]['intent'] for i in range(len(results))]) / len(results))

        slot_f1 = sum(slot_f1) / len(slot_f1)
    else:
        slot_f1 = None

    return intent_acc, slot_f1
