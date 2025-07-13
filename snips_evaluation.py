from utils.slu_evaluate import evaluate
import json
import sys

gold_set = {}
with open('./data/SNIPS/parsed_test.json', 'r', encoding='utf-8') as fin:
    gold = json.load(fin)
    for item in gold:
        key = item['text']
        gold_set[key] = item['golds']


def parse_answer(item):
    item['pred_slot'] = item['pred_slot'].strip().split('\n')[-1]
    if 'pred_intent' not in item:
        item['pred_intent'] = item["pred_slot"].split(';')[0]
    item['intent'] = item['pred_intent'].split('=')[-1]
    item['slots'] = {}
    for x in item["pred_slot"].split(';'):
        try:
            a,b = x.rsplit('=', maxsplit=1)
            a = a.strip()
            if '=' in a:
                a = a.split('=')[-1].strip()
            b = b.strip()
            if b not in item['text']:
                continue
            if a == 'Intent' or b == "":
                continue
            b = b.replace(':', '')
            item['slots'][a] = b
            
        except:
            continue
    return item

if __name__ == "__main__":
    path = sys.argv[1]
    with open(path, 'r', encoding='utf-8') as fin:
        if 'llama' in path.lower():
            preds = json.load(fin)
        else:
            preds = [json.loads(line) for line in fin.readlines()]
        
        preds_out = []
        golds = []
        for item in preds:
            item = parse_answer(item)
            try:
                if item['text'] in gold_set:
                    gold_item = {
                        'intent': item['gold_intent'],
                        'slots': gold_set[item['text']][0]['slots']
                    }
                else:
                    gold_item = {
                        'intent': item['gold_intent'],
                        'slots': gold_set[item['text'].replace(':', '')][0]['slots']
                    }
            except:
                print('NOT found:', item['text'])
                continue
            preds_out.append(item)
            golds.append(gold_item)
    print(len(preds_out))
    print(evaluate(preds_out, golds))

        


