from collections import defaultdict
import json


def get_answer(path):
    predict_item = []
    with open(path, 'r', encoding='utf-8') as fin:
        preds = json.load(fin)
        for item in preds:
            itemout = {}
            itemout['text'] = item['text']
            if 'pred_intent' not in item:
                item['pred_intent'] = item['pred_slot'].split(';')[0].split('=')[-1]
            itemout['intent'] = item['pred_intent'].split('=')[-1]
            itemout['slots'] = {}
            for x in item["pred_slot"].split(';'):
                try:
                    a,b = x.rsplit('=', maxsplit=1)
                    a= a.strip()
                    if '=' in a:
                        a = a.split('=')[-1].strip()
                    b = b.strip()
                    if b not in item['text']:
                        continue

                    if a == 'Intent' or b == "":
                        continue

                    itemout['slots'][a] = b
                except:
                    continue
            itemout['gold_intent'] = item['gold_intent']
            predict_item.append(itemout)
    return predict_item


answer_01 = get_answer('./snips__home_share_models_Meta-Llama-3-8B-Instruct_pipeline_0.1.jsonl')
answer_02 = get_answer('./snips__home_share_models_Meta-Llama-3-8B-Instruct_pipeline_0.2.jsonl')
answer_0 = get_answer('./snips__home_share_models_Meta-Llama-3-8B-Instruct_pipeline_0.jsonl')


def merge(answer_list):
    out = [[] for x in range(len(answer_list[0]))]
    for answer in answer_list:
        for idx, x in enumerate(answer):
            out[idx].append(x)
    print(len(out))
    for x in out:
        assert x[0]['text'] == x[1]['text'] 
        assert x[1]['text'] == x[2]['text']
    
    return out

answer_merge = merge([answer_0, answer_01, answer_02])



def essemble(answer_merge):
    answer_out = []
    for answer_same_item in answer_merge:
        if len(answer_same_item) != 3:
            print(answer_same_item[0]['text'])
            continue
        intent_counter = defaultdict(int)
        slot_key_counter = defaultdict(int)
        slot_key_value_counter = {}
        for answer in answer_same_item:
            intent_counter[answer['intent']] += 1
            for x, y in answer['slots'].items():
                slot_key_counter[x] += 1
                if x not in slot_key_value_counter:
                    slot_key_value_counter[x] = defaultdict(int)
                slot_key_value_counter[x][y] += 1
        
        item = {}
        item['text'] = answer_same_item[0]['text']
        max_intent = -1
        for x in intent_counter:
            if intent_counter[x] > max_intent:
                item['intent'] = x
                max_intent = intent_counter[x]
        item['slots'] = {}
        for x in slot_key_counter:
            if slot_key_counter[x] >= len(answer_same_item) // 2 + 1:
                max_slot = -1
                for y in slot_key_value_counter[x]:
                    if slot_key_value_counter[x][y] > max_slot:
                        item['slots'][x] = y
                        max_slot = slot_key_value_counter[x][y]
        answer_out.append(item)
        answer_out[-1]['gold_intent'] = answer_same_item[0]['gold_intent']
    return answer_out

answer_out = essemble(answer_merge)
with open('./snips_out_croprompt-consistency.jsonl', 'w', encoding='utf-8') as fout:
    item_out = []
    for x in answer_out:
        item = {}
        item['text'] = x['text']
        item['gold_intent'] = x['gold_intent']
        item['pred_intent'] = 'Intent={}'.format(x['intent'])
        item['pred_slot'] = ';'.join(['{}={}'.format(a, b) for a, b in x['slots'].items()])
        item_out.append(item)
    json.dump(item_out, fout, ensure_ascii=False, indent=2)
        
            

        





        
        
