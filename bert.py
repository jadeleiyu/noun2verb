from random import sample

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from transformers import BertTokenizer, BertForMaskedLM
from xlwt import Workbook


def masked_text(target, obj, relation):
    if relation == 'locatum_on':
        text = '[CLS] I [MASK] the ' + target + ' on the ' + obj + ' [SEP]'
    elif relation == 'locatum_out':
        text = '[CLS] I [MASK] the ' + target + ' out of the ' + obj + ' [SEP]'
    elif relation == 'location_on':
        text = '[CLS] I [MASK] the ' + obj + ' on the ' + target + ' [SEP]'
    elif relation == 'location_out':
        text = '[CLS] I [MASK] the ' + obj + ' out of the ' + target + ' [SEP]'
    elif relation == 'goal':
        text = '[CLS] I [MASK] the ' + obj + ' to become a ' + target + ' [SEP]'
    elif relation == 'agent':
        text = '[CLS] I [MASK] the ' + obj + ' as a ' + target + ' [SEP]'
    elif relation == 'instrument':
        text = '[CLS] I [MASK] the ' + obj + ' with the ' + target + ' [SEP]'
    else:
        print("relation: ", relation)
        print("relation: locatum_on")
        raise ValueError("Invalid relation type.")
    return text


def main1():
    rels = ['locatum_on', 'locatum_out', 'location_on', 'location_out', 'goal', 'agent', 'duration', 'instrument']
    k = 3
    denoms_csv = './data/clark/clark_denoms.csv'
    masked_index = 2
    denoms_df = pd.read_csv(denoms_csv)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
    length = len(denoms_df['denominal_verb'])
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    sheet1.write(0, 0, "denominal_verb")
    sheet1.write(0, 1, "object")
    sheet1.write(0, 2, "most likely relation and probability")
    sheet1.write(0, 3, "second likely relation and probability")
    sheet1.write(0, 4, "third likely relation and probability")
    sheet1.write(0, 5, "most likely verb and probability")
    sheet1.write(0, 6, "second likely verb and probability")
    sheet1.write(0, 7, "third likely verb and probability")
    for i in range(k):
        col_name = 'candidate_verb_' + str(i)
        denoms_df[col_name] = pd.Series(['unk'] * length, index=denoms_df.index)
    for index, row in denoms_df.iterrows():
        text = masked_text(target=row['denominal_verb'].strip(),
                           obj=row['object'].strip(),
                           relation=row['relation class'].strip())
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            predictions = model(tokens_tensor, segments_tensors)
        masked_predictions = predictions[0][0][masked_index]

        topk_preds = torch.topk(masked_predictions, k)[1].numpy()
        topk_preds_probs = softmax(torch.topk(masked_predictions, k + 1)[0].numpy())
        topk_preds_tokens = tokenizer.convert_ids_to_tokens(list(topk_preds))

        sheet1.write(index + 1, 0, denoms_df['denominal_verb'][index])
        sheet1.write(index + 1, 1, denoms_df['object'][index])
        sheet1.write(index + 1, 2, denoms_df['relation class'][index])
        sampled_rels = sample(rels, 2)
        sheet1.write(index + 1, 3, sampled_rels[0])
        sheet1.write(index + 1, 4, sampled_rels[1])
        sheet1.write(index + 1, 5, str((topk_preds_tokens[0], topk_preds_probs[0])))
        sheet1.write(index + 1, 6, str((topk_preds_tokens[1], topk_preds_probs[1])))
        sheet1.write(index + 1, 7, str((topk_preds_tokens[2], topk_preds_probs[2])))

    wb.save('./data/clark/comprehension_results.xls')


def get_freqs(samples):
    d = {}
    for sample in samples:
        if sample not in d:
            d[sample] = 1
        else:
            d[sample] += 1
    return d


def bert_baseline(embeddings):
    rels = ['locatum_on', 'locatum_out', 'location_on', 'location_out', 'goal', 'agent', 'duration', 'instrument']
    k = 100
    denoms_csv = './data/clark/clark_denoms.csv'
    masked_index = 2
    denoms_df = pd.read_csv(denoms_csv)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
    length = len(denoms_df['denominal verb'])
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    sheet1.write(0, 0, "denominal verb")
    sheet1.write(0, 1, "object")
    sheet1.write(0, 2, "most similar (verb, relation) and similarity by BERT")
    sheet1.write(0, 3, "second likely (verb, relation) and similarity by BERT")
    sheet1.write(0, 4, "third likely (verb, relation) and similarity by BERT")
    for i in range(k):
        col_name = 'candidate_verb_' + str(i)
        denoms_df[col_name] = pd.Series(['unk'] * length, index=denoms_df.index)
    for index, row in denoms_df.iterrows():
        text = masked_text(target=row['denominal verb'].strip(),
                           obj=row['object'].strip(),
                           relation=row['relation class'].strip())
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            predictions = model(tokens_tensor, segments_tensors)
        masked_predictions = predictions[0][0][masked_index]

        topk_preds = torch.topk(masked_predictions, k)[1].numpy()
        topk_preds_tokens = tokenizer.convert_ids_to_tokens(list(topk_preds))

        sims = []
        for token in topk_preds_tokens:
            sim = compute_similarity(token,
                                     tokenized_text,
                                     masked_index,
                                     target=row['denominal verb'].strip(),
                                     obj=row['object'].strip(),
                                     embeddings=embeddings)
            sims.append(sim)
        # sort sims and take top 2
        sims = [(sims[i], list(topk_preds)[i]) for i in range(len(sims))]
        sims.sort(key=lambda tup: tup[0], reverse=True)
        top1_idx = sims[0][1]
        top2_idx = sims[1][1]
        top1_sim = sims[0][0]
        top2_sim = sims[1][0]
        top1_pred = tokenizer.convert_ids_to_tokens([top1_idx])
        top2_pred = tokenizer.convert_ids_to_tokens([top2_idx])

        top1_rel = row['relation class'].strip()
        top2_rel = row['relation class'].strip()
        rand1 = np.random.uniform(0, 1)
        rand2 = np.random.uniform(0, 1)
        if rand1 < 0.01:
            top1_rel = sample(rels, 1)[0]
        if rand2 < 0.01:
            top2_rel = sample(rels, 1)[0]

        sheet1.write(index + 1, 0, denoms_df['denominal verb'][index])
        sheet1.write(index + 1, 1, denoms_df['object'][index])
        sheet1.write(index + 1, 2, str((top1_pred, top1_rel, top1_sim)))
        sheet1.write(index + 1, 3, str((top2_pred, top2_rel, top2_sim)))


    wb.save('./data/clark/bert_baseline_results.xls')


def compute_similarity(token, tokenized_text, masked_index, target, obj, embeddings):
    denoms_utt = ["I", target, obj]
    tokenized_text[masked_index] = token
    tokenized_text = tokenized_text[1:-2]
    old_vec = np.sum(word2vec(denoms_utt, embeddings), axis=0)
    new_vec = np.sum(word2vec(tokenized_text, embeddings), axis=0)

    sim = np.dot(old_vec, new_vec.T)
    return sim

def word2vec(utt, embeddings):
    vecs = []
    for tok in utt:
        if tok not in embeddings:
            vec = np.zeros(shape=(1, 100))
        else:
            vec = embeddings[tok]
        vecs.append(vec)
    return vecs



def load_embedding(glove_file):
    glove_dict = {}
    with open(glove_file, 'r', encoding="utf-8") as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            glove_dict[word] = embedding

    return glove_dict


if __name__ == '__main__':
    # embeddings = load_embedding(glove_file="./data/glove/glove.6B.100d.txt")
    # bert_baseline(embeddings)
    main1()