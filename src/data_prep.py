"""
data_prep.py:
    1. Convert .csv file with raw collocations into Utterance objects
    2. Compute most frequently co-occured subject-verb-object pairs with particular relations and targets
"""
import csv
import itertools
import operator
import os
import pickle

import numpy as np
import pandas as pd
import spacy
import torch
from torch.utils.data import DataLoader, Dataset

rels = ['locatum_on', 'locatum_out', 'location_on', 'location_out', 'goal', 'agent', 'duration', 'instrument']
novelties = ['established', 'novel']


class WordConversionDataSet(Dataset):
    def __init__(self, utterances):
        super(WordConversionDataSet, self).__init__()
        self.utterances = utterances

        # observed variables
        self.subs = [utterance.subject for utterance in self.utterances]
        self.objs = [utterance.object for utterance in self.utterances]
        self.ts = [utterance.target for utterance in self.utterances]

        # latent variables
        self.predicates = [utterance.predicate for utterance in utterances]
        self.relations = [utterance.relation for utterance in utterances]

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index):
        sub = self.subs[index]
        obj = self.objs[index]
        target = self.ts[index]
        relation = self.relations[index]
        predicate = self.predicates[index]

        return sub, obj, target, relation, predicate


class Utterance:
    def __init__(self, subject, obj, target, relation='None', predicate='None'):
        # "the boy dropped the newspaper on the porch" -- is_canonical == True
        # "the boy porched the newspaper" -- is_canonical == False
        self.relation = relation  # "on"
        self.predicate = predicate  # "drop"
        self.subject = subject  # "the boy"
        self.object = obj  # "the newspaper"
        self.target = target  # "the porch"
        # if no relation and predicate provided, then the utterance is denominal (non-canonical)
        if (self.relation == 'None') or (self.predicate == 'None'):
            self.is_canonical = False
        else:
            self.is_canonical = True

    def __str__(self):
        if self.relation == 'locatum_on' or self.relation == 'location_out':
            return ' '.join([self.subject, self.predicate, self.target, self.relation, self.object])
        else:
            return ' '.join([self.subject, self.predicate, self.object, self.relation, self.target])


class Vocab:
    def __init__(self):
        self.word_list = ['<pad>', '<unk>', '<s>', '<\s>']
        self.w2i = {}
        self.i2w = {}
        self.r2i = {'locatum_on': 0, 'locatum_out': 1, 'location_on': 2, 'location_out': 3, 'goal': 4, 'duration': 5,
                    'instrument': 6, 'agent': 7}
        self.count = 0
        self.embedding = None

    def add_vocab(self, vocab_file='./data/clark/words.csv'):
        with open(vocab_file, "r") as f:
            for line in f:
                self.word_list.append(line.split(',')[0])  # only want the word, not the count
        print("read %d words from vocab file" % len(self.word_list))

        for w in self.word_list:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1
        print(self.w2i.keys())

    def add_embedding(self, gloveFile="./data/glove/glove.6B.100d.txt", embed_size=100):
        print("Loading Glove embeddings")
        with open(gloveFile, 'r', encoding="utf-8") as f:
            model = {}
            w_set = set(self.word_list)
            embedding_matrix = np.zeros(shape=(len(self.word_list), embed_size))

            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                if word in w_set:  # only extract embeddings in the word_list
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    model[word] = embedding
                    embedding_matrix[self.w2i[word]] = embedding
                    if len(model) % 1000 == 0:
                        print("processed %d data" % len(model))
        self.embedding = embedding_matrix
        print("%d words out of %d has embeddings in the glove file" % (len(model), len(self.word_list)))


def load_vocab(experiment_type, vocab_dim=100):
    vocab_fn = './data/' + experiment_type + '/vocab_' + str(vocab_dim) + 'd.p'
    vocab = pickle.load(open(vocab_fn, "rb"))
    return vocab


def prepare_data(batched_subs, batched_objs, batched_targets, batched_relations, batched_predicates, vocab):
    # print('r2i: ', vocab.r2i)
    subs = torch.tensor(np.array([vocab.w2i[sub] for sub in batched_subs]))  # [1, 4, 7] shape: (batch_size)
    objs = torch.tensor(np.array([vocab.w2i[obj] for obj in batched_objs]))
    targets = torch.tensor(np.array([vocab.w2i[target] for target in batched_targets]))
    relations = torch.tensor(np.array([vocab.r2i[rel] for rel in batched_relations]))
    predicates = torch.tensor(np.array([vocab.w2i[pred] for pred in batched_predicates]))

    return subs, objs, targets, relations, predicates


def prepare_csvs(utterances, word_csv_file, rel_csv_file):
    subs, objs, preds, rels, targets = [], [], [], [], []
    for utterance in utterances:
        subs.append(utterance.subject)
        objs.append(utterance.object)
        preds.append(utterance.predicate)
        rels.append(utterance.relation)
        targets.append(utterance.target)
    subs.append('None')
    objs.append('None')
    preds.append('None')
    rels.append('None')
    targets.append('None')
    subs = list(set(subs))
    objs = list(set(objs))
    preds = list(set(preds))
    rels = list(set(rels))
    targets = list(set(targets))

    # write subs, objs , preds and targets into word file
    with open(word_csv_file, "w") as file:
        word_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        i = 0
        for sub in subs:
            word_writer.writerow([sub, i])
            i += 1
        for obj in objs:
            word_writer.writerow([obj, i])
            i += 1
        for target in targets:
            word_writer.writerow([target, i])
        for pred in preds:
            word_writer.writerow([pred, i])

    # write rels into rel file
    with open(rel_csv_file, "w") as file:
        rel_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        i = 0
        for rel in rels:
            rel_writer.writerow([rel, i])
            i += 1


def setup_data_loaders(sup_train_set, unsup_train_set, eval_set, test_set, batch_size):
    # ["sup_train", "unsup_train", "test", "eval"]
    data_loaders = {
        "sup_train": DataLoader(WordConversionDataSet(sup_train_set), batch_size=batch_size, shuffle=True),
        "unsup_train": DataLoader(WordConversionDataSet(unsup_train_set), batch_size=batch_size,
                                  shuffle=True),
        "eval": DataLoader(WordConversionDataSet(eval_set), batch_size=batch_size, shuffle=True),
        "test": DataLoader(WordConversionDataSet(test_set), batch_size=batch_size, shuffle=True)}
    return data_loaders


def extract_svo(doc, rel, denoms):
    svo = {}
    for token in doc:
        if token.lemma_ in denoms or token.text in denoms:
            target = token
            svo['target'] = target.lemma_
            break

    for token in doc:
        if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
            pred = token
            svo['pred'] = pred.lemma_
            break
    for token in doc:
        if token.head == pred and token.dep_ == 'nsubj':
            sub = token
            svo['subject'] = sub.lemma_
            break
    if rel == 'locatum_on' or rel == 'location_out':
        for token in doc:
            if token.dep_ == 'pobj' and token.head.pos_ == 'ADP':
                obj = token
                svo['object'] = obj.lemma_
                break
    else:
        for token in doc:
            if token.head == pred and token.dep_ == 'dobj':
                obj = token
                svo['object'] = obj.lemma_
                break
    return svo


def get_freq_svo(df_stats, rel, novelty):
    df = df_stats[df_stats['relation'] == rel]
    df = df[df['novelty'] == novelty]
    freq_max_stats = {}
    for index, row in df.iterrows():
        target = row['target']
        subject = row['subject']
        predicate = row['predicate']
        obj = row['object']
        if target not in freq_max_stats:
            freq_max_stats[target] = {'subject': {}, 'object': {}, 'predicate': {}}
            freq_max_stats[target]['subject'][subject] = 1
            freq_max_stats[target]['object'][obj] = 1
            freq_max_stats[target]['predicate'][predicate] = 1
        else:
            if subject in freq_max_stats[target]['subject']:
                freq_max_stats[target]['subject'][subject] += 1
            else:
                freq_max_stats[target]['subject'][subject] = 1
            if obj in freq_max_stats[target]['object']:
                freq_max_stats[target]['object'][obj] += 1
            else:
                freq_max_stats[target]['object'][obj] = 1
            if predicate in freq_max_stats[target]['predicate']:
                freq_max_stats[target]['predicate'][predicate] += 1
            else:
                freq_max_stats[target]['predicate'][predicate] = 1
    df2 = {
        'denominal verb': [],
        'most frequent subjects': [],
        'most frequent predicates': [],
        'most frequent objects': [],
        'relation': [],
        'novelty': []
    }
    for target in list(freq_max_stats.keys()):
        freq_subjects = sorted(freq_max_stats[target]['subject'].items(), key=operator.itemgetter(1), reverse=True)[:5]
        freq_predicates = sorted(freq_max_stats[target]['predicate'].items(), key=operator.itemgetter(1), reverse=True)[
                          :5]
        freq_objects = sorted(freq_max_stats[target]['object'].items(), key=operator.itemgetter(1), reverse=True)[:5]
        df2['denominal verb'].append(target)
        df2['most frequent subjects'].append(freq_subjects)
        df2['most frequent predicates'].append(freq_predicates)
        df2['most frequent objects'].append(freq_objects)
        df2['relation'].append(rel)
        df2['novelty'].append(novelty)
    df2 = pd.DataFrame(df2)
    return df2


def get_utterances_df(colloc_df, novel_denoms, established_denoms):
    nlp = spacy.load("en")
    stats = {'subject': [], 'predicate': [], 'object': [], 'target': [], 'relation': [],
             'novelty': []}

    for index, row in colloc_df.iterrows():
        rel = row['relation']
        novelty = row['novelty']
        denoms = novel_denoms if novelty == 'novel' else established_denoms
        sent = row['target collocation']
        doc = nlp(sent)
        for sent in doc.sents:
            try:
                svo = extract_svo(sent, rel, denoms)
            except:
                svo = {}
                # print('svo extraction failed')
            if 'subject' in svo and 'object' in svo and 'target' in svo and 'pred' in svo:
                stats['subject'].append(svo['subject'])
                stats['object'].append(svo['object'])
                stats['target'].append(svo['target'])
                stats['predicate'].append(svo['pred'])
                stats['relation'].append(rel)
                stats['novelty'].append(novelty)

    utterances_df = pd.DataFrame(stats)
    return utterances_df


def get_max_stats(utterances_df):
    df_freqs = []
    comb = [rels, novelties]
    for rel, novelty in list(itertools.product(*comb)):
        df_freq = get_freq_svo(utterances_df, rel, novelty)
        df_freqs.append(df_freq)
    max_stats_df = pd.concat(df_freqs)
    return max_stats_df


def generate_utterances(utterances_df):
    novel_df = utterances_df[utterances_df['novelty'] == 'novel']
    established_df = utterances_df[utterances_df['novelty'] == 'established']
    novel_utterances = []
    established_utterances = []
    for index, row in novel_df.iterrows():
        novel_utterance = Utterance(subject=row['subject'],
                                    obj=row['object'],
                                    target=row['target'],
                                    relation=row['relation'],
                                    predicate=row['predicate'])
        novel_utterances.append(novel_utterance)

    for index, row in established_df.iterrows():
        established_utterance = Utterance(subject=row['subject'],
                                          obj=row['object'],
                                          target=row['target'],
                                          relation=row['relation'],
                                          predicate=row['predicate'])
        established_utterances.append(established_utterance)

    return novel_utterances, established_utterances


def main():
    directory = r'./data/clark/'

    # list of novel/established denominal verbs
    novel_denoms = list(pd.read_csv("./data/clark/denoms_novel.csv")['denominal verb'])
    established_denoms = list(pd.read_csv("./data/clark/denoms_established.csv")['denominal verb'])

    # get most frequent svos
    raw_collocations = pd.read_csv('./data/clark/raw_collocations_2000.csv')
    utterances_df = get_utterances_df(raw_collocations, novel_denoms, established_denoms)
    utts_csv_fn = os.path.join(directory, 'utterances_df.csv')
    utterances_df.to_csv(utts_csv_fn, index=False)

    max_stats_df = get_max_stats(utterances_df)
    max_stats_fn = os.path.join(directory, 'max_stats_df.csv')
    max_stats_df.to_csv(max_stats_fn, index=False)

    # generate utterance objects
    novel_utterances, established_utterances = generate_utterances(utterances_df)

    # store list of utterances into .pickle file
    novel_utts_fn = os.path.join(directory, 'novel_utterances.p')
    estab_utts_fn = os.path.join(directory, 'established_utterances.p')
    pickle.dump(novel_utterances, open(novel_utts_fn, 'wb'))
    pickle.dump(established_utterances, open(estab_utts_fn, 'wb'))


def main_csv():
    # def prepare_csvs(utterances, word_csv_file, rel_csv_file):
    novel_utterances = pickle.load(open("./data/clark/novel_utterances.p", 'rb'))
    established_utterances = pickle.load(open("./data/clark/established_utterances.p", 'rb'))
    utterances = novel_utterances + established_utterances

    word_csv_file = './data/clark/words.csv'
    rel_csv_file = './data/clark/rels.csv'

    prepare_csvs(utterances, word_csv_file, rel_csv_file)


def main_vocab():
    vocab = Vocab()
    vocab.add_vocab()
    vocab.add_embedding()

    pickle.dump(vocab, open('./data/clark/vocab_100d.p', "wb"))


if __name__ == '__main__':
    main()
