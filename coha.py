import pickle
import re

import pandas as pd
from tqdm import tqdm

relations = ['locatum_on', 'locatum_out', 'location_on', 'location_out', 'agent', 'goal', 'duration', 'instrument']


def verb_pattern_make(targets):
    target_pattern_verb = '(?:' + '|'.join(targets) + ')_v\w+\s'
    det_pron_pattern = '[a-zA-Z]+_a\w+\s'
    noun_pattern = '[a-zA-Z]+_n\w+\s'
    query_pattern = re.compile('%s(?:%s)?%s' % (target_pattern_verb, det_pron_pattern, noun_pattern))
    return query_pattern


def locatum_on_pattern_make(targets):
    target_pattern_noun = '(?:' + '|'.join(targets) + ')_n\w+\s'
    det_pron_pattern = '[a-zA-Z]+_a\w+\s'
    verb_pattern = '[a-zA-Z]+_v\w+\s'
    noun_pattern = '[a-zA-Z]+_n\w+\s'
    paraphrase_pattern = re.compile('%s(?:%s)?%s(?:on|onto)_\w+\s%s%s' % (
        verb_pattern, det_pron_pattern, target_pattern_noun, det_pron_pattern, noun_pattern))

    return paraphrase_pattern


def locatum_out_pattern_make(targets):
    target_pattern_noun = '(?:' + '|'.join(targets) + ')_n\w+\s'
    det_pron_pattern = '[a-zA-Z]+_a\w+\s'
    verb_pattern = '[a-zA-Z]+_v\w+\s'
    noun_pattern = '[a-zA-Z]+_n\w+\s'
    paraphrase_pattern = re.compile('%s(?:%s)?%s(?:out|from)_\w+\s%s%s' % (
        verb_pattern, det_pron_pattern, target_pattern_noun, det_pron_pattern, noun_pattern))
    return paraphrase_pattern


def location_on_pattern_make(targets):
    target_pattern_noun = '(?:' + '|'.join(targets) + ')_n\w+\s'
    det_pron_pattern = '[a-zA-Z]+_a\w+\s'
    verb_pattern = '[a-zA-Z]+_v\w+\s'
    noun_pattern = '[a-zA-Z]+_n\w+\s'
    paraphrase_pattern = re.compile('%s(?:%s)?%s(?:on|onto)_\w+\s%s%s' % (
        verb_pattern, det_pron_pattern, noun_pattern, det_pron_pattern, target_pattern_noun))

    return paraphrase_pattern


def location_out_pattern_make(targets):
    target_pattern_noun = '(?:' + '|'.join(targets) + ')_n\w+\s'
    det_pron_pattern = '[a-zA-Z]+_a\w+\s'
    verb_pattern = '[a-zA-Z]+_v\w+\s'
    noun_pattern = '[a-zA-Z]+_n\w+\s'
    paraphrase_pattern = re.compile('%s(?:%s)?%s(?:out|from)_\w+\s%s%s' % (
        verb_pattern, det_pron_pattern, noun_pattern, det_pron_pattern, target_pattern_noun))
    return paraphrase_pattern


def agent_pattern_make(targets):
    target_pattern_noun = '(?:' + '|'.join(targets) + ')_n\w+\s'
    det_pron_pattern = '[a-zA-Z]+_a\w+\s'
    verb_pattern = '[a-zA-Z]+_v\w+\s'
    noun_pattern = '[a-zA-Z]+_n\w+\s'
    paraphrase_pattern = re.compile('%s(?:%s)?%s(?:as|like)_\w+\s%s%s' % (
        verb_pattern, det_pron_pattern, noun_pattern, det_pron_pattern, target_pattern_noun))
    return paraphrase_pattern


def goal_pattern_make(targets):
    target_pattern_noun = '(?:' + '|'.join(targets) + ')_n\w+\s'
    det_pron_pattern = '[a-zA-Z]+_a\w+\s'
    verb_pattern = '[a-zA-Z]+_v\w+\s'
    noun_pattern = '[a-zA-Z]+_n\w+\s'
    paraphrase_pattern = re.compile('%s(?:%s)?%s(?:become|be)_\w+\s%s%s' % (
        verb_pattern, det_pron_pattern, noun_pattern, det_pron_pattern, target_pattern_noun))
    return paraphrase_pattern


def duration_pattern_make(targets):
    target_pattern_noun = '(?:' + '|'.join(targets) + ')_n\w+\s'
    det_pron_pattern = '[a-zA-Z]+_a\w+\s'
    verb_pattern = '[a-zA-Z]+_v\w+\s'
    noun_pattern = '[a-zA-Z]+_n\w+\s'
    paraphrase_pattern = re.compile('%s(?:%s)?%sduring_\w+\s%s%s' % (
        verb_pattern, det_pron_pattern, noun_pattern, det_pron_pattern, target_pattern_noun))
    return paraphrase_pattern


def instrument_pattern_make(targets):
    target_pattern_noun = '(?:' + '|'.join(targets) + ')_n\w+\s'
    det_pron_pattern = '[a-zA-Z]+_a\w+\s'
    verb_pattern = '[a-zA-Z]+_v\w+\s'
    noun_pattern = '[a-zA-Z]+_n\w+\s'
    paraphrase_pattern = re.compile('%s(?:%s)?%s(?:with|by|through|using|via)_\w+\s%s%s' % (
        verb_pattern, det_pron_pattern, noun_pattern, det_pron_pattern, target_pattern_noun))
    return paraphrase_pattern


def pattern_extraction(query_utterance_df, paraphrase_utterance_df, filename, targets):
    year = filename.split('/')[-1].split('_')[1]
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        tagged_words = ['_'.join(line.split('\t')[1:]).strip('\n') for line in lines]
        text = ' '.join(tagged_words) + ' '
        query_pattern_maker = globals()['verb_pattern_make']


        query_pattern = query_pattern_maker(targets)
        query_utterances = query_pattern.findall(text)
        l_q = len(query_utterances)
        query_utterance_df['utterance'] += query_utterances
        query_utterance_df['year'] += [year] * l_q


        for rel in relations:
            maker_name = rel + '_pattern_make'
            paraphrase_pattern_maker = globals()[maker_name]
            paraphrase_pattern = paraphrase_pattern_maker(targets)
            paraphrase_utterances = paraphrase_pattern.findall(text)
            l_p = len(paraphrase_utterances)

            paraphrase_utterance_df['utterance'] += paraphrase_utterances
            paraphrase_utterance_df['year'] += [year] * l_p
            paraphrase_utterance_df['relation'] += [rel] * l_p
    except:
        print("error in file: ", filename)
        pass


def main():
    query_utterance_df = {
        'utterance': [],
        'year': []

    }
    paraphrase_utterance_df = {
        'utterance': [],
        'year': [],
        'relation': []

    }
    file_list = pickle.load(open("./data/historical/coha_file_list.p", 'rb'))
    targets = list(pd.read_csv('./data/historical/n2v_historical.csv', encoding='latin1')['word'])
    for file_name in tqdm(file_list):
        pattern_extraction(query_utterance_df, paraphrase_utterance_df, file_name, targets)
    query_utterance_df = pd.DataFrame(query_utterance_df)
    paraphrase_utterance_df = pd.DataFrame(paraphrase_utterance_df)

    query_utterance_df.to_csv('./data/historical/coha_query_utterances.csv', index=False)
    paraphrase_utterance_df.to_csv('./data/historical/coha_paraphrase_utterances.csv', index=False)


if __name__ == '__main__':
    main()
