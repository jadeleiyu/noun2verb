import argparse
import itertools
import os
import signal
import time
from contextlib import contextmanager
from tqdm import tqdm
import pandas as pd
import requests
from nltk.corpus import wordnet


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


rels = ['locatum_on', 'locatum_out', 'location_on', 'location_out', 'goal', 'agent', 'duration', 'instrument']
novelties = ['established', 'novel']
base_url = 'https://api.sketchengine.eu/bonito/run.cgi'


def download_collocs(query, args, rel, novelty):
    colloc_df = {
        'left context': [],
        'target collocation': [],
        'right context': [],
        'relation': [],
        'novelty': []
    }
    data = {
        'corpname': 'preloaded/ententen15_tt21',
        'format': 'json',
        'username': 'jadeleiyu',
        'api_key': '73d48ddb5e7f42c591b4a59444d32884',
        'q': query,
        'pagesize': args.page_size
    }
    for i in tqdm(range(args.num_pages)):
        data['fromp'] = i
        try:
            with time_limit(args.time_limit):
                d = requests.get(base_url + '/view', params=data).json()
                for line in d['Lines']:
                    left = line['Left'][0]['str']
                    colloc = line['Kwic'][0]['str']
                    right = line['Right'][0]['str']
                    if left and colloc and right:
                        colloc_df['left context'].append(left)
                        colloc_df['target collocation'].append(colloc)
                        colloc_df['right context'].append(right)
                        colloc_df['relation'].append(rel)
                        colloc_df['novelty'].append(novelty)
        except Exception as e:
            pass
        if len(colloc_df['target collocation']) >= args.examples_cap:
            break
        time.sleep(0.5)
    colloc_df = pd.DataFrame(colloc_df)
    return colloc_df


def make_query(denoms, rel):
    if rel == "locatum_on":
        query = "[tag=\"PP.?|DT\"]?[tag=\"N.*\"][tag=\"V.*\"][tag=\"PP.?|DT\"]?[lemma=\"" + "|".join(
            denoms) + "\"&tag=\"N.*\"][lemma=\"on|onto|in|into\"][tag=\"PP.?|DT\"]?[tag=\"N.*\"]"
    elif rel == "locatum_out":
        query = "[tag=\"PP.?|DT\"]?[tag=\"N.*\"][tag=\"V.*\"][tag=\"PP.?|DT\"]?[lemma=\"" + "|".join(
            denoms) + "\"&tag=\"N.*\"][lemma=\"out|from|off|down\"][tag=\"PP.?|DT\"]?[tag=\"N.*\"]"
    elif rel == "location_on":
        query = "[tag=\"PP.?|DT\"]?[tag=\"N.*\"][tag=\"V.*\"][tag=\"PP.?|DT\"]?[tag=\"N.*\"][lemma=\"on|onto|in|into\"][tag=\"PP.?|DT\"]?[lemma=\"" + "|".join(
            denoms) + "\"&tag=\"N.*\"]"
    elif rel == "location_out":
        query = "[tag=\"PP.?|DT\"]?[tag=\"N.*\"][tag=\"V.*\"][tag=\"PP.?|DT\"]?[tag=\"N.*\"][lemma=\"out|from|off|down\"][tag=\"PP.?|DT\"]?[lemma=\"" + "|".join(
            denoms) + "\"&tag=\"N.*\"]"
    elif rel == "duration":
        query = "[tag=\"PP.?|DT\"]?[tag=\"N.*\"][tag=\"V.*\"][lemma=\"with|at\"][tag=\"N.*|PP.?\"][lemma=\"during\"][]{0,1}[lemma=\"" + "|".join(
            denoms) + "\"&tag=\"N.*\"]"
    elif rel == "agent":
        query = "[tag=\"PP.?|DT\"]?[tag=\"N.*\"][tag=\"V.*\"][tag=\"IN\"]?[tag=\"PP.?|DT\"]?[tag=\"N.*\"][lemma=\"as|like\"][tag=\"PP.?|DT\"]?[lemma=\"" + "|".join(
            denoms) + "\"&tag=\"N.*\"]"
    elif rel == "goal":
        query = "[tag=\"PP.?|DT\"]?[tag=\"N.*\"][tag=\"V.*\"][tag=\"PP.?|DT\"]?[tag=\"N.*\"]" + \
                "[lemma=\"become|into\"][tag=\"PP.?|DT\"]?" + "[lemma=\"" + "|".join(denoms) + "\"&tag=\"N.*\"]"
    elif rel == "instrument":
        query = "[tag=\"PP.?|DT\"]?[tag=\"N.*\"][tag=\"V.*\"][tag=\"PP.?|DT\"]?[tag=\"N.*\"][lemma=\"with|by|through|via|using\"][tag=\"PP.?|DT\"]?[lemma=\"" + "|".join(
            denoms) + "\"&tag=\"N.*\"]"
    else:
        raise ValueError("Invalid relation type!")
    return 'q' + query


def enrich_denoms(denoms):
    enriched_denoms_dict = {}
    for denom in denoms:
        if wordnet.synsets(denom):
            enriched_denoms_dict[denom] = []
            synset_0 = wordnet.synsets(denom)[0]
            for lemma in synset_0.lemma_names():
                enriched_denoms_dict[denom].append(lemma)
            if synset_0.hypernyms():
                hypernym_set = synset_0.hypernyms()[0]
                for lemma in hypernym_set.lemma_names():
                    enriched_denoms_dict[denom].append(lemma)
    enriched_denoms_list = sum([enriched_denoms_dict[denom] for denom in list(enriched_denoms_dict.keys())], [])
    return enriched_denoms_dict, enriched_denoms_list


def main(args):
    directory = r'./data/clark/'
    # list of novel/established denominal verbs
    novel_denoms_df = pd.read_csv("./data/clark/denoms_novel.csv")
    established_denoms_df = pd.read_csv("./data/clark/denoms_established.csv")

    comb = [rels, novelties]
    colloc_dfs = []
    for rel, novelty in list(itertools.product(*comb)):
        if novelty == 'novel':
            denoms = novel_denoms_df[novel_denoms_df['relation class'] == rel]['denominal verb']
        else:
            denoms = established_denoms_df[established_denoms_df['relation class'] == rel]['denominal verb']
        enriched_denoms_dict, enriched_denoms_list = enrich_denoms(denoms)
        query = make_query(rel=rel, denoms=enriched_denoms_list)
        colloc_df = download_collocs(query, args, rel, novelty)
        colloc_dfs.append(colloc_df)
    df = pd.concat(colloc_dfs, ignore_index=True, sort=False)
    fname = os.path.join(directory,
                         'raw_collocations_2000.csv')  # data frame with raw sentences from all (rel, novelty) combinations
    df.to_csv(fname, index=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--page_size', default=200, type=int)
    parser.add_argument('--num_pages', default=10, type=int)
    parser.add_argument('--time_limit', default=800, type=int)
    parser.add_argument('--examples_cap', default=2000, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
