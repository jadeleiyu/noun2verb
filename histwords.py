import os
import re
from functools import reduce

import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm


def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        (With help from William. Thank you!)
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    base_embed.init_sims()
    other_embed.init_sims()

    # make sure vocabulary and indices are aligned
    models = [base_embed, other_embed]
    aligned_models = align_gensim_models(models, words=words)
    in_base_embed = aligned_models[0]
    in_other_embed = aligned_models[1]

    # get the embedding matrices
    base_vecs = in_base_embed.wv.vectors_norm
    other_vecs = in_other_embed.wv.vectors_norm

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs)
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operationdef smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    #     """Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    #     Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
    #         (With help from William. Thank you!)
    #     First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    #     Then do the alignment on the other_embed model.
    #     Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    #     Return other_embed.
    #     If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    #     """
    #
    #     # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    #     base_embed.init_sims()
    #     other_embed.init_sims()
    #
    #     # make sure vocabulary and indices are aligned
    #     models = [base_embed, other_embed]
    #     aligned_models = align_gensim_models(models, words=words)
    #     in_base_embed = aligned_models[0]
    #     in_other_embed = aligned_models[1]
    # 
    #     # get the embedding matrices
    #     base_vecs = in_base_embed.wv.vectors_norm
    #     other_vecs = in_other_embed.wv.vectors_norm
    #
    #     # just a matrix dot product with numpy
    #     m = other_vecs.T.dot(base_vecs)
    #     # SVD method from numpy
    #     u, _, v = np.linalg.svd(m)
    #     # another matrix operation
    #     ortho = u.dot(v)
    #     # Replace original array with modified one
    #     # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    #     other_embed.wv.vectors_norm = other_embed.wv.syn0 = other_embed.wv.vectors_norm.dot(ortho)
    #     return other_embed
    ortho = u.dot(v)
    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    other_embed.wv.vectors_norm = other_embed.wv.syn0 = other_embed.wv.vectors_norm.dot(ortho)
    return other_embed


def align_gensim_models(models, words=None):
    """
    Returns the aligned/intersected models from a list of gensim word2vec models.
    Generalized from original two-way intersection as seen above.

    Also updated to work with the most recent version of gensim
    Requires reduce from functools

    In order to run this, make sure you run 'model.init_sims()' for each model before you input them for alignment.

    ##############################################
    ORIGINAL DESCRIPTION
    ##############################################

    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocabs = [set(m.wv.vocab.keys()) for m in models]

    # Find the common vocabulary
    common_vocab = reduce((lambda vocab1, vocab2: vocab1 & vocab2), vocabs)
    if words:
        common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...

    # This was generalized from:
    # if not vocab_m1-common_vocab and not vocab_m2-common_vocab and not vocab_m3-common_vocab:
    #   return (m1,m2,m3)
    if all(not vocab - common_vocab for vocab in vocabs):
        print("All identical!")
        return models

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: sum([m.wv.vocab[w].count for m in models]), reverse=True)

    # Then for each model...
    for m in models:

        # Replace old vectors_norm array with new one (with common vocab)
        indices = [m.wv.vocab[w].index for w in common_vocab]

        old_arr = m.wv.vectors_norm

        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors_norm = m.wv.syn0 = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.wv.index2word = common_vocab
        old_vocab = m.wv.vocab
        new_vocab = {}
        for new_index, word in enumerate(common_vocab):
            old_vocab_obj = old_vocab[word]
            new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, count=old_vocab_obj.count)
        m.wv.vocab = new_vocab

    return models


class COHA_Copora_Iterator(object):
    def __init__(self, decade, root_dir, targets):
        self.decade = decade
        self.root_dir = root_dir
        self.targets = targets

    def __iter__(self):
        for filename in os.listdir(self.root_dir):
            if filename.endswith('.txt') and str(self.decade) in filename:
                with open(os.path.join(self.root_dir, filename), 'r') as f:
                    lines = f.readlines()
                tagged_words = ['_'.join(line.split('\t')[1:]).strip('\n') for line in lines]
                text = ' '.join(tagged_words)
                sents = text.split('.')
                target_pattern_verb = re.compile('(?:' + '|'.join(self.targets) + ')_v\w+\s')
                filtered_sents = [sent for sent in sents if not re.search(target_pattern_verb, sent)]
                for sent in filtered_sents:
                    words = [word.split('_')[0] for word in sent.split(' ')]
                    yield words


def train_histwords(decades, root_dir, targets):
    # train word2vec models with target words screened out using gensim
    word2vec_models = []
    for decade in tqdm(decades):
        coha_iterator = COHA_Copora_Iterator(decade, root_dir, targets)
        word2vec_model = gensim.models.Word2Vec(coha_iterator)
        # word2vec_model.init_sims()
        word2vec_models.append(word2vec_model)
    # align word embeddings across time
    base_model = word2vec_models[0]
    aligned_models = []
    for i in tqdm(range(len(word2vec_models))):
        model = word2vec_models[i]
        decade = decades[i]
        aligned_model = smart_procrustes_align_gensim(base_model, model)
        aligned_models.append(aligned_model)
        aligned_model.save('./data/historical/histwords/histwords_coha_' + str(decade))
    return aligned_models


def main_coha():
    decades = list(range(1800, 2000, 10))
    root_dir = './data/historical/COHA_txt_files/'
    targets = list(pd.read_csv('./data/historical/n2v_historical.csv', encoding='latin1')['word'])
    train_histwords(decades, root_dir, targets)


if __name__ == '__main__':
    main_coha()
