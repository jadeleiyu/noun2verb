This repository contains all implementation code, data and analysis results for the paper "How nouns surface as verbs: Probabilistic frame semantics for word class conversion".

The directory is organized as follows:

  src/: source code of data preporcessing pipeline and model implementation.
  
  data/: dataset of denominal verb usages across different languages and time periods:
  
      eng_adult.csv: denominal verb usages by adults in modern English, with denominal verbs extracted from (Clark et al., 1979), and utterances from iWeb-ENG-2015 corpus.
      eng_children.csv: denominal verb usages by children in modern English, with denominal verbs and utterances extracted from (Clark et al., 1982).
      chn.csv: denominal verb usages in modern mandarin Chinese, with denominal verbs extracted from (Bai et al., 2014), and utterances from iWeb-CHN-2013 corpus.
      eng_hist.csv: English denominal verb usages in history (from 1800 to present), with denominal verbs extracted from Google Syntactic N-Gram corpus, and utterances from the COHA corpus.     
  
  plots/: saved plots directory
  
  Python dependencies should be present in most scientific distributions, and
include:

    numpy
    pandas
    matplotlib
    seaborn
    scipy
    torch (with CUDA support)
    pyro (with CUDA support)
    nltk (with WordNet package)

Author: Lei (Jade) Yu

Data: 2020-12-01
