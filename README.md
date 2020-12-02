This repository contains all implementation code, data and analysis results for the paper "How nouns surface as verbs: Probabilistic frame semantics for word class conversion".

The directory is organized as follows:

  src/: source code of data preporcessing pipeline and model implementation.
  data/: dataset of denominal verb usages across different languages and time periods
      eng_adult.csv: denominal verb usages by adults in modern English
      eng_children.csv: denominal verb usages by children in modern English
      chn.csv: denominal verb usages in modern mandarin Chinese
      eng_hist.csv: English denominal verb usages in history (from 1800 to present)
  results/: cached results to produce all figures and tables in the paper
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
