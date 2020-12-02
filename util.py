import argparse
from collections import namedtuple

# Config = namedtuple('parameters',
#                     ['vocab_size', 'pretrained_embedding','relation_hidden_dim', 'num_relations', 'num_predicates', 'z_hidden_dim', 'z_dim',
#                      'aux_loss_multiplier', 'use_cuda'
#                      ])

Config = namedtuple('parameters',
                    ['word_embeddings', 'encoder_hidden_dim', 'decoder_hidden_dim', 'num_relations', 'aux_loss_multiplier']
                    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_relations', default=8, type=int)
    parser.add_argument('--encoder_hidden_dim', default=20, type=int)
    parser.add_argument('--decoder_hidden_dim', default=20, type=int)
    parser.add_argument('--sup_train_ratio', default=0.1, type=float)
    parser.add_argument('--eval_ratio', default=0.5, type=float)
    parser.add_argument('--vocab_dim', default=100, type=int)
    parser.add_argument('--homepath', type=str, default='/home/jingyihe/word_conversion')
    parser.add_argument('--seed', default=0, type=int,
                        help="seed for controlling randomness in this example")
    parser.add_argument('--cuda', action='store_true',
                        help="use GPU(s) to speed up training")
    parser.add_argument('--jit', action='store_true',
                        help="use PyTorch jit to speed up training")
    parser.add_argument('-n', '--num_epochs', default=20, type=int,
                        help="number of epochs to run")
    parser.add_argument('--aux_loss', action="store_true",
                        help="whether to use the auxiliary loss from NIPS 14 paper "
                             "(Kingma et al). It is not used by default ")
    parser.add_argument('-alm', '--aux_loss_multiplier', default=46, type=float,
                        help="the multiplier to use with the auxiliary loss")
    parser.add_argument('-enum', '--enum_discrete', default="parallel",
                        help="parallel, sequential or none. uses parallel enumeration by default")
    parser.add_argument('--learning_rate', default=0.005, type=float,
                        help="learning rate for Adam optimizer")
    parser.add_argument('--beta_1', default=0.9, type=float,
                        help="beta-1 parameter for Adam optimizer")
    parser.add_argument('--batch_size', default=16, type=int,
                        help="number of instances to be considered in a batch")
    parser.add_argument('--experiment_name', default='exp_0', type=str)
    parser.add_argument('--experiment_type', default='clark', type=str)
    parser.add_argument('--enumerate', action="store_true",
                        help="whether to use enumerate to reduce variance")
    parser.add_argument('--evaluate', action="store_true", help="whether to perform evaluation after training")
    parser.add_argument('--eval_sample_size', default=1000, type=int)
    return parser.parse_args()

