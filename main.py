import pickle
import random

import pyro
import torch
from pyro.contrib.examples.util import print_and_log
from pyro.infer import SVI, JitTrace_ELBO, JitTraceEnum_ELBO, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import ClippedAdam
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from data_prep import setup_data_loaders, load_vocab, Vocab, Utterance, WordConversionDataSet
from evaluate import evaluate
from models import Generator
from train import train_epoch
from util import Config, get_args


def main_train(args, vocab):
    # random seed setup
    if args.seed is not None:
        pyro.set_rng_seed(args.seed)

    # CUDA for PyTorch
    cuda_available = torch.cuda.is_available()
    if cuda_available and args.cuda:
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        print("using gpu acceleration")

    # generate config for model initialization

    print("Generating Config")
    config = Config(
        word_embeddings=torch.FloatTensor(vocab.embedding),
        decoder_hidden_dim=args.decoder_hidden_dim,
        num_relations=args.num_relations,
        encoder_hidden_dim=args.encoder_hidden_dim,
        aux_loss_multiplier=args.aux_loss_multiplier
    )

    # initialize the generator model
    generator = Generator(config)

    # setup the optimizer
    adam_params = {"lr": args.learning_rate, "betas": (args.beta_1, 0.999)}
    optimizer = ClippedAdam(adam_params)

    # set up the loss(es) for inference. wrapping the guide in config_enumerate builds the loss as a sum
    # by enumerating each class label for the sampled discrete categorical distribution in the model
    if args.enumerate:
        guide = config_enumerate(generator.guide, args.enum_discrete, expand=True)
    else:
        guide = generator.guide
    elbo = (JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO)(max_plate_nesting=1)
    loss_basic = SVI(generator.model, guide, optimizer, loss=elbo)

    # build a list of all losses considered
    losses = [loss_basic]

    # aux_loss: whether to use the auxiliary loss from NIPS 14 paper (Kingma et al)
    if args.aux_loss:
        elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
        loss_aux = SVI(generator.model_identify, generator.guide_identify, optimizer, loss=elbo)
        losses.append(loss_aux)

    # setup data loaders
    if args.experiment_type == 'clark':
        novel_utterances = pickle.load(open("./data/clark/novel_utterances.p", "rb"))
        established_utterances = pickle.load(open("./data/clark/established_utterances.p", "rb"))

        # def setup_data_loaders(sup_train_set, unsup_train_set, eval_set, test_set, batch_size):
        random.shuffle(novel_utterances)
        random.shuffle(established_utterances)

        num_train = len(established_utterances)
        num_eval_test = len(novel_utterances)
        num_sup = int(num_train * args.sup_train_ratio)  # sup_train_ratio: num_sup_train / num_train
        num_unsup = num_train - num_sup
        num_eval = int(num_eval_test * args.eval_ratio)  # eval_ratio: num_eval / (num_eval + num_test)

        sup_train_set = established_utterances[:num_sup]
        unsup_train_set = established_utterances[num_sup:]
        eval_set = novel_utterances[:num_eval]
        test_set = novel_utterances[num_eval:]

        data_loaders = setup_data_loaders(sup_train_set,
                                          unsup_train_set,
                                          eval_set,
                                          test_set,
                                          batch_size=args.batch_size)
    else:
        raise NotImplementedError("Have not implemented this experiment yet.")

    # how often would a supervised batch be encountered during inference
    # e.g. if sup_num is 3000, we would have every 16th = int(50000/3000) batch supervised
    # until we have traversed through the all supervised batches
    periodic_interval_batches = int(1.0 / (1.0 * args.sup_train_ratio))

    # setup the logger if a filename is provided
    log_fn = "./logs/" + args.experiment_type + '/' + args.experiment_name + '.log'
    logger = open(log_fn, "w")

    # run inference for a certain number of epochs
    for i in tqdm(range(0, args.num_epochs)):
        # get the losses for an epoch
        epoch_losses_sup, epoch_losses_unsup = \
            train_epoch(data_loaders=data_loaders,
                        models=losses,
                        periodic_interval_batches=periodic_interval_batches,
                        vocab=vocab)

        # compute average epoch losses i.e. losses per example
        avg_epoch_losses_sup = map(lambda v: v / len(sup_train_set), epoch_losses_sup)
        avg_epoch_losses_unsup = map(lambda v: v / len(unsup_train_set), epoch_losses_unsup)

        # store the loss in the logfile
        str_loss_sup = " ".join(map(str, avg_epoch_losses_sup))
        str_loss_unsup = " ".join(map(str, avg_epoch_losses_unsup))
        str_print = "{} epoch: avg losses {}".format(i, "{} {}".format(str_loss_sup, str_loss_unsup))
        print_and_log(logger, str_print)

    # save trained models
    # torch.save(generator.state_dict(), './models/test_generator_state_dict.pth')

    # do evaluation if needed
    if args.evaluate:
        predict_df = evaluate(generator,
                              eval_data_loader=data_loaders['eval'],
                              vocab=vocab,
                              sample_size=args.eval_sample_size,
                              batch_size=args.batch_size)

        # save the df with predictions (a dictionary, see evaluate.evaluate)
        eval_df_fn = './data/' + args.experiment_type + '/eval_df_' + args.experiment_name + '.p'
        pickle.dump(predict_df, open(eval_df_fn, 'wb'))

    return generator


def main_evaluate(vocab, args):
    # generate config for model initialization

    print("Generating Config")
    generator_config = Config(
        word_embeddings=torch.FloatTensor(vocab.embedding),
        decoder_hidden_dim=args.decoder_hidden_dim,
        num_relations=args.num_relations,
        encoder_hidden_dim=args.encoder_hidden_dim,
        aux_loss_multiplier=args.aux_loss_multiplier
    )

    # initialize the generator model
    generator = Generator(generator_config)

    # load models
    generator.load_state_dict(torch.load('./models/test_generator_state_dict.pth'))

    # load evaluation data set
    novel_utterances = pickle.load(open("./data/clark/novel_utterances.p", "rb"))
    random.shuffle(novel_utterances)
    num_eval_test = len(novel_utterances)
    num_eval = int(num_eval_test * args.eval_ratio)  # eval_ratio: num_eval / (num_eval + num_test)
    eval_set = novel_utterances[:num_eval]
    eval_data_loader = DataLoader(WordConversionDataSet(eval_set), batch_size=args.batch_size, shuffle=True)


    # model evaluations
    eval_stats = evaluate(generator=generator,
                          eval_data_loader=eval_data_loader,
                          vocab=vocab,
                          sample_size=args.eval_sample_size,
                          batch_size=args.batch_size)





if __name__ == '__main__':
    args = get_args()
    vocab = load_vocab(args.experiment_type, vocab_dim=args.vocab_dim)
    # main_train(args, vocab)
    main_evaluate(vocab, args)
