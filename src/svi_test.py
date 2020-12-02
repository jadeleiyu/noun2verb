import argparse
from collections import namedtuple

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer
import torch
from pyro.contrib.examples.util import print_and_log
from pyro.infer import SVI, JitTrace_ELBO, JitTraceEnum_ELBO, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import ClippedAdam
from scipy.special import softmax
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from train import diagonalize

Config = namedtuple('parameters',
                    ['word_embeddings', 'encoder_hidden_dim', 'decoder_hidden_dim', 'num_relations',
                     'num_predicates', 'batch_size']
                    )


class TestingDataSet(Dataset):
    def __init__(self, utterances):
        self.utterances = utterances

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index):
        sub = self.utterances[index][0]
        obj = self.utterances[index][1]
        tar = self.utterances[index][2]
        rel = self.utterances[index][3]
        pred = self.utterances[index][4]

        return sub, obj, tar, rel, pred


class RealModel:
    def __init__(self, rel_embeddings, word_embeddings,
                 rel_prior=softmax(np.array([1, 2, 3, 4, 5, 6, 7])),
                 pred_prior=softmax(np.array([1.] * 250 + [2.] * 250 + [3.] * 250 + [4.] * 250)),
                 test_size=1000):
        self.rel_prior = rel_prior
        self.pred_prior = pred_prior
        self.test_size = test_size
        self.rel_embeddings = rel_embeddings
        self.word_embeddings = word_embeddings

    def generate_data(self):
        data = []
        # num_relations = 7
        relation_prior = self.rel_prior
        # num_predicates = 100
        predicate_prior = self.pred_prior
        z_mean = np.array([0.] * 100)
        z_cov = np.identity(100, dtype=float)

        # vocab_size = 1000
        sub_decode = np.random.rand(1000, 100) * 40.0 + 20.0
        obj_decode = np.random.rand(1000, 100) * 10.0 - 8.0
        tar_decode = np.random.rand(1000, 100) * 50.0 + 12.0

        for i in range(self.test_size):
            oh_rel = np.random.multinomial(1, relation_prior, 1)[0]
            oh_pred = np.random.multinomial(1, predicate_prior, 1)[0]
            z = np.random.multivariate_normal(z_mean, z_cov)  # shape: (100, )
            rel_emb = np.dot(oh_rel.T, self.rel_embeddings).reshape(100)  # shape: (100, )
            pred_emb = np.dot(oh_pred.T, self.word_embeddings).reshape(100)  # shape: (100, )

            p_sub = softmax(np.dot(sub_decode, (z + rel_emb + pred_emb).T)).reshape(1000)
            p_obj = softmax(np.dot(obj_decode, (z + rel_emb + pred_emb).T)).reshape(1000)
            p_tar = softmax(np.dot(tar_decode, (z + rel_emb + pred_emb).T)).reshape(1000)

            oh_sub = np.random.multinomial(1, p_sub, 1).reshape(1000)
            oh_obj = np.random.multinomial(1, p_obj, 1).reshape(1000)
            oh_tar = np.random.multinomial(1, p_tar, 1).reshape(1000)

            sub = unhot(oh_sub)
            obj = unhot(oh_obj)
            tar = unhot(oh_tar)
            rel = unhot(oh_rel)
            pred = unhot(oh_pred)
            # print(sub, obj, tar)

            data.append([sub, obj, tar, rel, pred])
        return data


def setup_data_loaders(sup_train_set, unsup_train_set, eval_set, test_set, batch_size):
    data_loaders = {
        "sup_train": DataLoader(TestingDataSet(sup_train_set), batch_size=batch_size, shuffle=True),
        "unsup_train": DataLoader(TestingDataSet(unsup_train_set), batch_size=batch_size,
                                  shuffle=True),
        "eval": DataLoader(TestingDataSet(eval_set), batch_size=batch_size, shuffle=True),
        "test": DataLoader(TestingDataSet(test_set), batch_size=batch_size, shuffle=True)}
    return data_loaders


def unhot(vec):
    """ takes a one-hot vector and returns the corresponding integer """
    assert np.sum(vec) == 1  # this assertion shouldn't fail, but it did...
    return list(vec).index(1)


def training(args, rel_embeddings, word_embeddings):
    if args.seed is not None:
        pyro.set_rng_seed(args.seed)
    # CUDA for PyTorch
    cuda_available = torch.cuda.is_available()
    if (cuda_available and args.cuda):
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        print("using gpu acceleration")

    print("Generating Config")
    config = Config(
        word_embeddings=torch.FloatTensor(word_embeddings),
        decoder_hidden_dim=args.decoder_hidden_dim,
        num_relations=7,
        encoder_hidden_dim=args.encoder_hidden_dim,
        num_predicates=1000,
        batch_size=args.batch_size
    )

    # initialize the generator model
    generator = SimpleGenerator(config)

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

    # prepare data
    real_model = RealModel(rel_embeddings, word_embeddings)
    data = real_model.generate_data()
    sup_train_set = data[:100]
    unsup_train_set = data[100:700]
    eval_set = data[700:900]
    test_set = data[900:]

    data_loaders = setup_data_loaders(sup_train_set,
                                      unsup_train_set,
                                      eval_set,
                                      test_set,
                                      batch_size=args.batch_size)

    num_train = len(sup_train_set) + len(unsup_train_set)
    num_eval = len(eval_set)
    num_test = len(test_set)

    # how often would a supervised batch be encountered during inference
    # e.g. if sup_num is 3000, we would have every 16th = int(50000/3000) batch supervised
    # until we have traversed through the all supervised batches
    periodic_interval_batches = int(1.0 * num_train / len(sup_train_set))

    # setup the logger if a filename is provided
    log_fn = "./logs/" + args.experiment_type + '/' + args.experiment_name + '.log'
    logger = open(log_fn, "w")

    # run inference for a certain number of epochs
    for i in tqdm(range(0, args.num_epochs)):
        # get the losses for an epoch
        epoch_losses_sup, epoch_losses_unsup = \
            train_epoch(data_loaders=data_loaders,
                        models=losses,
                        periodic_interval_batches=periodic_interval_batches)

        # compute average epoch losses i.e. losses per example
        avg_epoch_losses_sup = map(lambda v: v / len(sup_train_set), epoch_losses_sup)
        avg_epoch_losses_unsup = map(lambda v: v / len(unsup_train_set), epoch_losses_unsup)

        # store the loss and validation/testing accuracies in the logfile
        str_loss_sup = " ".join(map(str, avg_epoch_losses_sup))
        str_loss_unsup = " ".join(map(str, avg_epoch_losses_unsup))
        str_print = "{} epoch: avg losses {}".format(i, "{} {}".format(str_loss_sup, str_loss_unsup))
        print_and_log(logger, str_print)

    # save trained models
    torch.save(generator.state_dict(), './models/test_generator_state_dict.pth')
    return generator


def train_epoch(data_loaders, models, periodic_interval_batches):
    num_models = len(models)

    # compute number of batches for an epoch
    sup_batches = len(data_loaders["sup_train"])
    unsup_batches = len(data_loaders["unsup_train"])
    batches_per_epoch = sup_batches + unsup_batches

    # initialize variables to store loss values
    epoch_losses_sup = [0.] * num_models
    epoch_losses_unsup = [0.] * num_models

    # setup the iterators for training data loaders
    sup_iter = iter(data_loaders["sup_train"])
    unsup_iter = iter(data_loaders["unsup_train"])

    # count the number of supervised batches seen in this epoch
    ctr_sup = 0
    for i in tqdm(range(batches_per_epoch)):

        # whether this batch is supervised or not
        is_supervised = (i % periodic_interval_batches == 1) and ctr_sup < sup_batches

        # extract the corresponding batch
        if is_supervised:
            (subs, objs, targets, relations, predicates) = next(sup_iter)
            ctr_sup += 1
        else:
            (subs, objs, targets, relations, predicates) = next(unsup_iter)

        # convert data into torch tensors
        subs = torch.tensor(subs)
        objs = torch.tensor(objs)
        targets = torch.tensor(targets)
        relations = torch.tensor(relations)
        predicates = torch.tensor(predicates)

        # run the inference for each loss with supervised or un-supervised
        # data as arguments
        for model_id in range(num_models):
            if is_supervised:
                new_loss = models[model_id].step(subs, objs, targets, relations, predicates)
                epoch_losses_sup[model_id] += new_loss
            else:
                new_loss = models[model_id].step(subs, objs, targets)
                epoch_losses_unsup[model_id] += new_loss

    # return the values of all losses
    return epoch_losses_sup, epoch_losses_unsup


class SimpleEncoder(nn.Module):
    def __init__(self, config):
        super(SimpleEncoder, self).__init__()
        self.vocab_size = config.word_embeddings.shape[0]
        self.embedding_dim = config.word_embeddings.shape[1]
        self.hidden_dim = config.encoder_hidden_dim
        self.num_relations = config.num_relations
        self.num_predicates = config.num_predicates

        self.embedding = nn.Embedding.from_pretrained(config.word_embeddings, freeze=True)  # must be a 2d float tensor
        self.enc_linear = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.pred_linear = nn.Linear(self.hidden_dim, self.num_predicates)
        self.rel_linear = nn.Linear(self.hidden_dim, self.num_relations)
        self.z_linear_loc = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.z_linear_scale = nn.Linear(self.hidden_dim, self.embedding_dim)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax()
        self.softplus = nn.Softplus()

    def forward(self, s, o, t):
        h_s, h_o, h_t = self.embedding(s), self.embedding(o), self.embedding(t)
        h = self.tanh(self.enc_linear(h_s + h_o + h_t))
        h = self.dropout(h)

        prob_pred = self.softmax(self.pred_linear(h))
        prob_rel = self.softmax(self.rel_linear(h))
        z_loc = self.z_linear_loc(h)
        z_scale = diagonalize(self.softplus(self.z_linear_scale(h)))

        # print("shape of prob_pred: ", prob_pred.shape)

        return prob_pred, prob_rel, z_loc, z_scale


class SimpleDecoder(nn.Module):
    def __init__(self, config):
        super(SimpleDecoder, self).__init__()
        self.vocab_size = config.word_embeddings.shape[0]
        self.embedding_dim = config.word_embeddings.shape[1]
        self.hidden_dim = config.decoder_hidden_dim

        self.embedding = nn.Embedding.from_pretrained(config.word_embeddings, freeze=True)  # must be a 2d float tensor
        self.rel_embedding = nn.Embedding(config.num_relations, self.embedding_dim)

        self.dec_linear = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.sub_linear = nn.Linear(self.hidden_dim, self.vocab_size)
        self.obj_linear = nn.Linear(self.hidden_dim, self.vocab_size)
        self.target_linear = nn.Linear(self.hidden_dim, self.vocab_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout()

    def forward(self, r, v, z):
        h_r = self.rel_embedding(r)
        h_v = self.embedding(v)
        h = self.tanh(self.dec_linear(h_r + h_v + z))
        h = self.dropout(h)

        prob_sub = self.softmax(self.sub_linear(h))
        prob_obj = self.softmax(self.obj_linear(h))
        prob_target = self.softmax(self.target_linear(h))

        # print("shape of prob_sub: ", prob_sub.shape)

        return prob_sub, prob_obj, prob_target


class SimpleGenerator(nn.Module):
    def __init__(self, config):
        super(SimpleGenerator, self).__init__()
        self.vocab_size = config.word_embeddings.shape[0]
        self.embedding_dim = config.word_embeddings.shape[1]
        self.num_relations = config.num_relations
        self.num_predicates = config.num_predicates
        # self.use_cuda = config.use_cuda

        self.encoder = SimpleEncoder(config)
        self.decoder = SimpleDecoder(config)

    def model(self, sub, obj, t, r=None, v=None):
        """
        This model corresponds to the following generative process:
        p(z) = Normal(0, I)         # latent "frame" variable that captures semantic information other than
                                    # "principle relation" r and "predicate" v
        p(r) = Categorical(r_prior)      # latent "principle relation" variables
        p(v) = Categorical(v_prior)      # latent predicate verb variable that the readers have to infer from the denominal verb
        p(x={sub, obj,t}|z,r,v) = Categorical(pi_x(z,r,v))     # components of the observed sentence
        pi_x is given by the "decoder" neural network
        :param sub: a batch of index for subjects in the observed sentences (e.g. "the boy" in "the boy porched the newspaper")
        :param obj: a batch of index for objects in the observed sentences (e.g. "the newspaper" in "the boy porched the newspaper")
        :param t: a batch of index for target words that evolves into denominal verbs in the observed sentences
               (e.g. "porch" in "the boy porched the newspaper")
        :param r: (optional) a batch of index for "principle relations" capturing ontological relation between the
               denominal verb ans its parent noun
               (e.g. "locatum_on" for "the boy porched the newspaper"
        :param v: (optional) a batch of index for "predicates" that the denominal verbs denote
               (e.g. "drop" for "the boy porched the newspaper)
        all params above should be torch.tensor of shape (batch_size,)
        :return: None
        """

        # register this module along with all its sub-modules within pyro
        pyro.module("generator", self)

        batch_size = t.size(0)
        # use plate to inform Pyro that the variables in the batch of xs = {subs, objs, ts}
        # are conditionally independent
        with pyro.plate("data"):
            # sample the semantic frame latent variable zs from constant prior distribution
            prior_loc = torch.zeros(batch_size, self.embedding_dim)
            prior_scale = diagonalize(torch.ones(batch_size, self.embedding_dim))
            z = pyro.sample('z', dist.MultivariateNormal(prior_loc, prior_scale).to_event(1))

            # if the predicate verb is observed (canonical sentences), score it against our prior
            # otherwise, sample z from the categorical prior pi_v
            v_prior = torch.ones(batch_size, self.num_predicates) / (1.0 * self.num_predicates)
            v = pyro.sample('v', dist.Categorical(probs=v_prior).to_event(1), obs=v)

            # similar to v above, sample the relation r for denominal verbs,
            # or score the observed r against prior pi_r for canonical sentences
            r_prior = torch.ones(batch_size, self.num_relations) / (1.0 * self.num_relations)
            r = pyro.sample('r', dist.Categorical(probs=r_prior).to_event(1), obs=r)

            # finally, score the observed sentence x = {sub, obj, t} using all latent variables above
            # against the parametrized distribution p(x|r,v,z) = Categorical(probs=decoder(r,v,z))
            # here 'decoder' is a deep neural network
            pi_sub, pi_obj, pi_t = self.decoder.forward(r, v, z)  # rs, vs are one-hot vectors
            pyro.sample('sub', dist.Categorical(pi_sub).to_event(1), obs=sub)
            pyro.sample('obj', dist.Categorical(pi_obj).to_event(1), obs=obj)
            pyro.sample('t', dist.Categorical(pi_t).to_event(1), obs=t)

    def guide(self, sub, obj, t, r=None, v=None):
        """
        The guide module corresponds to the following variational inference probability distributions:
        q(r|x) = categorical(loc=encoder_r(x))
        q(v|x,r) = categorical(loc=encoder_v(x,r))
        q(z|x, r, v) = normal((loc,scale)=encoder_z(x,r,v))
        parameters x = {sub, obj, t} are the same as those in model module
        """

        # use plate to inform Pyro that the variables in the batch of xs = {subs, objs, ts}
        # are conditionally independent
        with pyro.plate("data"):
            # if the predicates and relations are not observed (unsupervised), sample and score
            # them with variational distribution q(r,v|x) = q(r|x)q(v|x,r)
            prob_pred, prob_rel, z_loc, z_scale = self.encoder(sub, obj, t)
            if r is None:
                r = pyro.sample('r', dist.Categorical(probs=prob_rel).to_event(1))
            if v is None:
                # print("shape of r: ", r.shape)
                # print("shape of sub: ", sub.shape)
                v = pyro.sample('v', dist.Categorical(probs=prob_pred).to_event(1))

            # sample and score the latent frame variable z with
            # variational distribution q(z|x,r,v)
            pyro.sample('z', dist.MultivariateNormal(z_loc, z_scale).to_event(1))

    def identifier(self, sub, obj, t):
        """
        identify the underlying predicate and principle relation given contexts and target
        parameters x = {sub, obj, t} are the same as those in model module
        :return: a batch of inferred relations and predicates {r, v}
        """

        # use the trained model q(r,v|x) = q(r|x)q(v|x,r)
        # to compute probabilities for the latent variables
        # and then get the most probable relation and predicate from the posterior
        prob_pred, prob_rel, z_loc, z_scale = self.encoder(sub, obj, t)
        _, r_map = torch.topk(prob_rel, 1)
        r_map = torch.squeeze(r_map)
        _, v_map = torch.topk(prob_pred, 1)

        return r_map, v_map

    def model_identify(self, sub, obj, t, r=None, v=None):

        """
        this model is used to add an auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("generator", self)

        # use plate to inform Pyro that the variables in the batch of xs = {subs, objs, ts}
        # are conditionally independent
        with pyro.plate("data"):
            prob_pred, prob_rel, _, _ = self.encoder(sub, obj, t)
            # producing extra term to yield an auxiliary loss that we do gradient descent on
            # following Kingma et al.
            if r is not None:
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample("r_aux", dist.Categorical(prob_rel).to_event(1), obs=r)
            if v is not None:
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample("v_aux", dist.Categorical(prob_pred).to_event(1), obs=v)

    def guide_identify(self, sub, obj, t, r=None, v=None):
        """
        dummy guide function to accompany model_identify in inference
        """
        pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_hidden_dim', default=20, type=int)
    parser.add_argument('--decoder_hidden_dim', default=20, type=int)
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
    parser.add_argument('--batch_size', default=8, type=int,
                        help="number of instances to be considered in a batch")
    parser.add_argument('--experiment_name', default='exp_0', type=str)
    parser.add_argument('--experiment_type', default='test', type=str)
    parser.add_argument('--enumerate', action="store_true",
                        help="whether to use enumerate to reduce variance")
    return parser.parse_args()


def main():
    args = get_args()
    rel_embeddings = np.random.rand(7, 100)
    word_embeddings = np.random.rand(1000, 100)
    generator = training(args=args, rel_embeddings=rel_embeddings, word_embeddings=word_embeddings)


if __name__ == '__main__':
    main()
