import pyro
import pyro.distributions as dist
import pyro.infer
import torch
from torch import nn

from train import diagonalize


class Encoder(nn.Module):
    """
    Encoder network for parametrized variational posterior q(v|x,r)
    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        # initialize parameters
        self.vocab_size = config.word_embeddings.shape[0]
        self.embedding_dim = config.word_embeddings.shape[1]
        self.hidden_dim = config.encoder_hidden_dim
        self.num_relations = config.num_relations
        self.num_predicates = self.vocab_size

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


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
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


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.vocab_size = config.word_embeddings.shape[0]
        self.embedding_dim = config.word_embeddings.shape[1]
        self.num_relations = config.num_relations
        self.num_predicates = self.vocab_size
        self.aux_loss_multiplier = config.aux_loss_multiplier

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

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
