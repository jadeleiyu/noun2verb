import torch
from tqdm import tqdm
from data_prep import prepare_data

def diagonalize(tensor):
    vec_dim = tensor.shape[-1]
    vectors = tensor.view(-1, vec_dim)
    diag_mats = []
    for vector in vectors:
        diag_mats.append(torch.diag(vector))
    return torch.stack(diag_mats)


def train_epoch(data_loaders, models, periodic_interval_batches, vocab):
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
        # def prepare_data(batched_subs, batched_objs, batched_targets, batched_relations, batched_predicates, vocab):
        subs, objs, targets, relations, predicates = prepare_data(subs, objs, targets, relations, predicates, vocab)
        # subs = torch.tensor(subs)
        # objs = torch.tensor(objs)
        # targets = torch.tensor(targets)
        # relations = torch.tensor(relations)
        # predicates = torch.tensor(predicates)

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
