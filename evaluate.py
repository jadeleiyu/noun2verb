from pyro.infer.predictive import Predictive

from data_prep import prepare_data


def evaluate(generator, eval_data_loader, vocab, sample_size, batch_size):
    predict_fn = Predictive(generator.model, generator.guide, num_samples=sample_size, return_sites=('v', 'r', 'z'))

    num_batches = len(eval_data_loader) / batch_size
    eval_iter = iter(eval_data_loader)
    predict_df = {
        'subject': [],
        'object': [],
        'target': [],
        'true predicate': [],
        'true relation': [],
        'predicted predicates': [],
        'predicted relaitons': []
    }
    for i in range(num_batches):
        subs, objs, targets, relations, predicates = next(eval_iter)
        for j in range(batch_size):
            predict_df['subject'].append(subs[j])
            predict_df['object'].append(objs[j])
            predict_df['target'].append(targets[j])
            predict_df['true predicate'].append(predicates[j])
            predict_df['true relation'].append(relations[j])
        subs, objs, targets, relations, predicates = prepare_data(subs, objs, targets, relations, predicates, vocab)
        batch_pred_samples = predict_fn(subs, objs, targets, relations, predicates)['v'].view(batch_size, -1)
        batch_rel_samples = predict_fn(subs, objs, targets, relations, predicates)['r'].view(batch_size, -1)
        assert batch_pred_samples.shape[-1] == sample_size and batch_rel_samples.shape[-1] == sample_size
        for j in range(batch_size):
            # there're 'sample_size' sampled predicates and relations from the guide posterior
            sampled_predicates_idx = batch_pred_samples[j]
            sampled_rel_idx = batch_rel_samples[j]
            sampled_predicates = [vocab.i2w[pred_idx_tensor.item()] for pred_idx_tensor in sampled_predicates_idx]
            sampled_relations = [vocab.i2w[rel_idx_tensor.item()] for rel_idx_tensor in sampled_rel_idx]

            predict_df['predicted predicates'].append(sampled_predicates)
            predict_df['predicted relations'].append(sampled_relations)

    return predict_df





