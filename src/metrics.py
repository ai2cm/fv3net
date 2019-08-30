def r2_score(truth, pred, sample_dim, mean_dims=None):

    if mean_dims is None:
        mean_dims = [sample_dim]
    mean = truth.mean(mean_dims)

    sse = ((truth - pred) ** 2).sum(sample_dim)
    ss = ((truth - mean) ** 2).sum(sample_dim)

    return 1 - sse / ss
