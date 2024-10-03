import numpy as np
from scipy.spatial.distance import cosine


def pick_subjects(embeddings):

    n_subjects = len(embeddings)

    all_embeddings = []
    for sub in range(n_subjects):
        all_embeddings.append(embeddings[sub].measurements)

    all_embeddings = np.asarray(all_embeddings)
    avg_embeddings = all_embeddings.mean(axis=0)

    n_embeddings, n_dims = avg_embeddings.shape

    subjects_picked = []
    for emb in range(n_embeddings):
        this_avg = avg_embeddings[emb]
        sub_embeds = all_embeddings[:, emb, :].squeeze()

        coss = []
        for embeds in sub_embeds:
            coss.append(cosine(this_avg, embeds))

        subjects_picked.append(np.argmin(coss))

    return subjects_picked
