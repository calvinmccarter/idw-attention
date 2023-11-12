# Inverse distance weighting attention

We report the effects of replacing the scaled dot-product (within softmax) attention with the negative-log of Euclidean distance. This form of attention simplifies to inverse distance weighting interpolation. Used in simple one hidden layer networks and trained with vanilla cross-entropy loss on classification problems, it tends to produce a key matrix containing prototypes and a value matrix with corresponding logits. We also show that the resulting interpretable networks can be augmented with manually-constructed prototypes to perform low-impact handling of special cases.

[Poster at Associative Memory & Hopfield Networks Workshop @ NeurIPS 2023](https://neurips.cc/virtual/2023/workshop/66524#wse-detail-78166)

[Paper on OpenReview](https://openreview.net/forum?id=dHmAhYu89E)

[Preprint on arXiv](https://arxiv.org/abs/2310.18805)
