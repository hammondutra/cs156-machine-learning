# Session 21 — Two-slide SetFit overview (text companion)

This Markdown companion preserves the **concepts and speaker explanations** used in the submitted two-slide, earth-tone PCW presentation. The original PDF upload remains in the class submission; this text file is readable and diffable in GitHub.

## Slide 1 — What is SetFit and how does it work?

**SetFit** stands for *Sentence Transformer Fine-Tuning*. It adapts an existing pretrained text encoder to a **new classification problem with relatively few labeled examples**.

1. Start with labeled sentences describing a user request, the agent's proposed action, and context.
2. Create **similar-label and different-label pairs**.
3. Fine-tune the encoder to give more similar representations to examples with the same label.
4. Train a simple classification head on the improved embeddings.

**Illustration:** If the user requests a repository summary, reading the README is authorized, whereas retrieving credentials is not authorized by that request.

**Speaker note:** Similarity alone does not establish permission. A production security tool also needs explicit policy enforcement.

## Slide 2 — Mathematics, comparison, and evaluation

Given text embeddings $z_i=f_\theta(x_i)$ and $z_j=f_\theta(x_j)$, cosine similarity is

$$
\operatorname{sim}(z_i,z_j)=\frac{z_i^\top z_j}{\|z_i\|_2\|z_j\|_2}.
$$

Training uses pairs to adapt the embedding space. The PCW's **NumPy illustration** simplifies this to a linear projection $z_i=x_iW$ and a squared cosine-label loss. This is *not* the full SetFit objective or implementation.

**Compared with logistic regression on fixed features:** SetFit adapts the learned input representation, rather than relying only on a fixed bag-of-words vector. This may help on small text-classification datasets, but whether it improves results must be tested.

**Evaluation for the planned agent-security task:** precision, recall, F1, confusion matrix, false-positive rate, and performance on **held-out scenarios**. Avoid evaluating on pairs that reuse training examples.

**Speaker note:** SetFit is supervised few-shot classification, *not* a complete novelty/anomaly detector. A One-Class SVM on frozen embeddings could be a useful alternative comparison.

## References

- https://huggingface.co/docs/setfit/index
- https://huggingface.co/docs/setfit/quickstart
- https://huggingface.co/datasets/SetFit/sst2
