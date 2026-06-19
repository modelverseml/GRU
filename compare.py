"""Compare already-trained GRU implementations side by side.

Training is done by the caller (the notebook), which keeps the slow framework
training out of this function and lets you compare whatever set of models you
have on hand — all three (manual / TensorFlow / PyTorch) or just the manual one.

`compare_models` takes a ``{name: trained_model}`` mapping plus the train/test
data and generation settings, and prints train/test accuracy (or MSE) and a
sample generation for each model. Every model only needs a ``predict`` method
and ``n_x``/``n_y``/``task`` attributes — which the manual GRU and both
framework wrappers all provide.
"""

from utils import evaluate, generate


def compare_models(models, X_train, Y_train, X_test, Y_test, *,
                   embedding, decoder, seed_word,
                   num_gen=10, sample=False):
    """Score and generate from each model in ``models`` ({name: trained_model}).

    Returns the same ``models`` mapping for convenience.
    """
    # task is shared across the models being compared; read it off the first.
    task = next(iter(models.values())).task

    # For classification evaluate() returns accuracy (higher is better);
    # for regression it returns MSE (lower is better).
    metric = "accuracy" if task == "classification" else "MSE"
    print(f"=== {metric}: train / test ===")
    print(f"  {'model':16} {'train':>8} {'test':>8}")
    for name, mdl in models.items():
        train_score = evaluate(mdl, X_train, Y_train)
        test_score = evaluate(mdl, X_test, Y_test)
        print(f"  {name:16} {train_score:8.4f} {test_score:8.4f}")

    print(f"\n=== generation from '{seed_word}' ===")
    for name, mdl in models.items():
        text = generate(mdl, embedding, decoder, seed_word,
                        num_words=num_gen, sample=sample)
        print(f"  {name:16} {text}")

