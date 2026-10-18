"""Étape 3 bis : réglage des détecteurs et comparaison à fausses alertes égales.

Usage : python src/tuning.py
Les réglages sont choisis sur la validation ; les résultats finaux sont annoncés sur le test.
"""

import numpy as np

from evaluate import make_eval_set

SPLIT_SEED = 0  # seed du tirage validation / test


# --- 3.5 Séparation validation / test ---

def split_val_test(n_series, seed=SPLIT_SEED):
    """Coupe les indices 0..n_series-1 en deux moitiés tirées au hasard : (validation, test).

    La validation sert uniquement à choisir les réglages, le test uniquement à annoncer
    les résultats finaux.
    """
    order = np.random.default_rng(seed).permutation(n_series)
    half = n_series // 2
    return np.sort(order[:half]), np.sort(order[half:])


def main():
    # Pics de l'étape 3 (seed i pour la série i), puis découpage par indices
    history, spiked, positions = make_eval_set()
    val, test = split_val_test(len(history))
    print(f"Validation : {len(val)} séries | test : {len(test)} séries")


if __name__ == "__main__":
    main()
