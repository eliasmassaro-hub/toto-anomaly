# Projet : détection d'incidents sur métriques avec Toto

## Contexte

Datadog surveille les métriques des systèmes de ses clients : taux d'erreur, latence, nombre de requêtes, CPU, mémoire. Quand un incident arrive, une métrique fait souvent un pic :

- un déploiement buggé fait exploser le taux d'erreur d'une API ;
- une requête lente en base de données fait monter la latence ;
- un afflux de trafic anormal (bot, attaque) fait exploser le nombre de requêtes.

Le but est de repérer ces pics en quelques minutes, avant que les clients ne s'en rendent compte, sans déclencher de fausses alertes (trop de fausses alertes et les ingénieurs d'astreinte finissent par les ignorer). Datadog propose pour cela des *Anomaly monitors*.

Ce projet teste Toto, le modèle de prévision open source de Datadog, pour ce use case. Toto prévoit un intervalle pour les prochains points d'une série, et un point réel qui sort de cet intervalle est une anomalie. Toto est comparé à une méthode simple, puis servi derrière une API asynchrone avec batching et testé en charge.

Les données sont de vraies métriques du dataset BOOM, sur lesquelles on simule des incidents sous forme de pics.

## Règles pour Claude Code

- Faire **une seule étape à la fois**, uniquement celle demandée. Certaines étapes seront codées par l'utilisateur lui-même : ne pas les implémenter sans demande explicite.
- C'est un projet d'apprentissage : code simple, lisible, brièvement commenté. Pas de sur-ingénierie, pas de dépendances inutiles.
- À la fin de chaque étape, expliquer en quelques phrases ce qui a été fait et comment le vérifier, puis faire un commit git avec un message du type `Étape 1.2 : ...`.
- Si un fichier existe déjà (écrit par l'utilisateur), le relire et l'utiliser tel quel au lieu de le réécrire.
- Les étapes 0 à 3.3 sont terminées (`src/check_toto.py`, `src/data.py`, `src/plot_data.py`, `src/detectors.py`, `src/plot_detectors.py`, `src/evaluate.py`). Réutiliser ces fichiers au lieu de les réécrire.

## Paramètres fixés

| Paramètre | Valeur |
|---|---|
| Nombre de séries | 200 (une seule variable par série) |
| Historique (contexte) | 1024 points |
| Horizon de prévision | 64 points (= 1 patch Toto) |
| Pics injectés | 3 par série, hauteur = 5 × écart-type de l'historique |
| Méthode simple | médiane et MAD des 200 derniers points, intervalle = médiane ± k × 1,4826 × MAD, k = 3 par défaut |
| Toto | modèle `Datadog/Toto-Open-Base-1.0`, quantiles 0,005 et 0,995, `num_samples` configurable |

## Structure du projet

```
toto-anomaly/
  src/check_toto.py      # étape 0 (fait)
  src/data.py            # chargement BOOM + injection de pics
  src/detectors.py       # méthode simple + Toto
  src/evaluate.py        # évaluation hors API
  src/api.py             # API FastAPI
  src/worker.py          # file d'attente + batching
  client.py              # client de démonstration
  loadtest/locustfile.py
  docs/api_spec.md
  results/               # tableaux et figures
  requirements.txt
  README.md
```

---

## Étape 0 : Mise en place (terminée)

Structure de dossiers, `.gitignore`, `requirements.txt`, environnement virtuel, `src/check_toto.py` (Toto se charge et prévoit 64 points en environ 440 ms sur CPU avec 32 échantillons), repo git relié à GitHub.

## Étape 1 : Données (`src/data.py`) (terminée)

- **1.1** Charger 200 séries du dataset Hugging Face `Datadog/BOOM` sans tout télécharger (2,8 Go). Pour cela, regarder d'abord la structure des fichiers du dataset et le code de chargement dans le repo GitHub `DataDog/toto` (dossier `boom`). Garder une seule variable par série, ne garder que les séries d'au moins 1024 + 64 points. Si un label de domaine est disponible, privilégier les domaines « application » et « infrastructure ». Sauvegarder en local (`data/series.npz` ou parquet).
- **1.2** Pour chaque série, découper en `history` (1024 points) et `future` (64 points suivants).
- **1.3** Écrire `inject_spikes(future, history, n_spikes=3, k=5, seed=None)` : ajoute `n_spikes` pics de hauteur `k × std(history)` à des positions aléatoires distinctes de `future`. Renvoie la série modifiée et les positions des pics.
- **1.4** Script de visualisation : afficher une série (historique + futur) avec les pics injectés marqués. Sauvegarder dans `results/`.

## Étape 2 : Détecteurs (`src/detectors.py`) (terminée)

Interface commune : chaque détecteur a une méthode `predict_interval(history) -> (low, high)`, deux tableaux de longueur 64.

- **2.1** `BaselineDetector(window=200, k=3)` : médiane et MAD des `window` derniers points de l'historique, intervalle constant = médiane ± k × 1,4826 × MAD. Deux points d'attention :
  - **MAD nulle.** Sur une série presque plate (par exemple `ds-1525-5T`, qui reste à −0,5 avec des sauts réguliers), la MAD vaut 0 et l'intervalle a une largeur nulle. Si `1,4826 × MAD` est nul, utiliser à la place l'écart-type des `window` derniers points, avec un minimum de `1e-6` si la fenêtre est entièrement constante.
  - **Intervalle plat.** L'intervalle est le même pour les 64 points et ne suit pas les cycles de la série. Sur une série cyclique (par exemple `ds-277-T`), il doit couvrir tout le cycle et peut rater un pic posé dans un creux. Sur une série à sauts réguliers, il signale les sauts normaux comme anomalies. C'est une faiblesse attendue de la méthode simple : **ne pas la corriger**, c'est ce que la comparaison avec Toto doit mettre en évidence.
- **2.2** `find_anomalies(future, low, high)` : renvoie les positions des points de `future` hors de `[low, high]`.
- **2.3** `TotoDetector(num_samples=256)` : charge Toto une seule fois, prévoit 64 points à partir de l'historique, renvoie les quantiles 0,005 et 0,995. Ajouter une méthode `predict_interval_batch(histories)` qui traite plusieurs séries en un seul appel au modèle. La découper en paquets si la mémoire ne suffit pas.
- **2.4** Script de visualisation : une série avec les deux intervalles, les pics injectés, et les points détectés par chaque méthode.

## Étape 3 : Évaluation (`src/evaluate.py`) (terminée)

- **3.1** Fonctions de métriques :
  - rappel = pics détectés / pics injectés ;
  - fausses alertes = points signalés qui ne sont pas des pics ;
  - temps de calcul moyen par série en ms (`time.perf_counter()`), après un appel d'échauffement.
- **3.2** Boucle d'évaluation sur les 200 séries pour trois configurations : méthode simple, Toto avec 32 échantillons, Toto avec 256 échantillons (sur Google Colab avec GPU si c'est trop lent en local). Même seed pour les pics dans les trois cas.
- **3.3** Enregistrer le tableau dans `results/eval.csv` et l'afficher.

Résultats obtenus sur 200 séries (600 pics) :

| Détecteur | Rappel | Précision | Fausses alertes / série | ms / série |
|---|---|---|---|---|
| Méthode simple (k = 3) | 97,7 % | 23,2 % | 9,7 | 0,03 |
| Toto, 32 tirages | 91,7 % | 29,6 % | 6,6 | 274 |
| Toto, 256 tirages | 82,3 % | 46,7 % | 2,8 | 2048 |

Ces configurations ne sont pas réglées au même niveau de fausses alertes, donc on ne peut pas encore dire quelle méthode est la meilleure. L'étape 3 bis rend la comparaison juste.

## Étape 3 bis : réglages et comparaison juste (`src/tuning.py`)

But : régler chaque méthode, puis les comparer à nombre de fausses alertes égal, sans surestimer les résultats. Réutiliser les fonctions de `evaluate.py` et `detectors.py`.

- **3.4** Métriques. Ajouter dans `evaluate.py`, en plus du rappel et des fausses alertes :
  - vrais positifs (TP) = pics injectés signalés ; faux positifs (FP) = points signalés qui ne sont pas des pics ;
  - **précision** = TP / (TP + FP), calculée sur les totaux de toutes les séries (pas une moyenne par série, pour éviter les divisions par zéro sur les séries sans alerte) ;
  - **F1** = 2 × précision × rappel / (précision + rappel) ;
  - **fausses alertes par série** = FP / nombre de séries.

  Ajouter ces colonnes à `results/eval.csv`.
- **3.5** Séparation validation / test. Couper les 200 séries en deux moitiés de 100, tirées au hasard avec une seed fixe :
  - la **validation** sert uniquement à choisir les réglages ;
  - le **test** sert uniquement à annoncer les résultats finaux.

  Les pics injectés restent les mêmes qu'à l'étape 3 (seed i pour la série i).
- **3.6** Méthode simple : balayage de `k` sur la validation, avec k = 2, 3, 4, 5, 6, 8, 10. Pour chaque k, calculer rappel, précision, F1 et fausses alertes par série. Enregistrer dans `results/tuning_baseline.csv`.
- **3.7** Toto : balayage du niveau de l'intervalle sur la validation, avec 256 tirages.

  | Niveau | Quantiles |
  |---|---|
  | 90 % | 0,05 et 0,95 |
  | 95 % | 0,025 et 0,975 |
  | 98 % | 0,01 et 0,99 |
  | 99 % | 0,005 et 0,995 (réglage actuel) |
  | 99,5 % | 0,0025 et 0,9975 |

  **Ne faire tourner Toto qu'une seule fois par série.** Ajouter à `TotoDetector` une méthode `predict_samples_batch(histories)` qui renvoie les tirages remis à l'échelle d'origine (même normalisation que `predict_interval_batch`), de forme `(n, 64, num_samples)`. Calculer ensuite tous les quantiles avec `np.quantile` sur ces mêmes tirages. Retenir le niveau qui maximise le F1. Enregistrer dans `results/tuning_toto_quantiles.csv`.
- **3.8** Toto : nombre de tirages sur la validation, au niveau retenu en 3.7, avec 32, 64, 128 et 256 tirages. Pour chaque valeur, relancer Toto pour avoir un vrai temps de calcul, et calculer rappel, précision, F1 et ms par série. Retenir le plus petit nombre de tirages dont le F1 est à moins de 0,01 du meilleur : c'est le meilleur compromis entre qualité et coût. Enregistrer dans `results/tuning_toto_samples.csv`.
- **3.9** Comparaison à fausses alertes égales. Sur la validation, choisir le `k` de la méthode simple dont le nombre de fausses alertes par série est le plus proche de celui de Toto réglé (3.7 et 3.8). Si aucun `k` du balayage n'en est proche, affiner la grille avec un pas de 0,5.
- **3.10** Résultats finaux, sur le **test** uniquement, pour trois configurations :
  - méthode simple avec k = 3 (réglage par défaut) ;
  - méthode simple avec le `k` de 3.9 ;
  - Toto réglé (niveau de 3.7, tirages de 3.8).

  Enregistrer dans `results/eval_test.csv` et afficher :

  | Détecteur | Rappel | Précision | F1 | Fausses alertes / série | ms / série |
  |---|---|---|---|---|---|

- **3.11** Figure `results/recall_vs_false_alarms.png`, sur la validation : rappel en fonction des fausses alertes par série, avec une courbe pour la méthode simple (un point par `k`) et une pour Toto (un point par niveau d'intervalle). Marquer le réglage retenu pour chaque méthode. À nombre de fausses alertes égal, la courbe la plus haute correspond à la meilleure méthode.

## Étape 4 : API (`src/api.py`, `src/worker.py`)

- **4.1** Rédiger `docs/api_spec.md`. La spec doit couvrir :
  - les endpoints, versionnés en `/v1` et nommés avec des noms plutôt que des verbes (par exemple `/v1/detection-jobs`) : création d'un job, lecture d'un job, `/health` ;
  - le JSON envoyé et reçu pour chaque endpoint ;
  - les codes de statut : `202` à la création, `200` en lecture, `400` si une série est invalide ou trop courte, `404` si le job n'existe pas, `429` si la file est pleine ;
  - les statuts d'un job : `pending`, `running`, `done`, `failed` ;
  - un format d'erreur unique, par exemple `{"error": "...", "message": "..."}`.
- **4.2** Implémenter l'API en suivant **exactement** `docs/api_spec.md` : FastAPI, modèles Pydantic, stockage des jobs dans un dictionnaire en mémoire, `asyncio.Queue` avec taille maximale. Charger Toto une seule fois, au démarrage.
- **4.3** Implémenter un worker sans batching : il prend un job dans la file, lance la détection, écrit le résultat dans le stockage. L'appel au modèle doit passer par `asyncio.to_thread` pour ne pas bloquer l'API pendant le calcul.
- **4.4** Script `client.py` : envoie une série, récupère le `job_id`, interroge le job jusqu'à `done`, affiche les anomalies.

## Étape 5 : Batching et test de charge

- **5.1** Ajouter le batching au worker : il prend jusqu'à `BATCH_SIZE` jobs, ou ce qui est arrivé au bout de `MAX_WAIT_MS`, et appelle `predict_interval_batch` une seule fois. Les deux paramètres sont lus depuis des variables d'environnement.
- **5.2** Écrire `loadtest/locustfile.py` : chaque utilisateur simulé crée un job puis interroge son statut jusqu'à `done`. Mesurer aussi la latence de bout en bout (de la création au statut `done`) comme métrique personnalisée Locust.
- **Attention au nombre de tirages.** Utiliser le nombre de tirages retenu en 3.8. `TotoDetector` ne traite que `max_samples_per_call // num_samples` séries par appel (2 séries avec 256 tirages). Pour qu'un batch de 16 jobs soit vraiment traité en un seul appel, il faut `16 × num_samples ≤ max_samples_per_call` : ajuster l'un ou l'autre et noter ce choix dans le README.
- **5.3** Lancer le test avec `BATCH_SIZE=1` puis `BATCH_SIZE=16`, même nombre d'utilisateurs. Enregistrer dans `results/loadtest.csv` :

| Batch | Séries / s | Latence p95 (bout en bout) |
|---|---|---|

## Étape 6 : README

- **6.1** Rédiger `README.md` :
  - le problème en 3 lignes (incident → pic sur une métrique → il faut le repérer vite, sans fausses alertes) ;
  - le schéma d'architecture (client → API → file → worker → modèle, stockage des résultats) ;
  - comment lancer le projet ;
  - les résultats : le tableau final sur le test (`results/eval_test.csv`), la figure rappel / fausses alertes, le tableau du test de charge et quelques figures de séries ;
  - la méthode d'évaluation : rappel, précision, F1, réglages choisis sur la validation et résultats annoncés sur le test ;
  - la conclusion : comparaison à fausses alertes égales, compromis qualité / coût entre Toto et la méthode simple, et l'idée de cascade (la méthode simple filtre toutes les séries, Toto ne traite que les suspectes) ;
  - les limites : incidents simulés sur de vraies séries, et les deux faiblesses de la méthode simple (intervalle plat, cas de la MAD nulle) ;
  - un paragraphe « À l'échelle de Datadog » : Kafka à la place de la file en mémoire, partitionnement des séries par identifiant, inférence sur GPU avec gros batchs, workers stateless répliqués horizontalement, stockage persistant (Redis ou base de données) à la place du dictionnaire.
