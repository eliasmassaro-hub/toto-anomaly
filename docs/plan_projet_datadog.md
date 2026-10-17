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
- Les étapes 0 et 1 sont terminées (`src/check_toto.py`, `src/data.py`, `src/plot_data.py`). `src/check_toto.py` montre comment construire l'entrée `MaskedTimeseries` et appeler Toto : réutiliser ce code.

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

## Étape 2 : Détecteurs (`src/detectors.py`)

Interface commune : chaque détecteur a une méthode `predict_interval(history) -> (low, high)`, deux tableaux de longueur 64.

- **2.1** `BaselineDetector(window=200, k=3)` : médiane et MAD des `window` derniers points de l'historique, intervalle constant = médiane ± k × 1,4826 × MAD. Deux points d'attention :
  - **MAD nulle.** Sur une série presque plate (par exemple `ds-1525-5T`, qui reste à −0,5 avec des sauts réguliers), la MAD vaut 0 et l'intervalle a une largeur nulle. Si `1,4826 × MAD` est nul, utiliser à la place l'écart-type des `window` derniers points, avec un minimum de `1e-6` si la fenêtre est entièrement constante.
  - **Intervalle plat.** L'intervalle est le même pour les 64 points et ne suit pas les cycles de la série. Sur une série cyclique (par exemple `ds-277-T`), il doit couvrir tout le cycle et peut rater un pic posé dans un creux. Sur une série à sauts réguliers, il signale les sauts normaux comme anomalies. C'est une faiblesse attendue de la méthode simple : **ne pas la corriger**, c'est ce que la comparaison avec Toto doit mettre en évidence.
- **2.2** `find_anomalies(future, low, high)` : renvoie les positions des points de `future` hors de `[low, high]`.
- **2.3** `TotoDetector(num_samples=256)` : charge Toto une seule fois, prévoit 64 points à partir de l'historique, renvoie les quantiles 0,005 et 0,995. Ajouter une méthode `predict_interval_batch(histories)` qui traite plusieurs séries en un seul appel au modèle. La découper en paquets si la mémoire ne suffit pas.
- **2.4** Script de visualisation : une série avec les deux intervalles, les pics injectés, et les points détectés par chaque méthode.

## Étape 3 : Évaluation (`src/evaluate.py`)

- **3.1** Fonctions de métriques :
  - rappel = pics détectés / pics injectés ;
  - fausses alertes = points signalés qui ne sont pas des pics ;
  - temps de calcul moyen par série en ms (`time.perf_counter()`), après un appel d'échauffement.
- **3.2** Boucle d'évaluation sur les 200 séries pour trois configurations : méthode simple, Toto avec 32 échantillons, Toto avec 256 échantillons (sur Google Colab avec GPU si c'est trop lent en local). Même seed pour les pics dans les trois cas.
- **3.3** Enregistrer le tableau dans `results/eval.csv` et l'afficher :

| Détecteur | Rappel | Fausses alertes | ms / série |
|---|---|---|---|

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
- **5.3** Lancer le test avec `BATCH_SIZE=1` puis `BATCH_SIZE=16`, même nombre d'utilisateurs. Enregistrer dans `results/loadtest.csv` :

| Batch | Séries / s | Latence p95 (bout en bout) |
|---|---|---|

## Étape 6 : README

- **6.1** Rédiger `README.md` :
  - le problème en 3 lignes (incident → pic sur une métrique → il faut le repérer vite, sans fausses alertes) ;
  - le schéma d'architecture (client → API → file → worker → modèle, stockage des résultats) ;
  - comment lancer le projet ;
  - les deux tableaux de résultats et les figures ;
  - la conclusion : compromis qualité / coût entre Toto et la méthode simple, et l'idée de cascade (la méthode simple filtre toutes les séries, Toto ne traite que les suspectes) ;
  - les limites : incidents simulés sur de vraies séries, et les deux faiblesses de la méthode simple (intervalle plat, cas de la MAD nulle) ;
  - un paragraphe « À l'échelle de Datadog » : Kafka à la place de la file en mémoire, partitionnement des séries par identifiant, inférence sur GPU avec gros batchs, workers stateless répliqués horizontalement, stockage persistant (Redis ou base de données) à la place du dictionnaire.
