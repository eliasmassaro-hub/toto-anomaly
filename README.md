# Détection d'incidents sur métriques avec Toto

*README en cours : il sera complété à l'étape 6. Pour l'instant, seule la partie sur le batching et le test de charge (étape 5) est rédigée.*

## Batching et test de charge

### Le choix du nombre de tirages

Le worker regroupe jusqu'à `BATCH_SIZE` jobs et les envoie à Toto en un seul appel. Toto duplique chaque série `num_samples` fois dans son batch interne. Pour qu'un batch de 16 jobs passe vraiment en un seul appel, il faut donc `16 × num_samples ≤ max_samples_per_call`.

Avec les 256 tirages retenus à l'étape 3.8, il faudrait 4096 lignes par appel : sur CPU (16 Go), la machine swappe et l'appel devient environ 10 fois plus lent. Avec `max_samples_per_call = 512`, un batch de 16 séries est alors découpé en 8 appels de 2 séries, et le batching ne sert plus à rien. Mesure sur 16 séries, en régime établi :

| Tirages | Une série par appel | 16 séries d'un coup | Appels pour 16 séries |
|---|---|---|---|
| 32 | 440 ms / série | 318 ms / série | 1 |
| 256 | 2865 ms / série | 2956 ms / série | 8 |

**Choix : le test de charge utilise `TOTO_SAMPLES=32`.** On a alors 16 × 32 = 512 tirages, soit un seul appel par batch. Le prix à payer est la qualité : sur la validation, le F1 passe de 0,61 (256 tirages) à 0,48 (32 tirages), d'après `results/tuning_toto_samples.csv`. Par défaut, l'API garde 256 tirages. Sur GPU, on garderait 256 tirages et on monterait `max_samples_per_call` à 4096.

### Lancer le test

Terminal 1, le serveur (refaire avec `BATCH_SIZE=16`) :

```
BATCH_SIZE=1 TOTO_SAMPLES=32 uvicorn api:app --app-dir src --no-access-log
```

Terminal 2, Locust avec 32 utilisateurs pendant 3 minutes :

```
locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 32 -r 32 -t 3m --csv loadtest/out/batch1
```

Une fois les deux tests passés, `python loadtest/summarize.py` écrit `results/loadtest.csv`.

### Résultats

Toto (32 tirages, niveau 99 %) sur CPU (MacBook Air, 16 Go). Chaque job contient une série, et chaque utilisateur enchaîne les jobs sans pause.

| Batch | Séries / s | Latence p50 (bout en bout) | Latence p95 (bout en bout) | Jobs terminés |
|---|---|---|---|---|
| 1 | 2,67 | 12 s | 14 s | 478 |
| 16 | 3,04 | 10 s | 12 s | 544 |

- **Débit : +14 %.** Avec `BATCH_SIZE=16`, les 35 batchs étaient tous pleins. Le gain reste modeste, car sur CPU le modèle est déjà presque saturé par les 32 tirages de chaque série. Sur GPU, un batch plus gros occuperait des cœurs inutilisés, et le gain serait bien plus grand.
- **Latence : −2 s.** Ici, la latence est surtout de l'attente dans la file. Avec 32 utilisateurs, la loi de Little donne latence ≈ utilisateurs / débit, soit 32 / 2,67 ≈ 12 s et 32 / 3,04 ≈ 10,5 s, ce qui correspond aux mesures. Le batching réduit la latence parce qu'il augmente le débit. À faible charge, ce serait l'inverse : un job attendrait jusqu'à `MAX_WAIT_MS`, puis la fin de tout son batch.
- **Aucun échec**, aucun `429` : la file (100 jobs) n'a jamais été pleine avec 32 utilisateurs.
