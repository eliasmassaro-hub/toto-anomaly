# Spécification de l'API de détection d'anomalies

## Endpoints

| Méthode | Chemin | Rôle | Codes de réponse possibles |
|---|---|---|---|
| GET | /health | vérifier que le serveur tourne | 200 |
| POST | /v1/detection-jobs | créer un job de détection | 202, 400, 429 |
| GET | /v1/detection-jobs/{job_id} | lire le statut et le résultat d'un job | 200, 404 |

Tout endpoint peut aussi renvoyer `500` en cas d'erreur imprévue du serveur (voir le format d'erreur).

## GET /health

Réponse `200` :

```json
{"status": "ok"}
```

## POST /v1/detection-jobs

Corps de la requête :

```json
{
  "method": "toto",
  "series": [
    {"id": "api-paiement-erreurs",
     "history": [0.30, 0.31, ...],
     "recent": [0.29, 8.1, ...]}
  ]
}
```

Règles de validation (sinon `400`, et aucun job n'est créé) :

- `method` vaut `"toto"` ou `"baseline"` (`"toto"` par défaut) ;
- `series` contient entre 1 et 100 séries ;
- chaque série a un `id` (texte) ;
- `history` contient au moins 1024 nombres (seuls les 1024 derniers sont utilisés) ;
- `recent` contient entre 1 et 64 nombres ;
- `NaN` et `Infinity` sont refusés dans `history` et `recent`.

Réponse `202` :

```json
{"job_id": "a3f9c2...", "status": "pending"}
```

Réponse `400` (requête invalide, le message indique le premier champ en cause) :

```json
{"error": "invalid_request", "message": "body.series.0.history : List should have at least 1024 items after validation, not 10"}
```

Réponse `429` (file pleine, réessayer plus tard) :

```json
{"error": "queue_full", "message": "file d'attente pleine (100 jobs), réessayer plus tard"}
```

## GET /v1/detection-jobs/{job_id}

Réponse `200`, job en attente ou en cours (`"running"` une fois que le worker l'a pris) :

```json
{"job_id": "a3f9c2...", "status": "pending", "results": null, "error": null}
```

Réponse `200`, job terminé :

```json
{"job_id": "a3f9c2...", "status": "done", "error": null,
 "results": [{"id": "api-paiement-erreurs",
              "anomalies": [1],
              "low": [...], "high": [...]}]}
```

- `anomalies` : positions des points anormaux dans `recent` (0 = premier point) ;
- `low` et `high` : l'intervalle prévu, de même longueur que `recent`.

Réponse `200`, job échoué pendant le calcul (`error` suit le format d'erreur unique) :

```json
{"job_id": "a3f9c2...", "status": "failed", "results": null,
 "error": {"error": "internal_error", "message": "ValueError: ..."}}
```

Réponse `404` (job inconnu) :

```json
{"error": "not_found", "message": "job a3f9c2... introuvable"}
```

## Statuts d'un job

`pending` (dans la file) → `running` (en calcul) → `done` (terminé) ou `failed` (erreur pendant le calcul)

## Deux familles d'erreurs

- **Le client s'est trompé** (requête invalide) : réponse `400` au POST, aucun job n'est créé.
- **Le serveur a échoué pendant le calcul** : le job existe et passe à `failed`.

## Format d'erreur (toujours le même)

```json
{"error": "<code court>", "message": "<explication lisible>"}
```

Codes courts : `invalid_request` (400), `not_found` (404), `queue_full` (429), `internal_error` (500, ou dans le champ `error` d'un job `failed`).

Ce format sert aussi de champ `error` à un job `failed` : le client lit toutes les erreurs de la même façon.

## Limites connues

- **Jobs stockés en mémoire** : perdus si le serveur redémarre, et impossibles à partager entre plusieurs serveurs (il faudrait Redis ou une base de données).
- **Un seul worker** : les jobs sont traités un par un.
- **Pas de nettoyage** : les jobs terminés restent en mémoire indéfiniment (il faudrait une durée de vie, ou TTL).
- **Pas d'idempotence** : si le client renvoie la même requête (après une coupure réseau, par exemple), deux jobs sont créés.
- **Pas d'authentification** ni de limite par client (rate limiting).
- **Polling** : le client doit redemander le statut ; une alternative serait de le prévenir quand c'est fini (webhook).
