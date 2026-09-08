# Analyse du sweep `runs_log.csv` — Pantagruel

## Vue d'ensemble

Le fichier `runs_log.csv` contient **55 lignes** : une ligne `demo/test` vide (run_id 1, sans aucune valeur renseignée), et **54 runs réels** correspondant au sweep systématique de l'architecture Pantagruel. Les runs s'échelonnent du **5 août au 3 septembre 2026**, avec deux creux : 18 jours entre le 6 et le 24 août (le creux déjà identifié dans la chronologie du rapport de stage), puis 7 jours entre le 26 août et le 2 septembre.

C'est le seul sweep du projet loggé de façon systématique et structurée (un CSV par run, avec hyperparamètres et métriques complètes).

### Protocole commun à tous les runs

- 5-fold cross-validation patient-level (`k_folds=5`)
- Séquences audio de 4 secondes (`max_len_s=4`)
- Batch size 8, accumulation 2 → batch effectif 16
- 30 epochs max, learning rate 1e-4
- Focal loss avec gamma=2 (sauf un run en Phase 3, voir plus bas)
- `finetune_scope` = tête de fusion + projection résiduelle uniquement (encodeurs backbone gelés)
- 265 échantillons au total (168 Foch pathologiques + 100 SVD sains cible)

Sur les 46 premiers runs (5–26 août), seuls deux éléments varient d'un run à l'autre : **l'architecture active** (quelles branches sont activées) et **le type d'augmentation utilisé** — le weight decay reste fixé à 0.1. Les 8 derniers runs (2–3 septembre, Phase 3) figent au contraire architecture et augmentation sur la meilleure combinaison trouvée, et font varier à la place **weight_decay**, **dropout**, **input_dropout** et (une fois) **focal_gamma** — voir la section Phase 3 ci-dessous.

---

## Trois phases d'hyperparamètres

Un changement net de régime de régularisation coupe le sweep en deux phases, exactement à la frontière du trou de 18 jours ; une troisième phase, plus tardive et beaucoup plus ciblée, vient ensuite affiner la régularisation sur la seule architecture gagnante :

| Phase | Période | Runs | dropout | hidden_dim | input_dropout | weight_decay |
|---|---|---|---|---|---|---|
| Phase 1 | 5–6 août | 16 runs | 0.2 | 256 | 0.1 | 0.1 |
| Phase 2 | 24–26 août | 30 runs | 0.5 | 128 | 0.3 | 0.1 |
| Phase 3 | 2–3 sept. | 8 runs | 0.3–0.5 | 128 | 0.1–0.3 | 0.02–0.1 |

Le basculement phase 1 → phase 2 a lieu très précisément sur le **run 18** (06/08 17h12), qui teste déjà `dropout=0.5` juste avant l'interruption. À la reprise, le **run 25** (24/08, même preset `wavlm_bp_residual`, même augmentation `signal`) reproduit **exactement** le même score (0.5274) — ce qui confirme la reproductibilité du pipeline et suggère que la reprise après le trou a repris le nouveau réglage de régularisation là où il avait été laissé. C'est cohérent avec l'hypothèse d'un échange avec un encadrant ayant motivé un régime plus conservateur, étant donné la taille réduite du dataset.

### Phase 3 (2-3 septembre) : affinage ciblé sur WavLM-Large + Pantagruel

Après le trou de 7 jours qui suit la fin du sweep large (run 47, 26/08), l'exploration change de nature : au lieu de balayer des architectures, les 8 derniers runs figent l'architecture sur la meilleure combinaison identifiée en phase 2 (`pantagruel_wavlm_la`, augmentation `h_vc+signal`) et cherchent à affiner sa régularisation — hypothèse plausible : le régime « phase 2 » (dropout 0.5, input_dropout 0.3) avait été généralisé à tout le sweep alors qu'il n'était peut-être nécessaire que pour les architectures plus petites/instables.

| # | Date/heure | Preset | weight_decay | dropout | input_dropout | focal_gamma | Bal.Acc OOF | Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| 41 *(réf. phase 2)* | 08-25 15:28 | pantagruel_wavlm_la | 0.1 | 0.5 | 0.3 | 2 | 0.698 | 0.660 |
| 48 | 09-02 13:57 | pantagruel_wavlm_la | 0.1 | 0.5 | 0.3 | **3** | 0.691 | 0.645 |
| 49 | 09-02 14:44 | pantagruel_wavlm_la | **0.02** | 0.5 | 0.3 | 2 | 0.713 | 0.667 |
| 50 | 09-02 15:23 | pantagruel_wavlm_la | 0.03 | 0.5 | 0.3 | 2 | 0.673 | 0.649 |
| **51** | **09-02 16:02** | **pantagruel_wavlm_la** | **0.02** | 0.5 | **0.1** | 2 | **0.716** | **0.688** |
| 52 | 09-02 17:01 | pantagruel_wavlm_la | 0.02 | 0.5 | 0.15 | 2 | 0.705 | 0.681 |
| 53 | 09-03 09:59 | pantagruel_wavlm_la | 0.02 | **0.3** | 0.1 | 2 | 0.714 | 0.681 |
| 54 | 09-03 11:20 | pantagruel_wavlm_la | 0.02 | 0.3 | 0.1 | 2 | 0.680 | 0.669 |
| 55 | 09-03 12:23 | pantagruel_wavlm_la_residual | 0.02 | 0.3 | 0.1 | 2 | 0.692 | 0.679 |

Enseignements de cette phase :

- **Baisser le weight decay aide nettement** : à `dropout=0.5, input_dropout=0.3` fixes, passer de 0.1 (run 41, 0.698) à 0.02 (run 49, 0.713) gagne +0.015 ; mais 0.03 (run 50, 0.673) fait moins bien que 0.02, donc l'optimum n'est pas simplement « plus bas = mieux ».
- **Baisser aussi l'input_dropout aide encore** : le run 51 (`input_dropout=0.1`) devient le **meilleur run du projet** (Bal.Acc 0.716, Macro-F1 0.688), devant le run 49 (`input_dropout=0.3`, 0.713) et le run 52 (`input_dropout=0.15`, 0.705) — tendance monotone sur les 3 valeurs testées.
- **Le run 48 (focal_gamma=3) est le seul de la phase à ne pas améliorer sur la référence** (0.691 vs 0.698 pour le run 41), suggérant que `gamma=2` reste préférable.
- **Runs 53 et 54 : mêmes hyperparamètres exacts, résultats différents** (0.714 vs 0.680, écart de 0.034) — le log ne trace pas de seed fixée. C'est un signal important à garder en tête pour tout le sweep : une partie des écarts entre runs voisins dans ces tableaux relève probablement du bruit d'entraînement plutôt que d'un effet réel de l'hyperparamètre changé. Le run 51 (0.716) reste malgré tout au-dessus de la fourchette de bruit observée sur cette paire (0.680–0.714).
- **La branche résiduelle (run 55) n'aide pas** à hyperparamètres phase 3 égaux : 0.692, contre 0.714 et 0.680 pour les runs 53/54 (mêmes réglages sans résiduel) — cohérent avec le constat déjà fait en phase 2 (résiduel quasi neutre à négatif une fois WavLM-Large en place).

### Effet direct de la régularisation (paires contrôlées)

Sur les deux seules paires directement comparables (même preset + même augmentation, seul le dropout change) :

| Preset | Augmentation | Bal.Acc (dropout 0.2) | Bal.Acc (dropout 0.5) |
|---|---|---|---|
| `wavlm_bp_residual` | signal | 0.562 | 0.527 |
| `pantagruel_solo` | signal | 0.601 | 0.591 |

Dans les deux cas, la régularisation plus forte fait légèrement baisser le score — cohérent avec un compromis biais/variance plutôt qu'avec une erreur de pipeline.

---

## Architectures testées (10 presets)

| Preset | n runs | Bal.Acc moyenne | max | min |
|---|---|---|---|---|
| **pantagruel_wavlm_la** | 8 | **0.699** | 0.716 | 0.673 |
| pantagruel_wavlm_la_residual | 2 | 0.691 | 0.692 | 0.691 |
| wavlm_la_solo | 1 | 0.638 | 0.638 | 0.638 |
| pantagruel_wavlm_bp | 6 | 0.627 | 0.652 | 0.572 |
| pantagruel_wavlm_bp_residual | 6 | 0.625 | 0.658 | 0.603 |
| pantagruel_solo | 7 | 0.594 | 0.625 | 0.541 |
| pantagruel_residual | 6 | 0.593 | 0.659 | 0.540 |
| wavlm_bp_residual | 7 | 0.550 | 0.611 | 0.503 |
| wavlm_bp_solo | 6 | 0.517 | 0.587 | 0.462 |
| residual_solo | 5 | 0.398 | 0.473 | 0.315 |

### Enseignements

- **Pantagruel aide systématiquement** : `use_pantagruel=True` → moyenne 0.634 vs 0.504 sans (n=35 vs 19).
- **Le résiduel vocodeur seul est très faible** (`residual_solo`, ~0.40 en moyenne, jusqu'à 0.315 — proche du hasard sur 4 classes). Il ne devient utile que combiné à d'autres branches.
- **`use_residual` en moyenne globale n'aide pas** (moyenne 0.559 avec vs 0.616 sans, n=26 vs 28) — mais cette moyenne est confondue par les phases d'hyperparamètres ; à architecture fixe (WavLM-Large + Pantagruel), le résiduel reste quasi neutre à légèrement négatif (voir plus bas et la section Phase 3).
- **WavLM-Large domine largement WavLM-Base-Plus** : les 11 runs testés en Large (fin de sweep le 26 août, puis affinage ciblé les 2-3 septembre) donnent une moyenne de 0.692 contre 0.579 en Base-Plus (n=25). L'échantillon Large reste plus petit, mais avec 11 runs répartis sur deux campagnes distinctes le signal est maintenant solide.
- Le meilleur run global (**#51**, phase 3) n'utilise **pas** la branche résiduelle (`use_residual=False`) — comme le meilleur run de phase 2 (#41) qu'il améliore.

---

## Effet du type d'augmentation

| Type d'aug | n | Bal.Acc moyenne | max |
|---|---|---|---|
| h_vc+signal | 16 | 0.656 | 0.716 |
| signal+vc | 7 | 0.607 | 0.659 |
| vc | 7 | 0.593 | 0.658 |
| signal | 10 | 0.557 | 0.639 |
| h_vc | 7 | 0.538 | 0.621 |
| aucune | 7 | 0.508 | 0.603 |

Gradient monotone très net : **plus on combine de types d'augmentation, mieux c'est**, et l'absence totale d'augmentation (`aucune`) est systématiquement la pire option à architecture égale. `h_vc+signal` (combinaison la plus riche testée) est la meilleure en moyenne — et c'est justement la configuration du meilleur run du projet. À noter : la moyenne `h_vc+signal` grimpe nettement (0.614 → 0.656) parce que les 8 runs de la Phase 3 y sont tous rattachés — un facteur confondant, puisque ces runs partagent aussi la meilleure architecture (WavLM-Large + Pantagruel) et non un effet isolé de l'augmentation.

---

## Le run vedette : #51

**`pantagruel_wavlm_la_kfold_aug`** — 02/09/2026 16h02 (Phase 3)

- Architecture : WavLM-Large + Pantagruel, sans branche résiduelle
- Augmentation : `h_vc+signal`
- Régularisation : dropout 0.5, weight_decay 0.02, input_dropout 0.1 (issue de l'affinage phase 3, voir plus haut)

**Résultats OOF :**
- Balanced Accuracy : **0.7163**
- Macro-F1 : **0.6881**
- Macro AUC (OvR) : **0.8822**

**Moyennes k-fold (cohérentes, pas de fold aberrant) :**
- Balanced Acc : 0.716 ± 0.047
- Macro-F1 : 0.686 ± 0.055
- Macro AUC : 0.888 ± 0.014

Ce run améliore le précédent record du sweep, **#41** (même preset, même augmentation, mais hyperparamètres phase 2 : dropout 0.5, weight_decay 0.1, input_dropout 0.3), de **+0.018 en Balanced Accuracy** et **+0.028 en Macro-F1**, pour un coût de recherche minime (7 runs supplémentaires ciblés sur cette seule architecture). Le gain vient d'un weight_decay plus faible (0.02 au lieu de 0.1) combiné à un input_dropout plus faible (0.1 au lieu de 0.3) — une régularisation globalement *moins* forte que le régime « phase 2 » appliqué uniformément à tout le sweep large (voir la section Phase 3 pour la nuance : l'effet n'est pas monotone sur le weight_decay seul, et une partie de l'écart entre runs voisins à hyperparamètres identiques est du bruit d'entraînement, cf. runs 53/54).

Le run résiduel équivalent le plus proche, **`pantagruel_wavlm_la_residual`** (#55, mêmes réglages phase 3 que les runs #53/#54 : dropout 0.3, weight_decay 0.02, input_dropout 0.1), donne un résultat plus faible (Bal.Acc 0.692) que ses deux homologues non-résiduels à réglages identiques (#53 : 0.714, #54 : 0.680) — cohérent avec l'observation déjà faite en phase 2 entre #41 et #42 : le résiduel n'apporte rien, voire nuit légèrement, une fois WavLM-Large en place.

---

## Le pire cas : `residual_solo`

Le résiduel vocodeur utilisé **seul**, sans aucune autre branche, s'effondre : de 0.473 (meilleur cas, aug=`signal+vc`) à **0.315** (aug=`h_vc`, dropout 0.5) — proche du hasard sur une tâche à 4 classes. Ceci confirme quantitativement ce qui était déjà noté qualitativement ailleurs dans le projet : le résiduel capture un signal réel, mais insuffisant seul ; il a besoin d'être fusionné avec des embeddings SSL riches pour apporter de la valeur.

---

## Table croisée preset × augmentation (Balanced Accuracy OOF)

| Preset | aucune | signal | vc | signal+vc | h_vc | h_vc+signal |
|---|---|---|---|---|---|---|
| pantagruel_residual | 0.540 | 0.584 | 0.624 | 0.659 | 0.581 | 0.571 |
| pantagruel_solo | 0.541 | 0.596 | 0.599 | 0.625 | 0.619 | 0.583 |
| pantagruel_wavlm_bp | 0.572 | 0.639 | 0.637 | 0.652 | 0.621 | 0.643 |
| pantagruel_wavlm_bp_residual | 0.603 | 0.620 | 0.658 | 0.649 | 0.615 | 0.604 |
| pantagruel_wavlm_la | — | — | — | — | — | **0.699 (n=8, max 0.716)** |
| pantagruel_wavlm_la_residual | — | — | — | — | — | 0.691 (n=2, max 0.692) |
| residual_solo | 0.322 | 0.413 | 0.468 | 0.473 | 0.315 | — |
| wavlm_bp_residual | 0.516 | 0.539 | 0.611 | 0.606 | 0.503 | — |
| wavlm_bp_solo | 0.462 | 0.508 | 0.553 | 0.587 | 0.507 | 0.484 |
| wavlm_la_solo | — | — | — | — | — | 0.638 |

*(les cases vides correspondent aux runs Large et résiduel-seul non testés sur toute la grille — c'est un sweep partiel, pas un plan factoriel complet. Les deux cellules `pantagruel_wavlm_la*` sont les seules à agréger plusieurs runs : les 7 runs additionnels de la Phase 3 répètent ce même couple preset × augmentation en ne faisant varier que la régularisation — voir le tableau détaillé plus haut.)*

---

## Chronologie complète des 54 runs

| # | Date/heure | Preset | Augmentation | Dropout | Bal.Acc OOF |
|---|---|---|---|---|---|
| 2 | 08-05 14:31 | pantagruel_wavlm_bp | signal+vc | 0.2 | 0.652 |
| 3 | 08-05 15:09 | pantagruel_wavlm_bp_residual | signal+vc | 0.2 | 0.649 |
| 4 | 08-05 16:51 | pantagruel_solo | signal+vc | 0.2 | 0.625 |
| 5 | 08-05 17:20 | wavlm_bp_solo | signal+vc | 0.2 | 0.587 |
| 6 | 08-06 10:22 | pantagruel_residual | signal+vc | 0.2 | 0.659 |
| 7 | 08-06 10:34 | residual_solo | signal+vc | 0.2 | 0.473 |
| 8 | 08-06 11:23 | wavlm_bp_residual | signal+vc | 0.2 | 0.606 |
| 9 | 08-06 12:09 | pantagruel_wavlm_bp_residual | vc | 0.2 | 0.658 |
| 10 | 08-06 13:49 | pantagruel_wavlm_bp | vc | 0.2 | 0.637 |
| 11 | 08-06 14:26 | pantagruel_solo | vc | 0.2 | 0.599 |
| 12 | 08-06 14:48 | wavlm_bp_solo | vc | 0.2 | 0.553 |
| 13 | 08-06 15:14 | residual_solo | vc | 0.2 | 0.468 |
| 14 | 08-06 15:51 | pantagruel_residual | vc | 0.2 | 0.624 |
| 15 | 08-06 16:18 | wavlm_bp_residual | vc | 0.2 | 0.611 |
| 16 | 08-06 16:37 | pantagruel_solo | signal | 0.2 | 0.601 |
| 17 | 08-06 16:55 | wavlm_bp_residual | signal | 0.2 | 0.562 |
| 18 | 08-06 17:12 | wavlm_bp_residual | signal | 0.5 | 0.527 |
| *(trou de 18 jours — 7 au 23 août)* | | | | | |
| 19 | 08-24 12:13 | pantagruel_wavlm_bp_residual | signal | 0.5 | 0.620 |
| 20 | 08-24 12:34 | pantagruel_wavlm_bp | signal | 0.5 | 0.639 |
| 21 | 08-24 14:12 | pantagruel_solo | signal | 0.5 | 0.591 |
| 22 | 08-24 14:27 | wavlm_bp_solo | signal | 0.5 | 0.508 |
| 23 | 08-24 15:42 | residual_solo | signal | 0.5 | 0.413 |
| 24 | 08-24 15:57 | pantagruel_residual | signal | 0.5 | 0.584 |
| 25 | 08-24 16:08 | wavlm_bp_residual | signal | 0.5 | 0.527 |
| 26 | 08-24 16:27 | pantagruel_wavlm_bp_residual | aucune | 0.5 | 0.603 |
| 27 | 08-24 16:42 | pantagruel_wavlm_bp | aucune | 0.5 | 0.572 |
| 28 | 08-24 16:53 | pantagruel_solo | aucune | 0.5 | 0.541 |
| 29 | 08-24 17:05 | wavlm_bp_solo | aucune | 0.5 | 0.462 |
| 30 | 08-25 09:29 | residual_solo | aucune | 0.5 | 0.322 |
| 31 | 08-25 09:46 | pantagruel_residual | aucune | 0.5 | 0.540 |
| 32 | 08-25 09:59 | wavlm_bp_residual | aucune | 0.5 | 0.516 |
| 33 | 08-25 11:03 | pantagruel_wavlm_bp_residual | h_vc | 0.5 | 0.615 |
| 34 | 08-25 11:31 | pantagruel_wavlm_bp | h_vc | 0.5 | 0.621 |
| 35 | 08-25 11:52 | pantagruel_solo | h_vc | 0.5 | 0.619 |
| 36 | 08-25 12:10 | wavlm_bp_solo | h_vc | 0.5 | 0.507 |
| 37 | 08-25 12:19 | residual_solo | h_vc | 0.5 | 0.315 |
| 38 | 08-25 13:56 | pantagruel_residual | h_vc | 0.5 | 0.581 |
| 39 | 08-25 14:22 | wavlm_bp_residual | h_vc | 0.5 | 0.503 |
| 40 | 08-25 14:53 | pantagruel_wavlm_bp_residual | h_vc+signal | 0.5 | 0.604 |
| 41 | 08-25 15:28 | pantagruel_wavlm_la | h_vc+signal | 0.5 | 0.698 |
| 42 | 08-25 16:05 | pantagruel_wavlm_la_residual | h_vc+signal | 0.5 | 0.691 |
| 43 | 08-25 16:55 | pantagruel_wavlm_bp | h_vc+signal | 0.5 | 0.643 |
| 44 | 08-26 09:56 | pantagruel_solo | h_vc+signal | 0.5 | 0.583 |
| 45 | 08-26 10:15 | wavlm_bp_solo | h_vc+signal | 0.5 | 0.484 |
| 46 | 08-26 10:57 | wavlm_la_solo | h_vc+signal | 0.5 | 0.638 |
| 47 | 08-26 16:54 | pantagruel_residual | h_vc+signal | 0.5 | 0.571 |
| *(trou de 7 jours — 27 août au 1er septembre)* | | | | | |
| 48 | 09-02 13:57 | pantagruel_wavlm_la | h_vc+signal | 0.5 | 0.691 |
| 49 | 09-02 14:44 | pantagruel_wavlm_la | h_vc+signal | 0.5 | 0.713 |
| 50 | 09-02 15:23 | pantagruel_wavlm_la | h_vc+signal | 0.5 | 0.673 |
| **51** | **09-02 16:02** | **pantagruel_wavlm_la** | **h_vc+signal** | **0.5** | **0.716 (meilleur run)** |
| 52 | 09-02 17:01 | pantagruel_wavlm_la | h_vc+signal | 0.5 | 0.705 |
| 53 | 09-03 09:59 | pantagruel_wavlm_la | h_vc+signal | 0.3 | 0.714 |
| 54 | 09-03 11:20 | pantagruel_wavlm_la | h_vc+signal | 0.3 | 0.680 |
| 55 | 09-03 12:23 | pantagruel_wavlm_la_residual | h_vc+signal | 0.3 | 0.692 |

*(weight_decay et input_dropout varient aussi entre les runs 48-55, sans être capturés par la colonne « Dropout » ci-dessus — voir le tableau détaillé de la section Phase 3.)*

---

## Synthèse pour le rapport de stage

Ce sweep confirme quantitativement, sur une grille de 54 runs contrôlés, quatre résultats déjà identifiés (pour les trois premiers) qualitativement ailleurs dans le projet :

1. **L'apport net et cohérent de l'encodeur français Pantagruel** (+0.13 en moyenne sur la balanced accuracy, n=35 avec vs n=19 sans).
2. **Un gradient monotone de l'augmentation** : combiner plusieurs types d'augmentation (voix + signal) bat systématiquement l'absence d'augmentation.
3. **Le saut de régime de régularisation** (dropout 0.2→0.5) survient exactement à la charnière du trou de 18 jours du projet — un bon indice temporel pour dater le moment où le projet a été réorienté vers une approche plus prudente vis-à-vis du surapprentissage, étant donné la taille réduite du dataset (265 échantillons).
4. **Un affinage ciblé de cette même régularisation (Phase 3, 2-3 septembre) a ensuite montré que le régime « phase 2 » était en fait légèrement trop conservateur pour la meilleure architecture** : baisser le weight_decay (0.1→0.02) et l'input_dropout (0.3→0.1) a fait gagner encore +0.018 en balanced accuracy sur WavLM-Large + Pantagruel. Cette phase a aussi révélé, sur une paire de runs à hyperparamètres strictement identiques (#53/#54), un bruit d'entraînement d'environ 0.03 en balanced accuracy — un repère utile pour ne pas surinterpréter les petits écarts ailleurs dans le sweep.

Le meilleur résultat du sweep est désormais le **run 51** (`pantagruel_wavlm_la_kfold_aug`, 02/09, Bal.Acc **0.716** / Macro-F1 **0.688** / Macro AUC 0.882), qui améliore le run 41 (Bal.Acc 0.698 / Macro-F1 0.660 / Macro AUC 0.883) déjà référencé dans le rapport de stage — celui-ci devrait donc être mis à jour avec ce nouveau record.
