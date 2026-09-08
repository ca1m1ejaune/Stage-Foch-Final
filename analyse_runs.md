# Analyse des runs Pantagruel + autres branches — notes

Sweep loggé dans `runs_log.csv` : 54 runs réels du 5 août au 3 sept (2 trous : 18j après le run 18 j'étais en vacances, 7j après le run 47, retesté pour affiner les hyper-paramètres).

**Protocole fixe sur tous les runs** : 5-fold CV patient-level, audio 4s, batch effectif 16, 30 epochs, lr 1e-4, focal loss γ=2, seule la tête de fusion (+résiduel) est entraînée, backbones gelés. 265 échantillons (168 Foch + 100 SVD).

## （￣︶￣）↗　 Meilleur run : #51

`pantagruel_wavlm_la_kfold_aug` — 02/09 16h02 

WavLM-Large + Pantagruel (pas de résiduel), aug `h_vc+signal`, wd=0.02, dropout=0.5, input_dropout=0.1, 

- **Bal.Acc 0.716** / Macro-F1 0.688 / Macro AUC 0.882
- améliore mon ancien record (run 41 : 0.698) de +0.018

## Architectures (moyenne Bal.Acc par preset)

| Preset | n | moyenne | max |
|---|---|---|---|
| pantagruel_wavlm_la | 8 | 0.699 | 0.716 |
| pantagruel_wavlm_la_residual | 2 | 0.691 | 0.692 |
| wavlm_la_solo | 1 | 0.638 | — |
| pantagruel_wavlm_bp | 6 | 0.627 | 0.652 |
| pantagruel_wavlm_bp_residual | 6 | 0.625 | 0.658 |
| pantagruel_solo | 7 | 0.594 | 0.625 |
| pantagruel_residual | 6 | 0.593 | 0.659 |
| wavlm_bp_residual | 7 | 0.550 | 0.611 |
| wavlm_bp_solo | 6 | 0.517 | 0.587 |
| residual_solo | 5 | 0.398 | 0.473 |

**Ce que j'en tire :**
- Pantagruel aide systématiquement (+0.13 en moyenne, avec vs sans), c'est assez prometteur, celà veut dire qu'un Transformer pré entraîné sur du Français aide a avoir des métriques plus élevés à Foch. Donc c'est très important de savoir ça. 
- WavLM-Large >> Base-Plus (0.692 vs 0.579 en moyenne), finalement le modèle WavLM-Large garde des détails importants dans la représentation latente du signal. 
- Le résiduel seul est quasi random (0.40, jusqu'à 0.315) — utile seulement combiné à d'autres branches, c'est logique en vrai
- Le résiduel en plus de WavLM-Large+Pantagruel n'aide pas (0.691 vs 0.699 ; #55 < #53/#54 à hp identiques), ce résultat me déçoit un peu car les fin détails avec la différence entre la voix orignal et le vocodeur pour afficher des détails plus fin au niveau de la représentation latente des résidus. C'était une idée assez brillante du professeur Ossonce. **Il faut que la prochaine personne, qui prend le projet s'y penche, cest carrément une piste à approfondir !** 

## Augmentation : empiler aide, toujours

| Type | n | moyenne | max |
|---|---|---|---|
| h_vc+signal | 16 | 0.656 | 0.716 |
| signal+vc | 7 | 0.607 | 0.659 |
| vc | 7 | 0.593 | 0.658 |
| signal | 10 | 0.557 | 0.639 |
| h_vc | 7 | 0.538 | 0.621 |
| aucune | 7 | 0.508 | 0.603 |

Gradient monotone : plus je combine de types d'aug, mieux c'est. `aucune` = toujours la pire option à archi égale, ce qui est logique, les augmentations de données améliorent toujours la robustesse du modèle. 

## Régularisation : 3 phases

| Phase | Dates | dropout | input_dropout | weight_decay |
|---|---|---|---|---|
| 1 | 5-6 août | 0.2 | 0.1 | 0.1 |
| 2 | 24-26 août | 0.5 | 0.3 | 0.1 |
| 3 (ciblée, meilleure archi seulement) | 2-3 sept | 0.3-0.5 | 0.1-0.3 | 0.02-0.1 |

- Bascule phase 1→2 exactement au run 18, juste avant le trou de 18j (vacances)
- Run 25 (à la reprise) reproduit exactement le score du run 18 (0.5274) → pipeline reproductible
- Phase 3 : baisser wd (0.1→0.02) et input_dropout (0.3→0.1) gagne +0.018 Bal.Acc sur la meilleure archi — pas monotone pour autant (wd=0.03 fait moins bien que 0.02)
- ⚠️runs 53/54 : mêmes hyperparamètres exacts, écart de 0.034 (pas de seed fixée) → une partie des petits écarts ailleurs dans le sweep est probablement du bruit, pas du signal

## À reporter dans le rapport de stage

- [ ] Remplacer le run 41 par le run 51 comme meilleur résultat du projet (Bal.Acc 0.698→0.716, Macro-F1 0.655→0.688)
- [ ] Mentionner le bruit d'entraînement (~0.03, runs 53/54) comme repère pour ne pas surinterpréter les petits écarts du sweep
