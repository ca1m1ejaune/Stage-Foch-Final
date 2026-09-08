# Classification des pathologies vocales — Hôpital Foch

Pipeline de deep learning pour la classification automatique de pathologies laryngées à partir d'enregistrements de voyelles tenues. Projet de stage, données cliniques réelles de l'Hôpital Foch complétées par la base publique Saarbrücken (SVD) comme groupe contrôle sain.

Ce document s'adresse au **stagiaire ou personne qui reprend le projet**. Pour la chronologie détaillée, les résultats chiffrés de chaque architecture testée et les échanges avec les encadrants, voir `RAPPORT_CONTEXTE.md` (rédigé par le stagiaire précédent pour son propre rapport — narratif, pas destiné à être maintenu).

---

## 1. Vue d'ensemble

**Objectif clinique** : distinguer, à partir d'un enregistrement de voyelle tenue, quatre catégories :

| Classe | Description | Support (FOCH) |
|---|---|---|
| `PR` | Presbylarynx — pathologie vocale liée à l'âge | ~70-100 |
| `LMB` | Lésion laryngée bénigne | ~70-100 |
| `Fuite glottique` | Dysphonie par fermeture incomplète de la glotte | **26 — classe minoritaire critique** |
| `Healthy` | Groupe contrôle sain (base Saarbrücken/SVD) | sous-échantillonné pour équilibrage |

La classe `Cancer` est volontairement exclue ( en dessous de 20 échantillons, support jugé insuffisant pour un modèle fiable).

**Données** :
- **FOCH** : voyelles tenues segmentées, données cliniques réelles, confidentielles. Métadonnées dans un CSV maître (colonnes patient/pathologie) — **ce CSV et les fichiers audio ne sont jamais stockés dans ce dépôt**, ils vivent sur un chemin externe (voir §2).
- **SVD (Saarbrücken)** : base de référence publique, sert de groupe contrôle sain (voyelle `/a/` à hauteur habituelle uniquement, pour cohérence acoustique avec FOCH).

**Confidentialité** : les données FOCH sont des données patient réelles. Ne jamais copier de fichier audio, de CSV nominatif ou de sortie de notebook contenant de l'audio embarqué dans un dossier versionné ou public. Voir §7 pour l'historique de ce point et l'état à la reprise du projet.

---

## 2. Installation de l'environnement

### Contraintes réseau — mode hors ligne

La machine de développement **n'a pas accès à huggingface.co**. Tous les modèles pré-entraînés (WavLM, AST, DINOv2, etc.) doivent être téléchargés une fois puis chargés localement. Deux variables d'environnement **doivent être positionnées avant l'import de `transformers`** (la bibliothèque les lit à l'initialisation) :

```python
import os
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

Les modèles HuggingFace sont ensuite chargés avec `local_files_only=True` depuis un cache local (voir `C:/Users/AdminIA/Documents/models/` ci-dessous). Les notebooks `1_0` fournissent les commandes `huggingface-cli download` à lancer en amont, sur une machine ayant accès au réseau, avant de rapatrier le cache sur la machine de travail.

Pour les modèles chargés via `torch.hub` / `torchvision` (ResNet101, DINOv2), il est recommandé de fixer également `TORCH_HOME` vers un répertoire local persistant (hors du dossier utilisateur par défaut), pour les mêmes raisons de disponibilité réseau et de traçabilité du cache — configurez-le en variable d'environnement système si ce n'est pas déjà fait sur la machine.

### Dépendances Python

```bash
python -m pip install -r requirements.txt
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
```

`requirements.txt` (à la racine) couvre les notebooks de `notebooks/production/` : cœur deep learning, HuggingFace, data science, et les extras nécessaires à la lignée Pantagruel (`vocos`, `PyYAML`, `PyWavelets`) et à l'augmentation (`soundfile`, `praat-parselmouth`). Les dépendances propres à certains notebooks d'exploration (`librosa`, `audiomentations`, `optuna`, `reservoirpy`, `bitsandbytes`) sont listées en commentaire dans le fichier — à décommenter selon le notebook que vous rouvrez. Reconstruit à partir des imports réels des notebooks (l'original vivait dans `Classification_Algo/requirements.txt`, dossier supprimé depuis — voir §3).

Notes connues :
- Python 3.14 utilisé pour une partie du projet, 3.10/3.12 pour d'autres notebooks (`speech-large-14K`/Pantagruel exige 3.12) — **vérifiez la version Python attendue en tête de chaque notebook** avant de créer l'environnement.
- `audiomentations`/`librosa` volontairement absents (pas de wheels compatibles Python 3.14 sur cette machine à l'époque) : l'augmentation signal-level est réimplémentée en NumPy/torchaudio/SciPy.
- FFmpeg doit être dans le `PATH` (backend de décodage `torchaudio`) — chemin actuellement codé en dur dans plusieurs notebooks (`C:/Users/.../ffmpeg-8.1.1-full_build-shared/.../bin`), à adapter à votre installation.

### Chemins de données externes (non versionnés)

```
C:/Users/AdminIA/Documents/models/                          # cache modèles HF/torch en local
C:/Users/AdminIA/Documents/SVD/healthy/<patient_id>/vowels/*a_h.wav
C:/Users/AdminIA/Desktop/ORL_IA_FOCH_Callum_HOLLIDAY/
  ├── csv_best/orl_df_vowel_master.csv                       # métadonnées FOCH (nominatif)
  ├── Segmented_Voyelles_Best/                                # audio FOCH
  └── Segmented_Voyelles_Augmented/{signal,vc}/               # audio augmenté + manifestes
```

Ces trois racines sont propres à la machine de développement d'origine. Si vous migrez sur une autre machine, recréez la même arborescence ou mettez à jour les variables `CSV_PATH` / `FOCH_AUDIO_DIR` / `SVD_HEALTHY_DIR` / `AUG_DIR` en tête de chaque notebook (cellule "Configuration").

---

## 3. Structure du dépôt

```
Foch_Classif_Final/
├── README.md                    # ce fichier
├── RAPPORT_CONTEXTE.md          # journal narratif du stagiaire précédent (pour rédaction de rapport)
├── requirements.txt              # dépendances Python — voir §2
├── notebooks/
│   ├── production/               # notebooks de référence — dernière version stable de chaque lignée
│   └── exploration/               # essais, ablations, backbones abandonnés — archivés, pas supprimés
├── src/
│   └── foch_utils.py             # fonctions partagées (dataset, split patient, entraînement, éval)
└── data/                          # manifestes non sensibles/anonymisés seulement — voir .gitignore
```

Les dossiers d'origine (`Classification_Algo/`, `transfer_learning/`, `specaug/`, `vocoder/`, `pantagruel_test/`) ont été supprimés par l'ancien stagiaire une fois leur contenu utile copié dans `notebooks/`. Si vous avez besoin de retrouver une version de code non reprise ici (variante non retenue, cellule intermédiaire), trois dossiers sources restent disponibles ailleurs sur la machine :

- `C:\Users\AdminIA\Desktop\orl_ia_foch`
- `C:\Users\AdminIA\Desktop\ORL_IA_FOCH_Callum_HOLLIDAY`
- `C:\Users\AdminIA\Desktop\Classif_Aug`

⚠️ Ces trois dossiers n'ont pas été audités pour ce dépôt (contrairement à `notebooks/`, dont l'audio embarqué et les chemins périmés ont été nettoyés) — s'ils contiennent des enregistrements ou métadonnées patient, appliquez la même prudence qu'au §7 avant tout partage ou versionnement.

### Notebooks de production

| Notebook | Étape | Description | Résultat clé |
|---|---|---|---|
| `1_0_foch_test.ipynb` | Sélection d'architecture | Benchmark 9 modèles SSL, classification binaire Healthy vs. Pathologique | WavLM-Large, AUC 0.999 |
| `2_0_foch_test.ipynb` | Baseline | Classification 4 classes, WavLM-Large, sans augmentation | Balanced Acc ~0.69, Macro-F1 ~0.71 |
| `3_0_foch_test.ipynb` | Augmentation | Ablation E0→E3 (signal-level, SpecAugment, SeedVC), validation acoustique parselmouth | E2 (+SpecAugment) : Balanced Acc ~0.75 |
| `4_0_foch_test.ipynb` | Fusion | Bi-branche WavLM-Large + ResNet101 (spectrogramme), 5-fold CV | OOF Balanced Acc 0.646 ± 0.075 |
| `5_0_foch_test_AST_HP_fixed.ipynb` | Backbone alternatif | Meilleure itération de la lignée AST (Audio Spectrogram Transformer) fine-tuné | voir cellules d'évaluation du notebook |
| `10_0_foch_test_pantagruel.ipynb` | Fusion tri-branche | WavLM Base-Plus + Pantagruel (data2vec-2.0, 14k h de parole **française**) + résidu vocodeur (Vocos) | meilleur run `pantagruel_wavlm_la_kfold_aug` (variante WavLM-Large) : OOF Balanced Acc 0.698, Macro-F1 0.655, Macro AUC 0.883 |
| `11_0_foch_test_cwt_dinov2.ipynb` | Fusion quadri-branche | Ajoute une 4e branche : scalogramme CWT (ondelette de Morlet) → DINOv2-Large gelé. Architecture la plus complète, branches activables indépendamment | run le plus récent — comparer aux runs `runs_log.csv` |

`runs_log.csv` (à côté de ces notebooks) journalise 47 runs de la lignée Pantagruel (hyperparamètres + métriques OOF) — utile pour comparer sans tout relancer.

### Notebooks d'exploration (archivés par lignée d'origine)

- `exploration/classification_algo_iterations/` — suite non documentée après `4_0` (`5_0`, `6_0`, `6_0_upgrade`, `SVD.ipynb`), gardée pour référence.
- `exploration/transfer_learning_backbones/` — tous les essais de backbones alternatifs : AST (variantes HP, 3-classes, +SeedVC, +SpecAugment), MERT, CNN14, essai `reservoirpy` 3-branches, VoxCeleb transfer learning.
- `exploration/specaugment/` — pipeline d'augmentation SpecAugment autonome + essais bi-branche WavLM+VGGVox.
- `exploration/vocoder/` — notebook de classification AST+Vocoder et le notebook de R&D qui a fait naître l'idée de "résidu vocodeur" (repris ensuite dans `10_0`).
- `exploration/pantagruel/` — génération précédant `10_0`/`11_0` : `9_0` (variante traits Praat), `10_0_..._variante_praat.ipynb` (même variante, notebook 10), `10_1_...svm_optuna.ipynb` (ablation : mêmes données/plis que `10_0`, modèle remplacé par un SVM).
- `exploration/_doublons_ou_temp/` — fichiers identifiés comme quasi-doublons (copie antérieure, sauvegarde automatique horodatée) : conservés par prudence plutôt que supprimés, à trier/purger si confirmé inutiles.

**Sorties nettoyées** : les notebooks originaux `specaug/specaug.ipynb` et `vocoder/vocoder_residual_exploration.ipynb` (dossiers depuis supprimés) contenaient des lecteurs audio avec du WAV encodé en base64 directement dans les sorties de cellules (probablement des enregistrements réels FOCH/SVD utilisés pour illustrer une augmentation/resynthèse). Ces sorties ont été retirées des copies conservées ici (`notebooks/exploration/specaugment/specaug.ipynb` et `notebooks/exploration/vocoder/vocoder_residual_exploration.ipynb`, 22 sorties supprimées au total) — voir §7, ce point reste à traiter dans l'historique git déjà publié sur GitHub.

---

## 4. Lancer un entraînement ou une évaluation

1. Ouvrir le notebook de production visé dans `notebooks/production/`.
2. Vérifier/adapter la cellule "Configuration" en tête de notebook : chemins `CSV_PATH`, `FOCH_AUDIO_DIR`, `SVD_HEALTHY_DIR`, `AUG_DIR`, et les deux variables d'environnement hors-ligne (§2).
3. Vérifier `shutil.which("ffmpeg")` renvoie un chemin valide.
4. Exécuter les cellules dans l'ordre. Chaque notebook sauvegarde :
   - les poids du meilleur modèle par pli (`torch.save(model.state_dict(), ...)`),
   - un dashboard PNG (matrice de confusion brute + normalisée, F1 par classe, courbes d'apprentissage),
   - un JSON de métriques et un CSV de matrice de confusion,
   dans le répertoire `.foch_métriques`/`.poids_modèles` externe (§2) — pas dans ce dépôt.
5. Pour la lignée Pantagruel (`10_0`, `11_0`), chaque run ajoute une ligne à `runs_log.csv` (chemin relatif au notebook) — comparez avant de relancer un run déjà loggé avec les mêmes hyperparamètres.

Durée indicative (GPU 8 Go VRAM) : `1_0` ~1-2h (9 modèles), `2_0` ~20-30 min, `3_0` ~60-90 min, `4_0` ~2-3h (5 plis), lignée Pantagruel/CWT-DINOv2 : compter plusieurs heures par sweep complet vu le nombre de branches et de plis.

---

## 5. Décisions de conception importantes

- **Backbones gelés (`freeze_fe=True` / branches gelées)** : seule la tête de classification (et éventuellement la couche de fusion) est fine-tunée. Avec un jeu de données de cette taille (~200-300 échantillons pathologiques), fine-tuner un transformeur de plusieurs centaines de millions de paramètres en entier sur-apprendrait immédiatement. Dégeler progressivement les dernières couches est identifié comme piste non explorée à fond (voir §6).
- **Split groupé par patient (`StratifiedGroupKFold` sur `Last_Name`)** : un patient ne peut jamais apparaître à la fois en train et en validation. Sans cette garde-fou, un modèle pourrait apprendre à reconnaître un *timbre de voix* individuel plutôt qu'une pathologie, et la métrique de validation serait artificiellement gonflée. Chaque notebook vérifie explicitement l'absence de fuite (`assert not fuite`) en fin de cellule de split.
- **Déduplication patient (un enregistrement par `Last_Name`)** : évite qu'un patient sur-représenté dans le CSV source ne biaise l'entraînement. Limite connue : la clé utilisée est le nom de famille, pas un identifiant patient clinique — risque d'homonymes non résolu (voir §6).
- **Métriques macro/balanced plutôt que accuracy globale** : le jeu de données est déséquilibré (`fuite glottique` : 26 échantillons vs 70-100 pour les autres classes). Une accuracy globale serait dominée par les classes majoritaires et masquerait un modèle qui ignore purement la classe minoritaire. D'où l'usage systématique de balanced accuracy, macro-F1 et macro AUC (OvR).
- **Augmentation strictement train-only, post-split** : la validation reste à 100% des données originales non augmentées, et aucune variante augmentée d'un patient de validation n'est injectée en train (vérifié par assertion). Objectif : ne jamais laisser l'augmentation gonfler artificiellement la performance mesurée.
- **Validation acoustique des augmentations (parselmouth)** : toute augmentation synthétique (notamment SeedVC, conversion de voix) est vérifiée via jitter/shimmer/HNR — si ces marqueurs pathologiques s'effondrent de plus de 50%, la variante est rejetée (le label pathologique deviendrait faux).
- **Choix de Pantagruel (data2vec-2.0, 14k h de parole française)** : les autres backbones SSL testés (WavLM, AST, HuBERT, MERT) sont pré-entraînés majoritairement sur de l'anglais ou de la musique. Le corpus FOCH est en français — l'hypothèse testée est qu'un encodeur pré-entraîné en français capture mieux les indices phonétiques/prosodiques pertinents.
- **Résidu vocodeur (Vocos) comme feature** : hypothèse que la resynthèse d'un vocodeur neuronal "lisse" naturellement les irrégularités pathologiques (jitter, shimmer, bruit) puisqu'il a appris une distribution de parole saine ; le résidu (original − resynthèse) capturerait alors un signal directement corrélé à la pathologie. Idée née dans `vocoder/vocoder_residual_exploration.ipynb`, intégrée en feature dans `10_0`.

---

## 6. Limitations connues et pistes non résolues

- **Confusion `fuite glottique` ↔ `PR`** : observée sur plusieurs architectures. À interpréter avec prudence — `fuite glottique` n'a que 26 échantillons au total (5 par pli en validation 5-fold), donc une partie de la confusion peut être un simple effet de faible échantillon statistique plutôt qu'une réelle proximité acoustique. Piste non tranchée : regarder si l'erreur persiste à effectif comparable (sous-échantillonnage des autres classes) avant de conclure à une confusion acoustique intrinsèque.
- **Homonymes patients** : la déduplication utilise `Last_Name`, pas un identifiant clinique (`IPP`). Le README d'origine signale explicitement au moins un cas (nom porté par 7 enregistrements) — un audit avec l'identifiant patient réel n'a pas été fait.
- **Dégel progressif non exploré à fond** : `4_0` amorce l'idée (dégeler les top-K couches du transformeur) sans l'avoir généralisée aux architectures suivantes.
- **Chemins absolus périmés** : 15 notebooks référencent encore une ancienne racine de projet (`Classif_Aug` au lieu de `Foch_Classif_Final`) — corrigé automatiquement dans les copies sous `notebooks/`, mais **pas** dans les dossiers d'origine à la racine.
- **SeedVC (E3, notebook `3_0`)** : dépend d'une installation externe (dépôt + checkpoints GPU) souvent absente — le gate parselmouth rejette fréquemment ce tier, l'apport réel de cette augmentation reste peu concluant.
- **`runs_log.csv`** : le suivi d'expériences systématique (hyperparamètres + métriques) n'existe que pour la lignée Pantagruel (47 runs). Les lignées antérieures (Classification_Algo, transfer_learning, specaug) n'ont pas ce suivi structuré — seuls les résultats "typiques" en README/markdown font foi.
- **Deux "Notebook 10" concurrents** trouvés dans `pantagruel_test/` au moment de la reprise (variante traits Praat vs variante résidu vocodeur) — la variante résidu vocodeur a été retenue comme référence en `notebooks/production/`, la variante Praat archivée en exploration. À confirmer que c'est bien la bonne lecture si vous repartez de ces notebooks.

---

## 7. Point de sécurité des données — état à la reprise du projet

Deux dépôts GitHub existent, issus de ce projet (les dossiers git locaux correspondants, `Classification_Algo/` et `vocoder/`, ont depuis été supprimés — leur historique n'existe donc plus que sur GitHub) :
- `ca1m1ejaune/Classification-FOCH-Test`
- `ca1m1ejaune/Vocoder`, dont le notebook `vocoder_residual_exploration.ipynb` committé contenait (au moment de l'audit initial) de l'audio WAV encodé en base64 dans les sorties de cellules — potentiellement des enregistrements patients.

Ce dépôt local (`Foch_Classif_Final/`) n'est **pas** un dépôt git à ce jour. Avant d'en créer un, ou de committer quoi que ce soit vers l'un des deux dépôts ci-dessus :
1. Vérifiez leur visibilité et leur historique directement sur GitHub (les dossiers locaux ne permettent plus de le faire) — confirmez avec l'ancien stagiaire si le passage en privé / la purge d'historique a bien été fait.
2. N'ajoutez jamais `data/` au suivi git (un `.gitignore` y est déjà présent et exclut tout le contenu du dossier).
3. Avant tout `git add`, vérifiez qu'aucun fichier CSV/notebook ajouté ne contient de sortie audio embarquée (`grep -r "data:audio" *.ipynb`) ni de colonne nominative en clair.

---

## Références

- WavLM : Chen, S., et al. (2022). *WavLM: Large-Scale Self-Supervised Pre-Training for Speech*
- Wav2Vec 2.0 : Baevski, A., et al. (2020)
- HuBERT : Hsu, W.-N., et al. (2021)
- AST : Gong, Y., et al. (2021). *AST: Audio Spectrogram Transformer*
- Whisper : Radford, A., et al. (2022)
- Vocos (vocodeur neuronal) — voir `notebooks/exploration/vocoder/vocoder_residual_exploration.ipynb` pour les références précises utilisées
