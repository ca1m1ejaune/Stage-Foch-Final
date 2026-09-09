<div align="center">

# 📖 LIS MOI !

*Le petit guide (pas si petit) pour la prochaine personne qui reprend ce projet.*

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![Statut](https://img.shields.io/badge/Statut-Recherche-yellow) ![Hôpital](https://img.shields.io/badge/Hôpital-Foch-lightgrey)

---

</div>

<a name="presentation"></a>
## Petite présentation

Salut, stagiaire ou chercheur, je sais pas haha.

Je me présente vite fait, je m'appelle Callum Holliday (prononce callome stp ಠ_ಠ). J'ai fait Biotech Numérique et en vrai j'aime les maths, l'IA, le rugby et je veux tellllemmmeeennt aller au Viêt Nam mais j'ai pas d'argent... Un jour peut-être :')

Ce fichier .md était pour te dire que tu vas participer à un super projet d'IA !

>  Retrouve tout mon travail sur GitHub : **[github.com/ca1m1ejaune](https://github.com/ca1m1ejaune)**

---

<a name="projet"></a>
## Le projet en un coup d'œil

J'ai fait pas mal de travail pour savoir quelle était la meilleure façon d'exploiter les données de FOCH. En vrai je suis plutôt optimiste maintenant, tu vas voir que des fois c'est difficile, tu vas probablement t'endormir devant l'écran pendant que tu entraînes des modèles, mais ne baisse pas les bras ! C'est de la recherche avant tout :) et n'abandonne pas même si tes résultats sont pas satisfaisants, ne lâche rien !

Le plus dur pour moi, c'était vraiment de me dire si c'était vraiment faisable ou pas, pour moi je pense que oui ! Certes, les données de FOCH sont limitées : on a réellement 168 audios exploitables avec 3 pathos. On aura forcément des problèmes d'overfitting, ou d'underfitting, bref... La classique dans l'IA clinique quoi haha.

**Chiffres clés**

| | |
|---|---|
| 🎙️ Audios exploitables | 168 |
| 🩺 Pathologies | 3 |
| 🖥️ VRAM disponible | 8 GB |
| 🔁 Cross-validation | 5 plis |
| 🏆 Meilleure config trouvée | WavLM-Large + Pantagruel 14K |
| 📊 Balanced accuracy | 0.716 |

Je vais essayer de te dire en gros ce qu'il faut que tu fasses et ce que tu dois regarder pour bien commencer parce qu'il y'a pas mal d'informations et j'espère que je t'ai mis dans de bonnes conditions et que tu te dises que j'ai un peu travaillé ahhahah (comparé à la stagiaire de 2025 askip ￣へ￣).

J'ai essayé au mieux de rendre mes ressources accessibles aux prochains stagiaires sur GitHub, donc va sur ma page :
**[https://github.com/ca1m1ejaune](https://github.com/ca1m1ejaune)**

J'ai pas mal de repos si tu veux voir plus en détail mais en vrai c'est beaucoup de lecture et tout vraiment c'est relou... Je te conseille vraiment d'utiliser ce dossier pour commencer.

Donc, dans ce dossier j'ai installé toutes les bibliothèques pour toi dans la venv (je sais que peut-être tu voulais installer toi-même pour faire vraiment le geek mais bon fallait être MEILLEUR !).

---

<a name="architecture"></a>
## L'architecture

Le plus important, c'est les notebooks que j'ai écrits, et c'est là où mes architectures ont été testées.

En gros j'ai fait une architecture qui va fusionner les vecteurs/embeddings de gros modèles Transformers (WavLM-Large, HuBERT, Wav2Vec etc...) et des fois CNN (les classiques ResNet puis des réseaux plus niches sur GitHub genre VGGVox, CNN14, ResNetSE34L... Vraiment leurs docs sont une galère pas possible), ViT (DinoV2) avec les représentations spectrales du signal en image.

Par exemple j'ai pris le modèle WavLM développé par Microsoft qui, en gros, prend un audio et traverse son réseau de neurones, et il a été pré-entraîné sur une tonne de données audios genre 90K heures. Je pense que c'est le réseau de neurones qui fait le plus gros taffe d'un point de vue représentation du signal dans l'espace latent. 
Puis je fais une autre branche qui va transformer le signal en image donc spectrogramme, scalogramme pour les passer en entrée dans un CNN. 


###  Schéma de l'architecture (vue générale)

```text
                                  ┌──────────────────────┐
                                  │ Audio /a/ (patient)  │
                                  └──────────────────────┘
                                             │
              ┌───────────────────────────────┼───────────────────────────────┐
              │                               │                               │
┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
│       Signal brut        │    │      Représentation      │    │         Vocodeur         │
│        (waveform)        │    │   spectrale (→ image)    │    │     (branche bonus)      │
└──────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
              │                               │                               │
┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
│     Transformer SSL      │    │        CNN / ViT         │    │      signal_origine      │
│   WavLM-Large, HuBERT,   │    │  ResNet, VGGVox, CNN14,  │    │     − signal_synthé      │
│   Wav2Vec, Pantagruel    │    │   ResNetSE34L, DinoV2    │    │   → stats (μ, σ, ...)    │
│          14K ★           │    │                          │    │                          │
└──────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
              └───────────────────────────────┴───────────────────────────────┘
                                              │
                              ┌──────────────────────────────┐
                              │    Fusion des embeddings     │
                              │   (backbones gelés, seule    │
                              │   la tête est fine-tunée)    │
                              └──────────────────────────────┘
                                              │
                              ┌──────────────────────────────┐
                              │    Tête de classification    │
                              └──────────────────────────────┘
                                              │
                              ┌──────────────────────────────┐
                              │      Pathologie prédite      │
                              └──────────────────────────────┘

   ★ Meilleure config trouvée : WavLM-Large + Pantagruel 14K → Balanced accuracy = 0.716
```

>  Vue simplifiée de l'archi, pour plus de détails des variantes testées, regarde mon rapport de stage.  

J'ai fait plein de combinaisons possibles avec WavLM et X archis, j'ai même fait avec 3 branches : WavLM + CNN/Transformer + Vocoder. La branche avec le vocodeur c'était tout simplement pour capter les détails du signal en faisant genre signal_origine - signal_synthé, et tu analyses les détails avec des stats genre moyenne, écart-type etc...

Mais l'architecture qui a eu les meilleurs résultats au final c'était WavLM-Large + Pantagruel 14K. Pantagruel c'est un modèle très important, c'est un genre de Transformer de type Data2Vec mais il a été entraîné sur un corpus purement en français. C'est pratique pour nous car on utilise des données de FOCH qui sont en français lol.

J'ai atteint une balanced accuracy de 0.716, ce qui, en soi, n'est pas mal — j'en suis certain, plus on aura de données, plus on aura un modèle fiable chez FOCH au long terme.

---

<a name="technique"></a>
##  Les trucs un peu plus techniques

À FOCH tu as pas mal de freins en vrai, en premier c'est les pare-feu de ce PC + les pare-feu de la connexion internet, normalement tu interfaces avec le hub de Hugging Face pour avoir tes modèles mais là tu es obligé(e) d'avoir les modèles téléchargés en local et tu dois passer par `git clone XXXX` du modèle.

Vu qu'on a pas beaucoup de données on peut que fine-tuner la tête de classification, on peut pas dégeler des couches, ça coûtera trop et ça va tourner tellement lentement... même si le PC sur lequel tu vas travailler a 8 GB de VRAM et qu'il semble rapide, il n'est pas non plus la plus grande machine de guerre genre.

Utilise tout le temps la Cross-Validation, pas le choix, tu dois faire 5 plis (je pense que c'est la meilleure séparation) comme ça tu valides et tu t'entraînes partout.

---

<a name="bonus"></a>
##  Bonus

Si tu veux, lis mon rapport de stage pour qu'il serve à qqch de plus que de valider ma note de stage à l'école :') il y aura largement plus de détails mais c'est vraiment énergivore, donc synthétise avec une IA générative.

Aussi une des améliorations qu'il faut dans le modèle c'est la reconversion de voix vers le même sexe. Je n'ai pas eu le temps de rajouter la colonne sexe dans le fichier orl_df_vowel_master.csv. J'ai un modèle qui a en entrée 

Tu vas voir que dans les benchmarks dans le dossier data/resultats tu vas voir que le meilleur modèle (mise à part celui évalué sur 3 classes qui n'est pas util dans notre cas) c'est le run qui est classé 2ème qui est un modèle un peu expérimentale que je voulais tester. 
Ce modèle est une combinaison de pleins de choses, Pantagruel + WavLM + Scalogramme (représentation en ondelettes cf. https://irma.math.unistra.fr/~franck/cours/SciML/output/html/chapTS_sec2.html) ce scalogramme passe dans un CNN et on a bcp de dimensions à la fin donc je passe par une PCA pour réduire la dimension. Au moment où j'écris ça je pense que c'est une approche assez intéressante et qui mérite d'être poussé. Donc toi ! essaie d'améliorer mon pipeline dans mon dernier notebook dans notebooks/exploration/wavelet/14_0_foch_test_cnn_pca.ipynb

Je te fais confiance

---

<p align="center"><sub>Bon courage, et ne lâche rien 🫡</sub></p>
