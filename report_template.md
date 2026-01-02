# Rapport de projet — Classification d’images

**Oxford-IIIT Pet Dataset**

---

## Introduction

Ce projet porte sur la classification automatique d’images du jeu de données **Oxford-IIIT Pet**, composé de 37 races de chiens et de chats. L’objectif est de prédire la race présente sur une image à partir d’une photographie RGB, dans un cadre de **classification multi-classe**.

Bien que le dataset soit relativement équilibré, la tâche reste non triviale. Il s’agit d’un problème de **classification fine-grained**, dans lequel les classes se distinguent parfois par des différences visuelles subtiles (texture du pelage, forme des oreilles, proportions du museau), souvent perturbées par des variations importantes de pose, d’éclairage, de cadrage ou d’arrière-plan.

L’enjeu principal de ce projet n’est donc pas uniquement d’obtenir une bonne précision globale, mais de construire un pipeline cohérent, interprétable, et capable de généraliser, tout en comprenant précisément **où et pourquoi le modèle se trompe**.

---

## 1. Données et prétraitement

### 1.1 Description du jeu de données

Le jeu de données Oxford-IIIT Pet contient des images RGB de tailles variables, représentant des chiens et des chats répartis en **37 classes**. Chaque image est associée à une unique étiquette correspondant à la race.

Après chargement des données, les tailles des ensembles sont les suivantes :

* **Train** : 3311 images
* **Validation** : 369 images
* **Test** : 3669 images

Les dimensions d’entrée du modèle sont fixées à **(3, 128, 128)**.

La distribution des classes est quasi uniforme, aussi bien sur l’ensemble train/validation que sur l’ensemble test, comme le montrent les histogrammes de distribution. Cette propriété permet d’utiliser des métriques macro (macro-F1, balanced accuracy) sans craindre qu’elles soient dominées par quelques classes majoritaires.

---

### 1.2 Prétraitement des images

Toutes les images subissent les transformations suivantes :

* redimensionnement à 144×144 ;
* recadrage central à 128×128 ;
* normalisation à l’aide des statistiques ImageNet (mean/std).

Ces choix visent à :

* standardiser les entrées pour le réseau convolutif ;
* limiter les distorsions géométriques excessives ;
* conserver une compatibilité avec des architectures CNN classiques.

Les ensembles de validation et de test utilisent exactement les mêmes prétraitements déterministes, garantissant une évaluation cohérente.

---

### 1.3 Augmentation de données

L’augmentation est appliquée **uniquement sur l’ensemble d’entraînement**. Elle inclut :

* Random Resized Crop (variations d’échelle et de position) ;
* flip horizontal (symétrie naturelle des animaux) ;
* rotations légères ;
* jitter de couleur (luminosité, contraste, saturation) ;
* random erasing (régularisation spatiale).

Des visualisations d’exemples après augmentation montrent que les transformations restent réalistes et conservent l’identité visuelle des animaux. Cette étape est essentielle pour vérifier que l’augmentation n’introduit pas d’artefacts destructeurs.

---

## 2. Modèle et architecture

### 2.1 Architecture générale

Le modèle implémenté est un réseau convolutionnel en trois blocs principaux, avec une augmentation progressive du nombre de canaux. Chaque bloc est composé de convolutions suivies de normalisation et d’activation ReLU.

Des **blocs Squeeze-and-Excitation (SE)** sont intégrés afin de permettre au réseau de pondérer dynamiquement l’importance des canaux en fonction du contenu global de l’image. Cette approche est particulièrement adaptée aux tâches de classification fine, où certaines caractéristiques visuelles sont pertinentes uniquement dans certains contextes.

La tête de classification repose sur un **Global Average Pooling**, suivi d’une couche linéaire produisant 37 logits.

Le modèle contient environ **1,2 million de paramètres**, tous entraînables.

---

### 2.2 Justification des choix

L’utilisation de blocs SE vise à améliorer la capacité du modèle à exploiter des indices visuels fins sans recourir à une augmentation massive de la profondeur ou de la résolution. En revanche, ce mécanisme ne capture pas explicitement les relations spatiales fines, ce qui constitue une limite connue de cette architecture.

---

## 3. Vérifications initiales et validation du pipeline

Avant tout entraînement long, une étape de **sur-apprentissage volontaire sur un très petit sous-ensemble** a été réalisée. Le but est de vérifier que :

* la loss est correctement implémentée ;
* les gradients sont non nuls ;
* le modèle est capable de mémoriser des exemples simples.

Le modèle parvient à faire décroître la loss jusqu’à des valeurs proches de zéro sur ce sous-ensemble, ce qui valide l’implémentation du pipeline d’entraînement.

Cette étape ne vise pas la performance, mais la **fiabilité du code**, et permet d’éviter des erreurs silencieuses coûteuses.

---

## 4. Choix des hyperparamètres et stratégie d’entraînement

### 4.1 Recherche du taux d’apprentissage

Une exploration du taux d’apprentissage a été effectuée afin d’identifier une zone stable où la loss décroît rapidement sans divergence. Le taux retenu se situe juste en-dessous de la zone d’instabilité observée, ce qui permet une convergence rapide tout en conservant une dynamique stable.

L’optimiseur utilisé est **AdamW**, choisi pour sa robustesse et sa gestion explicite de la régularisation par weight decay.

Un scheduler de type **cosine decay avec warmup** est employé, afin de limiter l’instabilité en début d’entraînement et d’affiner la convergence finale.

---

### 4.2 Comparaison des stratégies de régularisation (A vs B)

Deux configurations principales ont été comparées :

* **Expérience A** : utilisation de mixup, sans label smoothing ;
* **Expérience B** : utilisation de label smoothing (0.1), sans mixup.

Ces deux techniques visent à améliorer la généralisation, mais reposent sur des mécanismes différents. Mixup agit directement sur les données, tandis que le label smoothing agit sur la fonction de perte et la confiance du modèle.

Les courbes d’entraînement montrent que les deux approches atteignent des performances comparables, mais que l’expérience B présente une dynamique plus régulière et une meilleure stabilité sur les métriques de validation.

---

## 5. Résultats finaux sur l’ensemble de test

Le modèle final sélectionné atteint les performances suivantes sur l’ensemble de test :

* **Accuracy top-1** : 57.1 %
* **Accuracy top-5** : 87.3 %
* **Macro-F1** : 0.56
* **Balanced accuracy** : 0.57
* **ECE (Expected Calibration Error)** : 0.032

Ces résultats sont cohérents avec la difficulté intrinsèque de la tâche, compte tenu de la résolution limitée et de l’entraînement depuis zéro.

---

## 6. Analyse des erreurs

### 6.1 Matrice de confusion

La matrice de confusion met en évidence une diagonale dominante, indiquant une bonne capacité de discrimination globale. Les erreurs restantes ne sont pas aléatoires : elles se concentrent principalement entre races visuellement proches.

Certaines confusions sont symétriques, suggérant une ambiguïté réelle entre classes, tandis que d’autres sont asymétriques, traduisant un biais du modèle vers certaines caractéristiques dominantes.

---

### 6.2 Analyse par classe

L’analyse des performances par classe montre une variabilité modérée, sans effondrement sur un sous-ensemble spécifique. Cette observation est cohérente avec la proximité entre la macro-F1 et la weighted-F1, indiquant une performance relativement homogène.

---

### 6.3 Analyse qualitative des erreurs

L’inspection visuelle des images mal classées révèle plusieurs facteurs récurrents :

* poses atypiques ;
* visages partiellement masqués ;
* recadrages défavorables ;
* éclairages extrêmes ;
* ambiguïtés intrinsèques entre certaines races.

Ces erreurs reflètent davantage la difficulté des données que des défaillances grossières du modèle.

---

## 7. Calibration et fiabilité des prédictions

L’analyse de la calibration à l’aide de courbes de fiabilité montre une bonne adéquation entre la confiance prédite et la précision empirique. La faible valeur de l’ECE confirme que les probabilités produites par le modèle sont globalement bien calibrées.

L’utilisation du label smoothing dans la configuration finale contribue probablement à cette propriété, en limitant les prédictions excessivement confiantes.

---

## 8. Limites et perspectives

Plusieurs limites peuvent être identifiées :

* entraînement depuis zéro, sans transfert d’apprentissage ;
* résolution d’entrée relativement faible pour une tâche fine-grained ;
* absence de mécanismes explicites d’attention spatiale ;
* recherche d’hyperparamètres limitée.

Des améliorations possibles incluraient l’utilisation de modèles pré-entraînés, une augmentation de la résolution, ou l’ajout de mécanismes d’attention spatiale ou de stratégies d’augmentation plus ciblées.

---

## Conclusion

Ce projet met en œuvre un pipeline complet de classification d’images, depuis la validation du code jusqu’à l’analyse fine des erreurs et de la calibration. Les résultats obtenus sont cohérents avec la difficulté du problème et illustrent l’importance d’une démarche méthodique, fondée sur l’analyse et la compréhension du comportement du modèle, au-delà des métriques brutes.
