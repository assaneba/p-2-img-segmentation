# Segmentation de vêtements avec SegFormer

Segmentation sémantique d'images de personnes (vêtements et parties du corps) avec le modèle
[`sayeed99/segformer_b3_clothes`](https://huggingface.co/sayeed99/segformer_b3_clothes) de Hugging Face,
**exécuté en local** : aucun token, aucune API payante, aucune limite de requêtes.

Chaque pixel reçoit l'une des 18 classes : `Background`, `Hat`, `Hair`, `Sunglasses`, `Upper-clothes`,
`Skirt`, `Pants`, `Dress`, `Belt`, `Left-shoe`, `Right-shoe`, `Face`, `Left-leg`, `Right-leg`,
`Left-arm`, `Right-arm`, `Bag`, `Scarf`.

## Pourquoi en local ?

Le notebook d'origine appelle l'API d'inférence de Hugging Face. Celle-ci exige désormais des
**crédits prépayés** : sans crédits, elle répond `402 Payment Required`, même avec un token valide.
Le même modèle tourne très bien sur un ordinateur portable (environ 190 Mo, ~1,5 s par image sur CPU).

> Les outils comme LM Studio ne conviennent pas ici : ils ne chargent que des LLM (GGUF/MLX),
> pas des modèles de vision comme SegFormer. On passe donc par `transformers` + `torch`.

## Installation en local

Prérequis : Python 3.10 ou plus récent, et environ 2 Go d'espace disque libre (PyTorch + modèle).

```bash
# 1. Cloner le dépôt
git clone https://github.com/assaneba/p-2-img-segmentation.git
cd p-2-img-segmentation

# 2. Créer et activer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

Aucune clé n'est nécessaire : le fichier `.env` n'est utile que si vous voulez réutiliser l'API
Hugging Face (variable `HF_TOKEN`). Il est ignoré par Git.

## Utilisation

### Script

```bash
python segmentation.py
```

Le script :
1. liste les images de `assets/IMG` (dossier non versionné : déposez-y vos propres images `.png`/`.jpg`) (3 par défaut, voir `max_images`) ;
2. télécharge le modèle au premier lancement (~190 Mo, mis en cache dans `~/.cache/huggingface`),
   puis fonctionne hors ligne ;
3. segmente chaque image et écrit un fichier JSON par image dans `assets/Mask/`
   (même format que l'API : `[{"label", "score", "mask"}]`, masque PNG encodé en base64) ;
4. affiche l'image originale et son masque coloré avec matplotlib.

Pour traiter toutes les images, mettez `max_images = None` dans le script.

### Notebook

Ouvrez `huggingface_api_cloth_seg.ipynb` avec le kernel du `.venv` (`pip install ipykernel` si besoin)
et exécutez les cellules dans l'ordre. Les étapes 5 (segmentation en batch) et 6 (affichage en grille)
utilisent le modèle local.

## Structure

```
.
├── segmentation.py                    # script complet (segmentation + export JSON + affichage)
├── huggingface_api_cloth_seg.ipynb    # notebook pas à pas
├── requirements.txt
└── assets/                            # non versionné (ignoré par Git)
    ├── IMG/                           # vos images d'entrée
    └── Mask/                          # JSON générés (image_N.json)
```

## Problèmes courants

| Symptôme | Cause / solution |
|---|---|
| `402 Payment Required` | Vous utilisez l'API HF sans crédits. Utilisez la version locale. |
| `No space left on device` | Libérez de l'espace : PyTorch + modèle demandent ~2 Go. |
| `IProgress not found` | `tqdm.notebook` hors Jupyter. Utilisez `from tqdm import tqdm`. |
| Avertissement `requires torchvision` | Sans conséquence ; `pip install torchvision` le supprime. |
