import os
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import base64
import json
import io
import os 
from dotenv import load_dotenv

# TODO: Modifiez ces valeurs selon votre configuration
image_dir = "./assets/IMG"  # Exemple : si vous êtes sur Colab et avez uploadé un dossier
max_images = 3  # Commençons avec peu d'images

# IMPORTANT: Remplacez "VOTRE_TOKEN_HUGGING_FACE_ICI" par votre véritable token API.
# Ne partagez jamais votre token publiquement.

load_dotenv()
api_token = os.getenv("HF_TOKEN")  # Récupération du token depuis les variables d'environnement

# Créons le dossier d'images s'il n'existe pas (pour l'exemple)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
    print(f"Dossier '{image_dir}' créé. Veuillez y ajouter des images .jpg ou .png.")
else:
    print(f"Dossier '{image_dir}' existant.")

if not api_token:
    print("\nATTENTION : Vous devez remplacer 'VOTRE_TOKEN_HUGGING_FACE_ICI' par votre token API personnel.")


API_URL = "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"
headers = {
    "Authorization": f"Bearer {api_token}"
    # Le "Content-Type" sera ajouté dynamiquement lors de l'envoi de l'image
}

# Lister les chemins des images à traiter
# Assurez-vous d'avoir des images dans le dossier 'image_dir'!
image_paths = sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))[:max_images]


if not image_paths:
    print(f"Aucune image trouvée dans '{image_dir}'. Veuillez y ajouter des images.")
else:
    print(f"{len(image_paths)} image(s) à traiter : {image_paths}")


CLASS_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17
}

def get_image_dimensions(img_path):
    """
    Get the dimensions of an image.

    Args:
        img_path (str): Path to the image.

    Returns:
        tuple: (width, height) of the image.
    """
    original_image = Image.open(img_path)
    return original_image.size

def decode_base64_mask(base64_string, width, height):
    """
    Decode a base64-encoded mask into a NumPy array.

    Args:
        base64_string (str): Base64-encoded mask.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Single-channel mask array.
    """
    mask_data = base64.b64decode(base64_string)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_array = np.array(mask_image)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Take first channel if RGB
    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)
    return np.array(mask_image)

def create_masks(results, width, height):
    """
    Combine multiple class masks into a single segmentation mask.

    Args:
        results (list): List of dictionaries with 'label' and 'mask' keys.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Combined segmentation mask with class indices.
    """
    combined_mask = np.zeros((height, width), dtype=np.uint8)  # Initialize with Background (0)

    # Process non-Background masks first
    for result in results:
        label = result['label']
        class_id = CLASS_MAPPING.get(label, 0)
        if class_id == 0:  # Skip Background
            continue
        mask_array = decode_base64_mask(result['mask'], width, height)
        combined_mask[mask_array > 0] = class_id

    # Process Background last to ensure it doesn't overwrite other classes unnecessarily
    # (Though the model usually provides non-overlapping masks for distinct classes other than background)
    for result in results:
        if result['label'] == 'Background':
            mask_array = decode_base64_mask(result['mask'], width, height)
            # Apply background only where no other class has been assigned yet
            # This logic might need adjustment based on how the model defines 'Background'
            # For this model, it seems safer to just let non-background overwrite it first.
            # A simple application like this should be fine: if Background mask says pixel is BG, set it to 0.
            # However, a more robust way might be to only set to background if combined_mask is still 0 (initial value)
            combined_mask[mask_array > 0] = 0 # Class ID for Background is 0

    return combined_mask



# --- Segmentation en local (aucune API, aucun token) ---
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_NAME = "sayeed99/segformer_b3_clothes"
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).eval()


def segment_image_local(img_path):
    """Retourne un masque (H, W) d'indices de classes, aligné sur CLASS_MAPPING."""
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, 18, h/4, w/4)
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    return upsampled.argmax(dim=1)[0].numpy().astype(np.uint8)


def segment_images_batch(list_of_image_paths):
    masks = []
    for path in tqdm(list_of_image_paths):
        try:
            masks.append(segment_image_local(path))
        except Exception as e:
            print(f"Erreur sur {path} : {e}")
            masks.append(None)
    return masks


def mask_to_api_json(mask):
    """Convertit un masque d'indices en liste [{'label', 'score', 'mask'(PNG base64)}], comme l'API HF."""
    id_to_label = {v: k for k, v in CLASS_MAPPING.items()}
    results = []
    for class_id in np.unique(mask):
        binary = ((mask == class_id) * 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(binary).save(buffer, format="PNG")
        results.append({
            "label": id_to_label.get(int(class_id), str(class_id)),
            "score": None,
            "mask": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        })
    return results


def save_segmentation_json(image_paths, masks, out_dir="./assets/Mask"):
    os.makedirs(out_dir, exist_ok=True)
    for path, mask in zip(image_paths, masks):
        if mask is None:
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(mask_to_api_json(mask), f)
        print(f"JSON sauvegardé : {out_path}")


def display_segmented_images_batch(original_image_paths, segmentation_masks):
    for path, mask in zip(original_image_paths, segmentation_masks):
        if mask is None:
            continue
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Image Originale")
        plt.imshow(Image.open(path))
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Masque de Segmentation")
        plt.imshow(mask, cmap="jet", vmin=0, vmax=17)
        plt.axis("off")
        plt.show()


if image_paths:
    print(f"\nTraitement de {len(image_paths)} image(s)...")
    batch_seg_results = segment_images_batch(image_paths)
    save_segmentation_json(image_paths, batch_seg_results)
    display_segmented_images_batch(image_paths, batch_seg_results)
else:
    print("Aucune image à traiter. Vérifiez 'image_dir' et 'max_images'.")
