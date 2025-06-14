import numpy as np
from sklearn.metrics import accuracy_score
from datasets import load_dataset, DatasetDict
import os
from transformers import ViTForImageClassification, ViTImageProcessor

# Etiquetas
label2id = {
    "Live_Knot": 0,
    "Dead_Knot": 1,
    "resin": 2,
    "knot_with_crack": 3,
    "Crack": 4,
    "Marrow": 5,
    "Quartzity": 6,
    "Knot_missing": 7,
    "Blue_Stain": 8,
    "overgrown": 9,
    "Normal": 10
}


def transform_example(example, processor):
    inputs = processor(images=example['image'], return_tensors="pt")

    if len(example['objects']) == 0:
        label_str = "Normal"
    else:
        # Si hay varias etiquetas, nos quedamos con la primera para clasificar
        label_str = example['objects'][0]['label']

    label_id = label2id[label_str]

    return {
        'pixel_values': inputs['pixel_values'].squeeze(0),
        'labels': label_id
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def get_or_prepare_dataset(data_dir="data/wood_surface_defects_split"):
    if os.path.exists(data_dir):
        print(f"Cargando dataset desde disco en {data_dir}...")
        dataset = DatasetDict.load_from_disk(data_dir)
        # Reducir el dataset para probar y entrenar mas rapido
        dataset['train'] = dataset['train'].select(range(300))
        dataset['validation'] = dataset['validation'].select(range(30))
        dataset['test'] = dataset['test'].select(range(30))
    else:
        print("Descargando dataset de Hugging Face y creando splits...")
        full_dataset = load_dataset("iluvvatar/wood_surface_defects")["train"]

        # Dividir 80% train, 10% val, 10% test
        train_val = full_dataset.train_test_split(test_size=0.2, seed=42)
        val_test = train_val['test'].train_test_split(test_size=0.5, seed=42)

        dataset = DatasetDict({
            "train": train_val['train'],
            "validation": val_test['train'],
            "test": val_test['test']
        })

        print(f"Guardando dataset dividido en {data_dir} para uso futuro...")
        dataset.save_to_disk(data_dir)

    return dataset


def get_or_download_model(model_dir="./models/vit", num_labels=9):
    if os.path.exists(model_dir) and os.path.isfile(os.path.join(model_dir, "config.json")):
        print(f"Cargando modelo desde disco en {model_dir}...")
        model = ViTForImageClassification.from_pretrained(
            model_dir, num_labels=num_labels, ignore_mismatched_sizes=True)
    else:
        print(
            f"Descargando modelo desde Hugging Face y guardando en {model_dir}...")
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_labels,
            cache_dir=model_dir,
            ignore_mismatched_sizes=True
        )
        # Guardar una copia explícitamente (por si el cache_dir no guarda todo como modelo final)
        model.save_pretrained(model_dir)

    return model


def get_or_download_processor(processor_dir="./models/vit"):
    if os.path.exists(processor_dir) and os.path.isfile(os.path.join(processor_dir, "preprocessor_config.json")):
        print(f"Cargando processor desde disco en {processor_dir}...")
        processor = ViTImageProcessor.from_pretrained(processor_dir)
    else:
        print(
            f"Descargando processor desde Hugging Face y guardando en {processor_dir}...")
        processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224",
            cache_dir=processor_dir
        )
        processor.save_pretrained(processor_dir)

    return processor
