import numpy as np
from sklearn.metrics import accuracy_score
from datasets import load_dataset, DatasetDict
import os
from transformers import ViTForImageClassification, ViTImageProcessor
import torch.nn as nn
from src.logger import get_logger
import subprocess
from collections import Counter
import evaluate




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


#cargar métricas
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

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
    # return {"accuracy": acc}
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(predictions=preds, references=labels, average="weighted")["precision"],
        "recall": recall.compute(predictions=preds, references=labels, average="weighted")["recall"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

'''
def get_or_prepare_dataset(data_dir=f"{project_root}/data/wood_surface_defects_split"):
    logger = get_logger()
    if os.path.exists(data_dir):
        logger.info(f"Cargando dataset desde disco en {data_dir}...")
        dataset = DatasetDict.load_from_disk(data_dir)

        # Reducir el dataset para probar y entrenar mas rapido
        dataset['train'] = dataset['train'].select(range(700))
        dataset['validation'] = dataset['validation'].select(range(655))
        dataset['test'] = dataset['test'].select(range(655))
    else:
        logger.info("Descargando dataset de Hugging Face y creando splits...")
        full_dataset = load_dataset("iluvvatar/wood_surface_defects")["train"]

        # Dividir 80% train, 10% val, 10% test
        train_val = full_dataset.train_test_split(test_size=0.2, seed=42)
        val_test = train_val['test'].train_test_split(test_size=0.5, seed=42)

        dataset = DatasetDict({
            "train": train_val['train'],
            "validation": val_test['train'],
            "test": val_test['test']
        })

        logger.info(
            f"Guardando dataset dividido en {data_dir} para uso futuro...")
        dataset.save_to_disk(data_dir)

    return dataset
'''

def get_or_prepare_dataset(data_dir=f"{project_root}/data/wood_surface_defects_split", max_per_class=80):
    logger = get_logger()

    if os.path.exists(data_dir):
        logger.info(f"Cargando dataset desde disco en {data_dir}...")
        dataset = DatasetDict.load_from_disk(data_dir)
    else:
        logger.info(
            "Descargando dataset de Hugging Face y creando splits balanceados...")
        full_dataset = load_dataset("iluvvatar/wood_surface_defects")["train"]

        # Agrupar índices por clase usando la primera etiqueta de cada imagen
        class_indices = defaultdict(list)
        for idx, example in enumerate(full_dataset):
            if example['objects']:
                label = example['objects'][0]['label']
                class_indices[label].append(idx)

        # Filtrar clases que tengan al menos max_per_class imágenes
        filtered_class_indices = {
            label: indices for label, indices in class_indices.items()
            if len(indices) >= max_per_class
        }

        logger.info(
            f"Se conservaron {len(filtered_class_indices)} clases con al menos {max_per_class} imágenes")

        # Seleccionar hasta max_per_class imágenes por clase
        selected_indices = []
        for label, indices in filtered_class_indices.items():
            random.shuffle(indices)
            selected = indices[:max_per_class]
            selected_indices.extend(selected)
            logger.info(f"{label}: {len(selected)} imágenes seleccionadas")

        # Crear dataset balanceado
        balanced_dataset = full_dataset.select(selected_indices)

        # Dividir: 80% train, 10% val, 10% test
        train_val = balanced_dataset.train_test_split(test_size=0.2, seed=42)
        val_test = train_val["test"].train_test_split(test_size=0.5, seed=42)

        dataset = DatasetDict({
            "train": train_val["train"],
            "validation": val_test["train"],
            "test": val_test["test"]
        })

        logger.info(
            f"Guardando dataset dividido en {data_dir} para uso futuro...")
        dataset.save_to_disk(data_dir)

    return dataset

def get_or_download_model(model_dir="./models/vit", num_labels=9):
    logger = get_logger()
    if os.path.exists(model_dir) and os.path.isfile(os.path.join(model_dir, "config.json")):
        logger.info(f"Cargando modelo desde disco en {model_dir}...")
        model = ViTForImageClassification.from_pretrained(
            model_dir, num_labels=num_labels, ignore_mismatched_sizes=True)
    else:
        logger.info(
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
    logger = get_logger()
    if os.path.exists(processor_dir) and os.path.isfile(os.path.join(processor_dir, "preprocessor_config.json")):
        logger.info(f"Cargando processor desde disco en {processor_dir}...")
        processor = ViTImageProcessor.from_pretrained(processor_dir)
    else:
        logger.info(
            f"Descargando processor desde Hugging Face y guardando en {processor_dir}...")
        processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224",
            cache_dir=processor_dir
        )
        processor.save_pretrained(processor_dir)

    return processor


class WrappedViTModel(nn.Module):
    def __init__(self, model):
        super(WrappedViTModel, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits



def run_jupyter_server():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(project_root)
    subprocess.run([
        'jupyter', 
        'lab', 
        '--ip=0.0.0.0', 
        '--port=9999', 
        '--no-browser',
        f'--notebook-dir={project_root}'
    ])


def assign_main_label_for_eda(sample):
    if not sample["objects"]:
        label_str = "Normal"
    else:
        labels = [obj["label"] for obj in sample["objects"]]
        label_str = Counter(labels).most_common(1)[0][0] #escoge el mas frecuente
    return {"labels": label2id[label_str]}
