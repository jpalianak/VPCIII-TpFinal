import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from src.utils import transform_example, get_or_prepare_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
from src.logger import get_logger

warnings.filterwarnings("ignore")


def run_evaluation():
    logger = get_logger()
    try:
        logger.info("Inicio de la evaluación")

        # Cargamos el modelo entrenado y el proccesor
        model = ViTForImageClassification.from_pretrained(
            "./outputs/checkpoints/final_model")
        processor = ViTImageProcessor.from_pretrained(
            "./outputs/checkpoints/final_model")
        logger.info("Modelo y processor cargados")

        # Cargamos el dataset
        dataset = get_or_prepare_dataset()
        logger.info("Dataset cargado/preparado")

        # Preprocesar datasets
        dataset = dataset.map(lambda x: transform_example(
            x, processor), batched=False, num_proc=4, remove_columns=dataset["train"].column_names)
        dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
        logger.info("Dataset preprocesado para evaluación")

        # Nos quedamos solo con el test
        test_dataset = dataset["test"]

        dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
        logger.info("Dataloader creado para dataset de test")

        model.eval()
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Modelo configurado en dispositivo: {model.device}")

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(model.device)
                labels = batch["labels"].to(model.device)
                outputs = model(pixel_values=pixel_values)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        logger.info("Evaluación completada, generando matriz de confusión")
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.show()

    except Exception as e:
        logger.exception(f"Error durante la evaluación: {e}")
        print(f"Error durante la evaluación: {e}")


if __name__ == "__main__":
    run_evaluation()
