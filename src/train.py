import torch
from transformers import TrainingArguments, Trainer
from src.utils import transform_example, compute_metrics, get_or_prepare_dataset, get_or_download_model, get_or_download_processor, WrappedViTModel
import warnings
import mlflow
import numpy as np
from src.logger import get_logger
import os

warnings.filterwarnings("ignore")


def run_training():
    logger = get_logger()
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(
            f"Inicio del entrenamiento en dispositivo: {device.upper()} ({device})")

        dataset = get_or_prepare_dataset()
        logger.info("Dataset cargado/preparado")

        processor = get_or_download_processor(processor_dir="./models/vit")
        logger.info("Processor cargado/preparado")

        model = get_or_download_model(model_dir="./models/vit", num_labels=11)
        logger.info("Modelo cargado/preparado")

        # Preprocesar datasets
        dataset = dataset.map(lambda x: transform_example(
            x, processor), batched=False, num_proc=4, remove_columns=dataset["train"].column_names)
        dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
        logger.info("Dataset preprocesado")

        training_args = TrainingArguments(
            output_dir="./outputs/checkpoints",
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            logging_first_step=True,
            learning_rate=3e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=50,
            weight_decay=0.01,
            logging_dir="./outputs/logs",
            fp16=True if device == 'cuda' else False,
            dataloader_pin_memory=True if device == 'cuda' else False,
            report_to="mlflow",
            optim="adamw_torch",
            save_strategy="epoch",
            save_total_limit=2
        )
        logger.info("Configuración de entrenamiento creada")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            compute_metrics=compute_metrics
        )

        # Iniciar experimento en MLflow
        mlflow.set_experiment("vit-wood-defects")

        with mlflow.start_run():
            # Log hiperparámetros
            mlflow.log_params({
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "epochs": training_args.num_train_epochs,
                "weight_decay": training_args.weight_decay
            })

            logger.info("Entrenando el modelo...")

            # Buscar último checkpoint (si existe)
            last_checkpoint = None
            if os.path.isdir(training_args.output_dir):
                checkpoints = [
                    os.path.join(training_args.output_dir, d)
                    for d in os.listdir(training_args.output_dir)
                    if d.startswith("checkpoint-")
                ]
                if checkpoints:
                    last_checkpoint = max(checkpoints, key=os.path.getmtime)
                    logger.info(
                        f"Reanudando desde el checkpoint: {last_checkpoint}")
                else:
                    logger.info(
                        "No se encontró checkpoint previo. Entrenando desde cero.")

            # Entrenamiento (continúa desde el último checkpoint si lo hay)
            trainer.train(resume_from_checkpoint=last_checkpoint)

            # Evaluar en validación
            eval_metrics = trainer.evaluate()
            mlflow.log_metrics(eval_metrics)

            # Guardar modelo y processor como artefactos
            model_path = "./outputs/checkpoints/final_model"
            trainer.save_model(model_path)
            processor.save_pretrained(model_path)

            wrapped_model = WrappedViTModel(model)
            example_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
            mlflow.pytorch.log_model(
                wrapped_model, name="model", input_example=example_input)
            mlflow.log_artifacts(model_path, artifact_path="processor")

            logger.info("Entrenamiento finalizado correctamente")

    except Exception as e:
        logger.exception(f"Error durante el entrenamiento: {e}")
        print(f"Error durante el entrenamiento: {e}")

    for handler in logger.handlers:
        handler.flush()
        handler.close()


if __name__ == "__main__":
    run_training()
