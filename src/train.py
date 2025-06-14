import torch
from transformers import TrainingArguments, Trainer
from src.utils import transform_example, compute_metrics, get_or_prepare_dataset, get_or_download_model, get_or_download_processor


def run_training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = get_or_prepare_dataset()

    processor = get_or_download_processor(processor_dir="./models/vit")

    model = get_or_download_model(model_dir="./models/vit", num_labels=11)

    # Preprocesar datasets
    dataset = dataset.map(lambda x: transform_example(
        x, processor), batched=False, num_proc=4, remove_columns=dataset["train"].column_names)
    dataset.set_format(type='torch', columns=['pixel_values', 'labels'])

    training_args = TrainingArguments(
        output_dir="./outputs/checkpoints",
        evaluation_strategy="epoch",
        logging_strategy="epoch",            # <-- nuevo
        logging_first_step=True,             # <-- nuevo
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir="./outputs/logs",
        fp16=True if device == 'cuda' else False,
        dataloader_pin_memory=True if device == 'cuda' else False,
        report_to="none",
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    print("Entrenando el modelo...")
    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)
    trainer.save_model("./outputs/checkpoints/final_model")
    processor.save_pretrained("./outputs/checkpoints/final_model")


if __name__ == "__main__":
    run_training()
