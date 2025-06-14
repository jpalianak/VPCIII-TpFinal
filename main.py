import argparse
from src import train, evaluate, inference


def main():
    parser = argparse.ArgumentParser(
        description="Proyecto Vision por Computadora III - ViT")
    parser.add_argument(
        "task", choices=["train", "evaluate", "inference"], help="Tarea a ejecutar")
    parser.add_argument(
        "--image", type=str, help="Ruta a la imagen para inferencia (sólo para 'inference')")

    args = parser.parse_args()

    if args.task == "train":
        train.run_training()
    elif args.task == "evaluate":
        evaluate.run_evaluation()
    elif args.task == "inference":
        if not args.image:
            print(
                "Debe proporcionar una ruta a la imagen con --image para ejecutar inferencia.")
        else:
            inference.run_inference(args.image)


if __name__ == "__main__":
    main()
