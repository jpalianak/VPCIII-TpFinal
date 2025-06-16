import argparse
from src.logger import get_logger
from src.inference import run_inference
from src.evaluate import run_evaluation
from src.train import run_training
from src.utils import run_jupyter_server
import warnings

logger = get_logger()

warnings.filterwarnings("ignore")


def main():

    logger.info("Inicio del programa")

    parser = argparse.ArgumentParser(
        description="Proyecto Vision por Computadora III - ViT")
    parser.add_argument(
        "task", choices=["analysis","train", "evaluate", "inference"], help="Tarea a ejecutar")
    parser.add_argument(
        "--image", type=str, help="Ruta a la imagen para inferencia (sólo para 'inference')")
    args = parser.parse_args()

    logger.info(f"Tarea seleccionada: {args.task}")

    try:
        if args.task == "analysis":
            run_jupyter_server()
        if args.task == "train":
            logger.info("Ejecutando entrenamiento")
            run_training()
        elif args.task == "evaluate":
            logger.info("Ejecutando evaluacion")
            run_evaluation()
        elif args.task == "inference":
            if args.image:
                logger.error("No se proporcionó ruta a imagen para inferencia")
                print(
                    "Debe proporcionar una ruta a la imagen con --image para ejecutar inferencia.")
            else:
                logger.info(f"Ejecutando inferencia con imagen: {args.image}")
                run_inference(args.image)

    except Exception as e:
        logger.exception(
            f"Error en la ejecución de la tarea '{args.task}': {e}")
        print(f"Error en la ejecución de la tarea '{args.task}': {e}")

    logger.info("Fin del programa")

    # Asegurar flush y cierre de logs
    for handler in logger.handlers:
        handler.flush()
        handler.close()


if __name__ == "__main__":
    main()
