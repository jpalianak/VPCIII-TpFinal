import streamlit as st
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import warnings
from src.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger()

def run_inference(image_path):
    try:
        logger.info(f"Inferir imagen: {image_path}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = ViTForImageClassification.from_pretrained(
            "./outputs/checkpoints/final_model").to(device)
        processor = ViTImageProcessor.from_pretrained(
            "./outputs/checkpoints/final_model")

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        logger.info(f"Predicción obtenida: {prediction}")
        return prediction

    except Exception as e:
        logger.exception(f"Error durante la inferencia: {e}")
        print(f"Error durante la inferencia: {e}")
        return None


id2label = {
    0: "Crack",
    1: "Dead_Knot",
    2: "Knot_missing",
    3: "Live_Knot",
    4: "Marrow",
    5: "Quartzity",
    6: "knot_with_crack",
    7: "resin",
    8: "normal"
}


def run_app():
    st.title("Clasificador de defectos en imágenes de madera")

    uploaded_file = st.file_uploader(
        "Subí una imagen para clasificar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Guardar temporalmente para pasar path a run_inference
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.image(uploaded_file, caption="Imagen cargada",
                     use_container_width=True)

            # Ejecutar inferencia
            label_idx = run_inference("temp_image.jpg")
            st.write(f"Predicción: **{id2label[label_idx]}**")

        except Exception as e:
            logger.exception(f"Error en la app Streamlit: {e}")
            st.error(f"Error durante la clasificación: {e}")


if __name__ == "__main__":
    run_app()
