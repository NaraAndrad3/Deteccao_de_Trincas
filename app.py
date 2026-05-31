from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "runs" / "segment" / "runs" / "crack_yolov8n_seg_10ep" / "weights" / "best.pt"

CONF_THRESHOLD = 0.37


st.set_page_config(
    page_title="Detector de Trincas",
    layout="wide"
)


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


st.title(" >> Sistema Inteligente de Detecção de Trincas e Fissuras")

st.markdown(
    """
Aplicação web desenvolvida para detectar e localizar **trincas e fissuras**
em superfícies de paredes, concreto e pavimentos utilizando **YOLOv8n-seg**.

O sistema realiza segmentação das regiões danificadas, permitindo visualizar
a posição das falhas diretamente sobre a imagem.
"""
)

st.divider()


if not MODEL_PATH.exists():
    st.error(f"Modelo não encontrado em: `{MODEL_PATH}`")
    st.stop()


model = load_model()

uploaded_file = st.file_uploader(
    "Envie uma imagem para análise",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)

    with st.spinner("Processando imagem..."):
        results = model.predict(
            source=image_np,
            conf=CONF_THRESHOLD,
            verbose=False
        )

    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    n_detections = 0
    avg_conf = 0.0

    if result.boxes is not None and len(result.boxes) > 0:
        n_detections = len(result.boxes)
        avg_conf = float(result.boxes.conf.mean())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagem Original")
        st.image(image_pil, use_container_width=True)

    with col2:
        st.subheader("Resultado da Detecção")
        st.image(annotated_rgb, use_container_width=True)

    st.divider()

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Trincas detectadas", n_detections)

    with m2:
        st.metric("Confiança média", f"{avg_conf:.2f}")

    with m3:
        st.metric("Threshold utilizado", CONF_THRESHOLD)

    with st.expander("Como interpretar o resultado?"):
        st.markdown(
            """
- As regiões destacadas indicam áreas onde o modelo identificou trincas ou fissuras.
- A máscara colorida representa a segmentação da falha.
- A confiança indica o grau de certeza do modelo para cada detecção.
- O threshold de confiança foi definido com base nas curvas F1 obtidas na validação.
"""
        )

else:
    st.info("Envie uma imagem para iniciar a detecção.")