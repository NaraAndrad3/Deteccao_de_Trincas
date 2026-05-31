from pathlib import Path
from ultralytics import YOLO


MODEL_PATH = "runs/segment/runs/crack_yolov8n_seg_10ep/weights/best.pt"
SOURCE_DIR = "dataset_yolo/images/test"
OUTPUT_PROJECT = "results"
OUTPUT_NAME = "predictions"
CONF_THRESHOLD = 0.37


def main():
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {model_path}\n"
            "Verifique se o treinamento foi concluído e se o caminho está correto."
        )

    output_dir = Path(OUTPUT_PROJECT) / OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DETECÇÃO DE TRINCAS E FISSURAS")
    print("  YOLOv8n-seg")
    print("=" * 60)
    print(f"Modelo     : {MODEL_PATH}")
    print(f"Entrada    : {SOURCE_DIR}")
    print(f"Saída      : {output_dir}")
    print(f"Confiança  : {CONF_THRESHOLD}")
    print("=" * 60)

    model = YOLO(str(model_path))

    results = model.predict(
        source=SOURCE_DIR,
        conf=CONF_THRESHOLD,
        save=True,
        save_txt=True,
        save_conf=True,
        project=OUTPUT_PROJECT,
        name=OUTPUT_NAME,
        exist_ok=True
    )

    print("\nPredição concluída.")
    print(f"Resultados salvos em: {output_dir}")
    print(f"Total de imagens processadas: {len(results)}")


if __name__ == "__main__":
    main()