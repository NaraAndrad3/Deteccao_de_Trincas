from pathlib import Path
import cv2
import pandas as pd


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def get_images(image_dir: Path):
    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(image_dir.glob(f"*{ext}"))
        images.extend(image_dir.glob(f"*{ext.upper()}"))

    return sorted(images)


def inspect_dataset(dataset_dir="dataset"):
    dataset_dir = Path(dataset_dir)

    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"

    images = get_images(image_dir)
    labels = sorted(label_dir.glob("*.txt"))

    rows = []

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"

        img = cv2.imread(str(image_path))

        if img is None:
            width, height = None, None
        else:
            height, width = img.shape[:2]

        has_label = label_path.exists()

        n_objects = 0
        valid_segmentation = True

        if has_label:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            n_objects = len(lines)

            for line in lines:
                parts = line.split()

                # formato esperado:
                # class x1 y1 x2 y2 ... xn yn
                if len(parts) < 7:
                    valid_segmentation = False

                # precisa ter classe + número par de coordenadas
                if (len(parts) - 1) % 2 != 0:
                    valid_segmentation = False

        rows.append({
            "image": image_path.name,
            "has_label": has_label,
            "width": width,
            "height": height,
            "n_objects": n_objects,
            "valid_segmentation": valid_segmentation,
        })

    df = pd.DataFrame(rows)

    print("\n===== DATASET INSPECTOR =====")
    print(f"Total de imagens: {len(images)}")
    print(f"Total de labels : {len(labels)}")
    print(f"Imagens com label: {df['has_label'].sum()}")
    print(f"Imagens sem label: {(~df['has_label']).sum()}")
    print(f"Total de objetos anotados: {df['n_objects'].sum()}")

    print("\nDistribuição de objetos por imagem:")
    print(df["n_objects"].describe())

    print("\nLabels inválidos:")
    print(df[df["valid_segmentation"] == False])

    output_dir = Path("results/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "dataset_inspection.csv", index=False)

    print("\nRelatório salvo em:")
    print(output_dir / "dataset_inspection.csv")

    return df


if __name__ == "__main__":
    inspect_dataset()