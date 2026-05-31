from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def find_image_by_stem(image_dir: Path, stem: str):
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

        candidate = image_dir / f"{stem}{ext.upper()}"
        if candidate.exists():
            return candidate

    return None


def read_yolo_segmentation_label(label_path: Path, width: int, height: int):
    polygons = []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        values = line.split()

        class_id = int(values[0])
        coords = list(map(float, values[1:]))

        points = []

        for i in range(0, len(coords), 2):
            x = int(coords[i] * width)
            y = int(coords[i + 1] * height)
            points.append([x, y])

        polygons.append({
            "class_id": class_id,
            "points": np.array(points, dtype=np.int32)
        })

    return polygons


def draw_polygons(image, polygons):
    overlay = image.copy()
    output = image.copy()

    for idx, poly in enumerate(polygons, start=1):
        points = poly["points"]

        cv2.fillPoly(
            overlay,
            [points],
            color=(0, 255, 0)
        )

        cv2.polylines(
            output,
            [points],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )

        M = cv2.moments(points)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.putText(
                output,
                f"crack {idx}",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    alpha = 0.25

    output = cv2.addWeighted(
        overlay,
        alpha,
        output,
        1 - alpha,
        0
    )

    return output


def visualize_labels(
    dataset_dir="dataset",
    output_dir="results/label_samples",
    max_samples=20
):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"

    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(label_dir.glob("*.txt"))

    print("=" * 60)
    print("  VISUALIZAÇÃO DOS RÓTULOS")
    print("=" * 60)

    count = 0

    for label_path in label_files:
        image_path = find_image_by_stem(
            image_dir,
            label_path.stem
        )

        if image_path is None:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        polygons = read_yolo_segmentation_label(
            label_path,
            width,
            height
        )

        annotated = draw_polygons(
            image,
            polygons
        )

        output_path = output_dir / f"{label_path.stem}_label.jpg"

        cv2.imwrite(
            str(output_path),
            annotated
        )

        print(f"Salvo: {output_path}")

        count += 1

        if count >= max_samples:
            break

    print("\nConcluído.")
    print(f"{count} amostras salvas em {output_dir}")


if __name__ == "__main__":
    visualize_labels()