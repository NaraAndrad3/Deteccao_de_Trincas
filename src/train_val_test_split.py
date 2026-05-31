from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def get_images(image_dir: Path):
    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(image_dir.glob(f"*{ext}"))
        images.extend(image_dir.glob(f"*{ext.upper()}"))

    return sorted(images)


def create_dirs(output_dir: Path):
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_pairs(image_paths, source_label_dir: Path, output_dir: Path, split: str):
    copied = 0
    skipped = 0

    for image_path in image_paths:
        label_path = source_label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            skipped += 1
            continue

        dst_image = output_dir / "images" / split / image_path.name
        dst_label = output_dir / "labels" / split / label_path.name

        shutil.copy2(image_path, dst_image)
        shutil.copy2(label_path, dst_label)

        copied += 1

    return copied, skipped


def split_dataset(
    dataset_dir="dataset",
    output_dir="dataset_yolo",
    train_size=0.70,
    val_size=0.20,
    test_size=0.10,
    random_state=42
):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"

    images = get_images(image_dir)

    valid_images = [
        img for img in images
        if (label_dir / f"{img.stem}.txt").exists()
    ]

    print("=" * 60)
    print("  DIVISÃO DO DATASET")
    print("=" * 60)
    print(f"Total de imagens encontradas : {len(images)}")
    print(f"Imagens com label válido     : {len(valid_images)}")

    create_dirs(output_dir)

    train_imgs, temp_imgs = train_test_split(valid_images,train_size=train_size,random_state=random_state,shuffle=True)

    val_ratio_adjusted = val_size / (val_size + test_size)

    val_imgs, test_imgs = train_test_split(temp_imgs,train_size=val_ratio_adjusted,random_state=random_state,shuffle=True)

    train_copied, train_skipped = copy_pairs(train_imgs,label_dir,output_dir,'train')

    val_copied, val_skipped = copy_pairs(val_imgs,label_dir,output_dir,'val')

    test_copied, test_skipped = copy_pairs(test_imgs,label_dir,output_dir,'test')

    print("\nArquivos copiados:")
    print(f"Train: {train_copied}")
    print(f"Val  : {val_copied}")
    print(f"Test : {test_copied}")

    print("\nArquivos ignorados:")
    print(f"Train: {train_skipped}")
    print(f"Val  : {val_skipped}")
    print(f"Test : {test_skipped}")

    print("\nDataset YOLO criado em:")
    print(output_dir)

    total = train_copied + val_copied + test_copied

    print("\nDistribuição final:")
    print(f"Train: {train_copied / total:.2%}")
    print(f"Val  : {val_copied / total:.2%}")
    print(f"Test : {test_copied / total:.2%}")


if __name__ == "__main__":
    split_dataset()