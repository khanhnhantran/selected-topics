import glob
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

logger = logging.getLogger(__name__)

HW2_CLASS_NAMES = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")


def register_dataset(
    train_json,
    train_images_dir,
    valid_json=None,
    valid_images_dir=None,
    dataset_name_train="train",
    dataset_name_valid="valid",
    class_names=HW2_CLASS_NAMES,
):
    assert os.path.isfile(train_json), f"train.json not found: {train_json}"
    assert os.path.isdir(train_images_dir), f"train images dir not found: {train_images_dir}"
    register_coco_instances(dataset_name_train, {}, train_json, train_images_dir)
    MetadataCatalog.get(dataset_name_train).set(thing_classes=list(class_names))
    logger.info("Registered training dataset '%s'", dataset_name_train)

    if valid_json and valid_images_dir:
        assert os.path.isfile(valid_json), f"valid.json not found: {valid_json}"
        assert os.path.isdir(valid_images_dir), f"valid images dir not found: {valid_images_dir}"
        register_coco_instances(dataset_name_valid, {}, valid_json, valid_images_dir)
        MetadataCatalog.get(dataset_name_valid).set(thing_classes=list(class_names))
        logger.info("Registered validation dataset '%s'", dataset_name_valid)


def register_test_split(test_images_dir):
    name = "hw2_test"
    if name in DatasetCatalog.list():
        return name

    image_paths = sorted(
        glob.glob(os.path.join(test_images_dir, "*.png"))
        + glob.glob(os.path.join(test_images_dir, "*.jpg"))
    )

    def _get_dicts():
        import struct

        dicts = []
        for i, img_path in enumerate(image_paths):
            with open(img_path, "rb") as f:
                header = f.read(24)
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", header[16:24])
            else:
                from PIL import Image

                with Image.open(img_path) as im:
                    w, h = im.size
            stem = os.path.splitext(os.path.basename(img_path))[0]
            dicts.append(
                {
                    "file_name": img_path,
                    "image_id": int(stem) if stem.isdigit() else i + 1,
                    "height": h,
                    "width": w,
                    "annotations": [],
                }
            )
        return dicts

    DatasetCatalog.register(name, _get_dicts)
    MetadataCatalog.get(name).set(thing_classes=list(HW2_CLASS_NAMES))
    logger.info("Registered test split '%s' (%d images).", name, len(image_paths))
    return name
