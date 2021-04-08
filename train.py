
import flash
import os
from argparse import ArgumentParser
from flash.core.data import download_data
from flash.vision import ObjectDetectionData, ObjectDetector


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--backbone', type=str, default="resnet50")
    parser.add_argument('--download', type=bool, default=True)
    parser.add_argument('--train_folder', type=str, default=os.path.join(os.getcwd(),
                        "data/coco128/images/train2017/"))
    parser.add_argument('--train_ann_file', type=str, default=os.path.join(os.getcwd(),
                        "data/coco128/annotations/instances_train2017.json"))
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--gpus', type=int, default=None)
    args = parser.parse_args()

    # 1. Download the data
    if args.download:
        # Dataset Credit: https://www.kaggle.com/ultralytics/coco128
        download_data("https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip", 
                      os.path.join(os.getcwd(), "data/"))

    # 2. Load the Data
    datamodule = ObjectDetectionData.from_coco(
        train_folder=args.train_folder,
        train_ann_file=args.train_ann_file,
        batch_size=2
    )

    # 3. Build the model
    model = ObjectDetector(num_classes=datamodule.num_classes)

    # 4. Create the trainer
    trainer = flash.Trainer(max_epochs=args.max_epochs, gpus=args.gpus)

    # 5. Finetune the model
    trainer.finetune(model, datamodule)

    # 6. Save it!
    trainer.save_checkpoint("object_detection_model.pt")