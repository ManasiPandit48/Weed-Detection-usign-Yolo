# Weed Detection using YOLOv8

This project utilizes the YOLOv8 model for weed detection. The workflow includes data acquisition from Roboflow, model training using Ultralytics YOLOv8, and making predictions on new images.

## Technologies Used

- **YOLOv8:** YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system. YOLOv8 is the latest version, offering improvements in speed and accuracy.
- **Roboflow:** A platform for managing and preprocessing datasets for computer vision projects. It provides tools for dataset labeling, augmentation, and export.
- **Python:** The primary programming language used for model training and prediction.
- **Jupyter Notebook:** An interactive computing environment for writing and running code.

## Dataset

The dataset used for this project is the "Weed Detection ISA" dataset, which is hosted on Roboflow. The dataset consists of images labeled for various types of weeds, facilitating the training of a robust weed detection model.

### Creating and Downloading the Dataset

First, create and label your dataset on Roboflow. Then, download the dataset using the Roboflow API. The following code snippet shows how to download the dataset:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("object-detection-dt-wzpc6").project("weed-detection-isa")
dataset = project.version(1).download("yolov8")
```

Replace `YOUR_API_KEY` with your actual Roboflow API key.

## Execution Process

### 1. Installing Required Packages

First, install the necessary packages:

```bash
%pip install ultralytics
%pip install roboflow
```

### 2. Running Ultralytics Checks

```python
import ultralytics
ultralytics.checks()
```

### 3. Training the Model

Train the YOLOv8 model using the downloaded dataset:

```bash
!yolo train model=yolov8n.pt data=/path/to/your/data.yaml epochs=100 imgsz=640
```

### 4. Fine-tuning the Model

Subsequent training can be performed to fine-tune the model using weights from previous training sessions:

```bash
!yolo train model=/path/to/your/previous/weights/best.pt data=/path/to/your/data.yaml epochs=100 imgsz=640
```

### 5. Making Predictions

After training, use the model to make predictions on new images:

```bash
!yolo predict model=/path/to/your/final/weights/best.pt source=/path/to/your/image.jpg conf=0.5
```

### Complete Training and Prediction Workflow

Here is the complete workflow including multiple rounds of training and making predictions:

```bash
# Initial training
!yolo train model=yolov8n.pt data=/content/Weed-Detection-ISA-1/data.yaml epochs=100 imgsz=640

# Fine-tuning with subsequent rounds
!yolo train model=/content/runs/detect/train2/weights/best.pt data=/content/Weed-Detection-ISA-1/data.yaml epochs=100 imgsz=640
!yolo train model=/content/runs/detect/train3/weights/last.pt data=/content/Weed-Detection-ISA-1/data.yaml epochs=100 imgsz=640
!yolo train model=/content/runs/detect/train4/weights/best.pt data=/content/Weed-Detection-ISA-1/data.yaml epochs=100 imgsz=640
!yolo train model=/content/runs/detect/train5/weights/best.pt data=/content/Weed-Detection-ISA-1/data.yaml epochs=100 imgsz=640
!yolo train model=/content/runs/detect/train6/weights/best.pt data=/content/Weed-Detection-ISA-1/data.yaml epochs=100 imgsz=640
!yolo train model=/content/runs/detect/train7/weights/best.pt data=/content/Weed-Detection-ISA-1/data.yaml epochs=100 imgsz=640

# Making predictions
!yolo predict model=/content/runs/detect/train8/weights/best.pt source=/content/IMG_20220910_134408.jpg conf=0.5
!yolo predict model=/content/runs/detect/train5/weights/best.pt source=/content/IMG_20220910_134408.jpg conf=0.5
```

## Conclusion

This project demonstrates the end-to-end process of using YOLOv8 for weed detection. By leveraging Roboflow for dataset management and YOLOv8 for model training, we achieve an efficient workflow for detecting weeds in images. This solution can be extended and applied to various object detection tasks with suitable datasets.
