# Weed Detection using YOLOv8

This project leverages YOLOv8, a cutting-edge object detection model, to identify weeds in images. The workflow includes:

1. **Data Acquisition:** Using the "Weed Detection ISA" dataset from Roboflow, which provides labeled images of various weeds.
2. **Model Training:** Utilizing Ultralytics YOLOv8 to train the model on the acquired dataset.
3. **Predictions:** Making predictions on new images to detect weeds with the trained model.

## Technologies Used

- **YOLOv8:** For real-time object detection.
- **Roboflow:** For dataset management and preprocessing.
- **Python and Jupyter Notebook:** For coding and execution.

## Steps

1. **Install Required Packages:**
   ```bash
   %pip install ultralytics roboflow
   ```

2. **Download Dataset:**
   ```python
   from roboflow import Roboflow

   rf = Roboflow(api_key="YOUR_API_KEY")
   project = rf.workspace("object-detection-dt-wzpc6").project("weed-detection-isa")
   dataset = project.version(1).download("yolov8")
   ```

3. **Train the Model:**
   ```bash
   !yolo train model=yolov8n.pt data=/path/to/your/data.yaml epochs=100 imgsz=640
   ```

4. **Fine-tune the Model:**
   ```bash
   !yolo train model=/path/to/your/previous/weights/best.pt data=/path/to/your/data.yaml epochs=100 imgsz=640
   ```

5. **Make Predictions:**
   ```bash
   !yolo predict model=/path/to/your/final/weights/best.pt source=/path/to/your/image.jpg conf=0.5
   ```

## Conclusion

The project successfully demonstrates the application of YOLOv8 for weed detection, providing a comprehensive workflow from dataset acquisition to making predictions on new images.
