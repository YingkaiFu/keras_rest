# How to run this code
1. Move tensorflow object detection api in this project. for this project, simply add nets, deployment and object_detection these three directory into this project.
2. Download a model and modify the path to your model
```python
def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    PATH_TO_CKPT = 'model/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    # PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
```
3. run the object_detection_rest.py
```python
python object_detection_rest.py
```
4. Use curl to post your image to your server and get the result.