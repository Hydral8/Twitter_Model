# Twitter_Model

This model was built with the tensorflow detection api to detect multiple twitter "following" boxes in twitter images. 
<br />
You can find the link to the Tensorflow Object Detection Api [here](https://github.com/tensorflow/models/tree/master/research/object_detection).

<hr />

## Model Usage Guide
1. First Clone the Repository and make sure all required files and ipynb notebooks exist and match the ones on the repository.
1. Download the pretrained ResNet 50 [model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) (for Transfer learning), unzip the tar file, and place the folder in your working directory.
1. Go through the ipynb notebooks and the configuration file in the model_dir folder, and update the paths so that they match the file structure in your directory. Any paths will have a comment `#configure path`.

### Model Training
* Use the `Twitter_Image_Set_Model.ipynb` file to run the training algorith musing tensorflow object detection api.
* Follow the code cells step by step in the file, and make sure to clone the tensorflow object detection directory.
* Once you are done make sure you save the optimized trained model(using the `optimize_graph function`, or the nonoptimized one.

### Model Output
* Use the `Display_Object_Detection_results.ipynb` file to display the results of your newly trained model.
* You can use the `draw_boxes` if you want to visually see the bounding boxes that are outputted by the model based on an input image or `get_boxes` if you only want the bounding box coordinate data.
