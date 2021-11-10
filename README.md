# TF Object Detection

The version of TensorFlow has some requirements about the versions of: Python, Compiler, Build Tools, cuDNN and CUDA

[https://www.tensorflow.org/install/source](https://www.tensorflow.org/install/source)

![Captura de pantalla 2021-10-19 a las 18.59.12.png](TF%20Object%20Detection%20f9b248b256de4d7ea7eb3f948fa6955a/Captura_de_pantalla_2021-10-19_a_las_18.59.12.png)

![Captura de pantalla 2021-10-19 a las 18.59.06.png](TF%20Object%20Detection%20f9b248b256de4d7ea7eb3f948fa6955a/Captura_de_pantalla_2021-10-19_a_las_18.59.06.png)

# Object detection model

[models/tf2_detection_zoo.md at master · tensorflow/models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

Different models are shown in this format:

![Captura de pantalla 2021-10-21 a las 10.03.22.png](TF%20Object%20Detection%20f9b248b256de4d7ea7eb3f948fa6955a/Captura_de_pantalla_2021-10-21_a_las_10.03.22.png)

- mAP: Mean average precision (on the COCO dataset)

Since we want a model that is going to be ran in a smartphone or Raspberry → the faster the better. Then we're gonna be using *"SSD MobileNet V2 FPNLite 320x320"* . 

**Features of the models**

- The advantage of these models is that you don't have to do a hard pre-processing or post-processing.
- It uses some data augmentation techniques. The more post-processing it provides, the slower the model

The following model outputs a mask (ex the accurate shape of your face). Instead of just boxes. In exchange its speed is slow.

![Captura de pantalla 2021-10-21 a las 10.10.08.png](TF%20Object%20Detection%20f9b248b256de4d7ea7eb3f948fa6955a/Captura_de_pantalla_2021-10-21_a_las_10.10.08.png)

# Training

We are gonna be using a pre-trained model (`ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8`) and we're gonna train our own custom model, that we're calling it `my_ssd_mobnet`. 

Then we're defining some variables with the name of the models, URL, a TFRecord transformer script name (it does not exist yet) and a label map filename.

- TFRecord is a binary format to store data

- Label map filename →

![Captura de pantalla 2021-10-21 a las 17.59.34.png](TF%20Object%20Detection%20f9b248b256de4d7ea7eb3f948fa6955a/Captura_de_pantalla_2021-10-21_a_las_17.59.34.png)

# Tensorboard

Once we have trained the model, it automatically generates TensorBoard files inside `train` and `eval` folders:

![Untitled](TF%20Object%20Detection%20f9b248b256de4d7ea7eb3f948fa6955a/Untitled.png)

They should be located inside the model folder (in this case `my_ssd_mobnet`). In order to open them with TensorBoard:

```bash
(tfod) % cd Tensorflow/workspace/models/my_ssd_mobnet
(tfod) % cd train                
(tfod) % tensorboard --logdir=.  

NOTE: ....

Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.7.0 at **http://localhost:6006/** (Press CTRL+C to quit)
```

Then we should open the provide url: ***http://localhost:6006/***

![Untitled](TF%20Object%20Detection%20f9b248b256de4d7ea7eb3f948fa6955a/Untitled%201.png)

# Checkpoints

See that the latest checkpoint is the one called *ckpt-3.index.* Therefore, it is the most trained model.

![Untitled](TF%20Object%20Detection%20f9b248b256de4d7ea7eb3f948fa6955a/Untitled%202.png)

# Freezing the graph

The idea of this is exporting a model on a similar format as the pre-trained model

Exported model

![Untitled](TF%20Object%20Detection%20f9b248b256de4d7ea7eb3f948fa6955a/Untitled%203.png)

Pre-trained model

![Untitled](TF%20Object%20Detection%20f9b248b256de4d7ea7eb3f948fa6955a/Untitled%204.png)

### TFJS

This conversion exports a model.json file. It can be used inside web applications.

### TFLite

It is a file format typically used for mobile applications or machines that won't be using a full version of TFOD (like raspberry)

# Performance tuning

Strategies to be used to achieve a better performance of the model

1. Getting more labeled data
2. Changing the architecture. For that we may choose a different model. Specially those that theoretically provide more accuracy (heavier models):
    1. [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
3. Training for longer. If do not have GPU, use colab
