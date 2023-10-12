
from google.colab import drive
drive.mount('/content/drive')

from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

import tensorflow as tf
assert tf.__version__.startswith('2')

import matplotlib.pyplot as plt
import numpy as np



train_data = DataLoader.from_folder(/content/drive/MyDrive/botany_items/train)
test_data = DataLoader.from_folder(/content/drive/MyDrive/botany_items/test)

"""2. Customize the TensorFlow model."""

model = image_classifier.create(train_data)

"""3. Evaluate the model."""

loss, accuracy = model.evaluate(test_data)

"""4.  Export to TensorFlow Lite model.
You could download it in the left sidebar same as the uploading part for your own use.
"""

model.export(export_dir='.')

"""5. Download the trained model by clicking on the folder icon on the left hand side. Right-click on "model.tflite" and select download. Or run the following code:"""

from google.colab import files
files.download('model.tflite')

"""After this simple 5 steps, we can now continue to the next step in the [codelab](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android-beta/#2).

For a more comprehensive guide to TFLite Model Maker, please refer to this [notebook](https://colab.sandbox.google.com/github/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/demo/image_classification.ipynb) and its [documentation](https://github.com/tensorflow/examples/tree/master/tensorflow_examples/lite/model_maker).
"""