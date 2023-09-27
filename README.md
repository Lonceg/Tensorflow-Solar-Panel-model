# Tensorflow-Solar-Panel-model

Python code which uses TensorFlow library for training neural networks used in PV power estimation based on weather.

Prepared models use Reccurent Neural Network with following structures:
-LSTM
-Encoder-Decoder LSTM
-Bidirectional LSTm


Estimation is done based on 24 hours of predicted weather data in following many-to-many arrangement:

![image](https://github.com/Lonceg/Tensorflow-Solar-Panel-model/assets/92753179/bd5dddbb-902d-4429-9650-9d075b2f35f7)

This code uses NVIDIA libraries for accelerated training with the use of GPU. GPU used: RTX 2060.

Comparision of GPU vs CPU:

![image](https://github.com/Lonceg/Tensorflow-Solar-Panel-model/assets/92753179/68d0ef63-efad-4240-b5cf-106bfca70669)

Python version: 3.9.13 Tensorflow version: 2.10 CUDA Toolkit version: 11.2 cuDNN version: 8.11 Visual C++: 2017

![image](https://github.com/Lonceg/Tensorflow-Solar-Panel-model/assets/92753179/6900bffc-94b8-44df-ae5f-4139522fc8c8)

Downloading NVIDIA files requires an account (for one of these) cuDNN available under this link: https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64 CUDA available under this link: https://developer.nvidia.com/rdp/cudnn-archive

Running cuDNNN requires correct C++ compiler, all version available here: https://quasar.ugent.be/files/doc/cuda-msvc-compatibility.html

Code prepares decomposition of the input data as well as a heat correlation map of the features. Also all of the metrics are plotted.

![image](https://github.com/Lonceg/Tensorflow-Solar-Panel-model/assets/92753179/01006388-e7ff-4861-a23f-4470cbaafac7)

