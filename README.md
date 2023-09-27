# Tensorflow-Solar-Panel-model

My older Python code which uses TensorFlow library for training neural networks used in PV power estimation based on weather forecasts.
Due to non availability of 1 day ahead weather forecasts which would include solar data, actual weather data has been used.

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

![image](https://github.com/Lonceg/Tensorflow-Solar-Panel-model/assets/92753179/c49d0258-f0b8-4bbe-bf00-d307577d0675)

Model manages to perform some accurate predictions on test dataset

![image](https://github.com/Lonceg/Tensorflow-Solar-Panel-model/assets/92753179/8f9a4604-f377-47a6-9443-3ed67da07429)

Model also sometimes stumbles:

![image](https://github.com/Lonceg/Tensorflow-Solar-Panel-model/assets/92753179/866c886d-ee54-4a1b-ab7f-4cd0505d3569)

Ideas of improvements would include, training on a larger normalized dataset (using properly normalized data from several locations and PV installations assuming similar setups and efficiency),
training models for specific seasons or weather types (spring, summer, autumn, winter or sunny, cloudy, rainy days).
