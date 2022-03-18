===========================================================================================================================
Lip Reading - Cross Audio-Visual Recognition using 3D Convolutional Neural Networks 
===========================================================================================================================

This repository contains the code developed by TensorFlow_ for the following paper:


| `3D Convolutional Neural Networks for Cross Audio-Visual Matching Recognition


.. _3D Convolutional Neural Networks for Cross Audio-Visual Matching Recognition: http://ieeexplore.ieee.org/document/8063416/
.. _TensorFlow: https://www.tensorflow.org/
.. _Official Project Page: https://codeocean.com/2017/07/14/3d-convolutional-neural-networks-for-audio-visual-recognition/code
.. _Amirsina Torfi: https://astorfi.github.io/
.. _Seyed Mehdi Iranmanesh: http://community.wvu.edu/~seiranmanesh/
.. _Nasser M. Nasrabadi: http://nassernasrabadi.wixsite.com/mysite


.. |im1| image:: 1.gif


.. |im2| image:: 2.gif


.. |im3| image:: 3.gif


|im1| |im2| |im3|

The input pipeline must be prepared by the users. This code is aimed to provide the implementation for **Coupled 3D Convolutional Neural Networks** for
audio-visual matching. **Lip-reading** can be a specific application for this work.


-----
DEMO
-----

~~~~~~~~~~~~~~~~~~~~~~~~
Training/Evaluation DEMO
~~~~~~~~~~~~~~~~~~~~~~~~

|training|

.. |training| image:: liptrackingdemo.png
    :target: https://asciinema.org/a/kXIDzZt1UzRioL1gDPzOy9VkZ

~~~~~~~~~~~~~~~~~
Lip Tracking DEMO
~~~~~~~~~~~~~~~~~

|liptrackingdemo|

.. |liptrackingdemo| image:: liptrackingdemo.png
    :target: https://asciinema.org/a/RiZtscEJscrjLUIhZKkoG3GVm
.. https://asciinema.org/a/m1r1OaoUXsEECNZKzpkfAXg7y

--------------
General View
--------------

*Audio-visual recognition* (AVR) has been considered as
a solution for speech recognition tasks when the audio is
corrupted, as well as a visual recognition method used
for speaker verification in multi-speaker scenarios. The approach of AVR systems is to leverage the extracted
information from one modality to improve the recognition ability of
the other modality by complementing the missing information.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Problem and the Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The essential problem is to find the correspondence between the audio and visual streams, which is the goal
of this work. **We proposed the utilization of a coupled 3D Convolutional Neural Network (CNN) architecture that can map
both modalities into a representation space to evaluate the correspondence of audio-visual streams using the learned
multimodal features**.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
How to leverage 3D Convolutional Neural Networks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The proposed architecture will incorporate both spatial and temporal information jointly to
effectively find the correlation between temporal information
for different modalities. By using a relatively small network architecture and much
smaller dataset, our proposed
method surpasses the performance of the existing similar
methods for audio-visual matching which use CNNs for
feature representation. We also demonstrate that effective
pair selection method can significantly increase the performance.


--------------------
Code Implementation
--------------------

The input pipeline must be provided by the user. The rest of the implementation consider the dataset
which contains the utterance-based extracted features.

~~~~~~~~~~~
Processing
~~~~~~~~~~~

In the visual section, the videos are post-processed to have an equal frame rate of 30 f/s. Then, face tracking and mouth area extraction are performed on the videos using the
dlib library. Finally, all mouth areas are resized to have the same size and concatenated to form the input feature
cube. The dataset does not contain any audio files. The audio files are extracted from
videos using FFmpeg framework. The processing pipeline is the below figure.

.. image:: processing.gif

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input Pipeline for this work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. .. image:: https://github.com/astorfi/3D-convolutional-speaker-recognition/blob/master/_images/Speech_GIF.gif
..     :target: https://github.com/astorfi/3D-convolutional-speaker-recognition/blob/master/_images/Speech_GIF.gif

The proposed architecture utilizes two non-identical ConvNets which uses a pair of speech and video
streams. The network input is a pair of features that represent lip movement and
speech features extracted from 0.3 second of a video clip. The main task is to determine if a
stream of audio corresponds with a lip motion clip within the desired stream duration. In the two next sub-sections,
we are going to explain the inputs for speech and visual streams.


**Speech Net**


On the time axis, the temporal features are non-overlapping
20ms windows which are used for the generation of spectrum features
that possess a local characteristic.
The input speech feature map, which is represented as an image cube,
corresponds to the spectrogram
as well as the first and second order derivatives of the
MFEC features. These three channels correspond to the image depth. Collectively from a 0.3 second
clip, 15 temporal feature sets (each
forms 40 MFEC features) can be derived which form a
speech feature cube. Each input feature map for a single audio stream has the dimensionality of 15 × 40 × 3.
This representation is depicted in the following figure:

.. image:: Speech_GIF.gif


**Visual Net**

The frame rate of each video clip used in this effort is 30 f/s.
Consequently, 9 successive image frames form the 0.3 second visual stream.
The input of the visual stream of the network is a cube of size 9x60x100,
where 9 is the number of frames that represent the temporal information. Each
channel is a 60x100 gray-scale image of mouth region.

.. image:: lip_motion.jpg



~~~~~~~~~~~~
Architecture
~~~~~~~~~~~~

The architecture is a **coupled 3D convolutional neural network** in which *two
different networks with different sets of weights must be trained*.
For the visual network, the lip motions spatial information alongside the temporal information are
incorporated jointly and will be fused for exploiting the temporal
correlation. For the audio network, the extracted energy features are
considered as a spatial dimension, and the stacked audio frames form the
temporal dimension. In the proposed 3D CNN architecture, the convolutional operations
are performed on successive temporal frames for both audio-visual streams.

.. image:: DNN-Coupled.png



--------
Results
--------

The below results demonstrate effects of the proposed method on the accuracy
and the speed of convergence.

.. |accuracy| image:: accuracy-bar-pairselection.png


.. |converge| image:: convergence-speed.png


|accuracy|

The best results, which is the right-most one, belongs to our proposed method.

|converge|

The effect of proposed **Online Pair Selection** method has been shown in the figure.


