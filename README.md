# Introduction
Blender is a powerful, open-source 3D modeling and animation software that offers a wide variety of tools and applications. However, when it comes to navigating and manipulating objects in 3D space, users are limited to using 2D input from a computer mouse. Wouldn’t it be more intuitive to be able to use 3-dimensional input in this 3D environment? In this tutorial, you will learn how to use MediaPipe’s hand tracking feature, along with two different feature extraction methods, to map hand gestures to Blender commands, providing you an overall more immersive experience! You will use four different hand gestures to pan around Blender’s viewport, rotate the viewport, and pick up different objects in your scene. All you need is a computer, a webcam, and free-to-use software.  

# Setup
Before you can begin, you will need to download the Blender software from their website, if you do not already have it: https://www.blender.org/download/

Optionally, you may choose to run some of the provided code outside of Blender’s environment, in which you will need to make sure you have Python downloaded as well: https://www.python.org/downloads/

Next, you will need the following Python libraries:
- OpenCV-Python: https://pypi.org/project/opencv-python/
- MediaPipe: https://pypi.org/project/mediapipe/
- NumPy: https://numpy.org/install/
- SciPy: https://scipy.org/install/
- Joblib: https://joblib.readthedocs.io/en/latest/installing.html
- Pandas: https://pandas.pydata.org/docs/getting_started/install.html
- Scikit-learn: https://scikit-learn.org/stable/install.html

Make sure to install them using the Python version bundled with Blender and optionally, with your separate version of Python if you will be running some of the code elsewhere as well. To download these libraries for use within Blender, you need to find the python.exe file within Blender’s program files, which is typically located in the following folder:

…/Blender Foundation/Blender 4.4/4.4/python/bin

Open the command prompt as an administrator and download the previously mentioned libraries using Pip and this python.exe version. If you end up getting MediaPipe errors within Blender, try downloading an older version of MediaPipe instead.

Finally, download the provided code and files from this GitHub page: https://github.com/HannaMG/blender-hand-gesture-system.git
 
Now you are ready to implement your hand gesture system!

Note: Whenever you run any of the provided programs, make sure to run it within the provided folder! Many of these programs utilize the specific directory structure to retrieve data and files from other locations.

# Dataset
To be able to detect different hand gestures within Blender’s environment, you first need a dataset of hand gesture images to train the system with. For best results, you will need to create this dataset yourself using your computer’s webcam and the provided folder structure from the GitHub repository. 

Four different hand gestures are needed for this implementation. First, open your computer’s webcam or camera app and start taking pictures of your hand gestures using different angles, distances, lighting conditions, and locations on the screen. The more varied your hand gesture pictures are, the better your system will respond to these different gestures! While not strictly required, the following four hand gestures are recommended for this implementation:

|<img src="/Dataset/RAW_dataset/fist/sample_fist.jpg" width="200"><br />Fist Hand Gesture |<img src="/Dataset/RAW_dataset/open_hand/sample_open_hand.jpg" width="200"><br />Open Hand Gesture|
|:-------------------------:|:-------------------------:|
|<img src="/Dataset/RAW_dataset/pick/sample_pick.jpg" width="200"><br />Pick Hand Gesture |<img src="/Dataset/RAW_dataset/pinch/sample_pinch.jpg" width="200"><br />Pinch Hand Gesture|

Additionally, it is recommended to take these hand gesture pictures while your hand is in motion, on different locations across the screen, and with slight variations on how your fingers are positioned. This will better simulate your hand gesture movements within Blender. If you later find out that a certain hand gesture orientation or position is causing trouble, go back and take more hand gesture pictures with that specific hand position and orientation in mind. Once you finish taking your pictures, sort them into the provided ‘fist’, ‘open_hand’, ‘pick’, and ‘pinch’ folders within the “…/Dataset/RAW_dataset” directory”.

# Feature Extraction
Now you are ready to extract features from your images! Two methods were used in this implementation: HOG (histograms of oriented gradients) and a custom landmark distances method. 

In the HOG implementation, the provided image first needs to go through some preprocessing. First, to calculate the bounding box for the hand, the minimum and maximum x and y coordinates from MediaPipe’s detected landmarks are used. These values are then converted to pixel coordinates to extract an image of just the hand gesture. Next, for consistency, the hand image is resized to a 256x256 image and is converted to grayscale. 

Now, the actual HOG algorithm begins. The horizontal and vertical gradients, also known as the derivatives $${∂f\over ∂x}$$ and $${∂f\over ∂y}$$ respectively, are calculated for the image by apply the following kernels:

$$\begin{bmatrix} 1 & 0 & -1 \\\ 2 & 0 & -2 \\\ 1 & 0 & -1 \end{bmatrix}$$

<p align="center">Horizontal Sobel Filter</p>

$$\begin{bmatrix} 1 & 2 & 1 \\\ 0 & 0 & 0 \\\ -1 & -2 & -1 \end{bmatrix}$$

<p align="center">Vertical Sobel Filter</p>

The magnitude of the gradients is then calculated using the following equation:

$$||∇f|| = \sqrt{({∂f\over ∂x})^2 + ({∂f\over ∂y})^2}$$

And the direction of the gradients is calculated using the following equation:

$$θ = tan^{-1}({∂f\over ∂y}/{∂f\over ∂x})$$

The 256x256 image is then divided into 8x8 cells, resulting in a 32x32 grid. For each cell, a histogram is created using nine bins with intervals of 20 degrees, starting at zero and going up to 160 degrees. The magnitude and corresponding direction of gradients for each cell block is then iterated through, one by one, and a histogram is constructed by distributing the magnitude into each of the bins. The bin is found using the angle from the direction of gradients, and if the angle falls between two bins, then linear interpolation is used to distribute the magnitude between the two adjacent bins. Finally, once all of the histograms have been constructed, they are concatenated onto a list to create one feature descriptor for the hand image. More details about HOG and how it is used for object detection and classification can be found in the publication by Dalal et al. [1]. 

In this implementation, the HOG descriptors are calculated and saved into a .csv with their corresponding hand label so that they can be used to train a classifier later on. To create this HOG dataset, run the create_hog_dataset.py code provided in the “../Code” directory. 

The second feature extraction method relies solely on the MediaPipe landmark data, and was developed purely through experimentation. I tried using the raw landmark data points, tried calculating different statistics on the data, such as mean and standard deviation, but ultimately found that calculating the distances between all of the points created the best feature descriptor for classification. To do this, I simply iterated through every single point in the landmarks data and used the distance formula between every possible pair of landmarks. Finally, I used min-max normalization on all of the distances to create a final descriptor for the images.

Similar to the HOG implementation, the distance descriptors and their corresponding hand labels are saved into a .csv file to be used on a classifier later on. To create this dataset, run the create_mediapipe_dataset.py code provided in the “../Code” directory. 

Once you run both feature extraction programs, you should see two .csv files in the “../Dataset/MediaPipe_dataset” directory.

# SVM Training
Now, onto training the classifier. In this implementation, an SVM classifier is trained and used for hand gesture classification. You will see two programs in the provided code: svm_training.py and svm_testing.py. Both programs train SVM classifiers using the two created datasets, but the first uses all of the data to train the classifier while the second splits the data into training and testing portions for evaluation. If you are just looking to use this hand gesture system in Blender, then you should run the svm_training.py code to get a fully trained classifier. The two trained models will then be saved as .pkl files within the same “../Code” directory. Make sure you know where these files are located as you will need their paths for the Blender scripts in the next step. The results for the svm_testing.py code can be seen in the “System Testing” section of this tutorial.

# Blender Scripts
You are almost ready to start using hand gestures in Blender! In the provided code, you will find two programs in the “../Code/” directory: mediapipe_blender.py and hog_blender.py. Open these files in your favorite code or text editor, and locate the “svm_model_path” variable at the top of each file. Change this to the path to where your SVM models were saved. 

Finally, open up Blender and click the “Scripting” tab at the top of the screen. Click “New” at the top of the Scripts Window and copy and paste either the mediapipe_blender.py or hog_blender.py code into the window. Run the script using the triangular run button at the top of the Scripts Window. It might take a few seconds before the system sets up, but once the webcam footage window pops up, you can start using your hand gestures!

Let me walk you through the different hand gestures and their corresponding actions in Blender. First, there is the open hand gesture. This hand gesture serves as a transition between the other hand gestures and does not perform a specific action in Blender. In the Viewport window, however, you will see Blender’s 3D cursor follow your right hand and an empty object follow your left hand. These points serve as references to where your hands are in the viewport and will be seen for all hand gestures.

Next, there is the fist hand gesture. This hand gesture is used to pan around in the viewport. The screen will pan in the direction that you move your hand using your hand’s position in the webcam footage. Additionally, you can pan inwards or outwards by moving your hand closer or farther from the webcam. This functionality is implemented using the change in your hand’s size, which is computed by taking the distances between different hand landmarks.

The pinch hand gesture is used to rotate the viewport. A yaw rotation is achieved by moving your hand left or right, a pitch rotation is achieved by moving your hand up or down, and a roll rotation is achieved by moving your hand closer or farther away from the webcam. Again, landmark distances are used for hand size calculations, and landmark coordinates are used to get the hand’s location.

Finally, there is the pick hand gesture. This hand gesture is used to pick up objects in the scene so that you can move and rotate them. To pick up an object, make sure that your hand’s cursor is close to the object when performing this hand gesture. Once an object is picked up, you can move it left, right, up, down, or even farther or closer to you by moving your hand in that particular direction. To rotate the object, simply rotate your hand. While you can rotate the object in the roll and pitch directions, unfortunately, the system does not yet support object rotation in the yaw direction. 

The system also supports hand gestures using two hands. However, pinch and fist commands cannot be performed by both hands simultaneously. When picking up objects, the same object cannot be picked up by both hands.

Here are a few more notes on the hand gesture implementation itself. During an earlier implementation of this system, the raw landmark data points were being used for hand location and size calculations. However, this resulted in jittery and shaky movements within the Blender viewport. In order to help stabilize the movements, the following smooth function was created:

```
def smooth(new_value, old_values):
    return 0.35 * new_value + 0.35 * old_values[0] + 0.3 * old_values[1]
```

Here, a weighted average is computed using the current data and data from the previous two frames from the webcam footage. This function was applied on the hand landmark data and a variety of other calculations, helping achieve smoother controls within Blender.

Next, within the Blender script, a modal operator is used for the hand gesture controls. This allows the script to run in the background so that the viewport can be updated in real-time as the user performs different hand gestures. Additionally, the hand gesture detection code is performed on a separate thread so that it will not block the modal operator. Global variables are used to pass information from the hand gestures detection thread to the modal operator, where the viewport updates are performed.

Finally, matrix transformations were used to ensure all movements were relative to the view within the viewport. Blender’s API provides numerous helper functions to extract rotation and position data from either the current view and or from different objects. This, along with different calculations for the hand’s position and rotation axes, were used to create all of the viewport controls. For cursor movements and picking up of objects, Blender’s region_2d_to_location_3d and location_3d_to_region_2d functions were used to calculate cursor positions and calculate what object the cursor was sitting on top of.

# OpenPose
Originally, OpenPose was going to be used for hand landmark detection instead of MediaPipe. In fact, the system was already downloaded and set up to work within Blender, which involved using CMake and Visual Studio to configure and build the program. However, once the provided OpenPose test code was run, the results were slow and the entire arm needed to be within the webcam’s frame in order for hand landmarks to be detected. For these two reasons, MediaPipe was used instead for hand landmark detection. 

# System Testing
The first set of tests performed on this system were on the trained SVM classifiers. To ensure a fair comparison between the two classifiers, IDs were assigned to each image in the dataset, and both the HOG and distance descriptor datasets were combined using these IDs. This way, if for some reason one feature extraction method failed to extract features from a particular image while the other method succeeded, this image is left out of the SVM training to ensure both classifiers are trained with the same images. Additionally, both datasets used the exact same train and test split. Then, once the SVM training is complete, the accuracy, macro-precision, and macro-recall of each classifier are calculated and recorded. Metric functions provided by the Scikit-learn Python library were used for these calculations. You can view the used test code in the svm_testing.py file provided in the “../Testing” directory.

The following table summarizes the results for each classifier:

||Accuracy|Macro-Precision|Macro-Recall|
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|HOG Model|59.55%|60.81%|59.64%|
|Distance Model|94.19%|94.49%|94.23%|

As seen by these results, the distance feature extraction method works significantly better on unseen data over the HOG model, which is exactly what is needed for this hand gesture system to perform smoothly within Blender. These results are most likely attributed to the HOG SVM classifier overfitting on the used training dataset, which would need to be significantly larger and more varied in order for the HOG model to better generalize to unseen data. This claim is further supported by the accuracy results in the svm_training.py code:

||Accuracy|
|:-------------------------:|:-------------------------:|
|HOG Model|100%|
|Distance Model|96.93%|

Here, the entire dataset is used to both train and test the model. These results show that the HOG SVM classifier is indeed fitting too closely to the training data. I have tried applying Principal Component Analysis (PCA) to try to simplify and remove some of the noise within the data that the classifier might be overfitting to, and was able to somewhat improve the results.Here are the results for the HOG model using PCA: 

||Accuracy|Macro-Precision|Macro-Recall|
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|HOG Model + PCA|67.60%|67.48%|67.68%|

Overall, it is recommended that you use the distance implementation within Blender for best results.

Next, the system’s overall effectiveness was tested by performing specific tasks within Blender and then calculating the differences between the recreated scenes. To do this, three different starting scenes and three corresponding goal scenes were created within Blender. Then, using the hand gesture system, I attempted to recreate the goal scenes starting from the corresponding base scenes, making sure to record the time it took to complete the task as well. Finally, the vertex coordinates from the goal scene and the recreated scene were extracted onto .csv files, and the root mean squared error (RMSE) was calculated for all of the vertex points within the two scenes. The code to extract the scene vertices and calculate the scene differences can be found within the extract_blender_data.py and the calculate_scene_difference.py files respectively, located in the “../Testing” directory. The three base and result Blender scenes used to test this system are also found within the “../Testing” directory

Each test was performed three times per input type and the result averages were recorded.  For the computer mouse baseline, only viewport commands were used along with corresponding keyboard shortcuts. The recorded time is in minutes and seconds. Due to the HOG system’s poor performance, it was not used for these tests. Results for these experiments can be found in the following table:

| |Scene 1||Scene 2||Scene 3||
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|Input Type|RMSE|Time|RMSE|Time|RMSE|Time|
|Distance Model|0.7259|2:00.99|0.3681|00:40.91|1.3011|01:21.06|
|Computer Mouse|0.4540|1:22.30|0.0793|00:32.53|1.1923|00:27.57|

Overall, there is still a lot of work to be done in order for this hand gesture system to be comparable to that of traditional computer mouse controls. For all three tests, the computer mouse consistently recreated the scenes with higher accuracy and within a shorter amount of time than that of the hand gesture system. One observation during these tests was that the object rotation controls for the hand gesture system were particularly difficult to master, especially since yaw rotations are not supported. While the MediaPipe system does a good job of accurately extracting data points in 2D space, it does not do well with capturing point depth, causing some 3D controls to be difficult to implement. With further refinement and the addition of new controls, this system has the potential to attract newcomers to Blender by providing a more intuitive interface.

# Extra — Related Work
Connecting the real world to a virtual one has long been a challenge and a topic of interest for many researchers. For example, instead of using preexisting software, such as Blender, researchers Jang et al. developed their own augmented reality 3D sculpting system called AiRSculpt. In this system, users wear a head-mounted display (HMD) with an attached RGB-D camera to create clay-like figures in the space in front of them using different hand gestures and movements. Through a user study, Jang et al. discovered that overall, users enjoyed using the AiRSculpt system and found it easy to use [2]. While this work represents a significant step towards combining 3D modeling software with 3D user input, there is still plenty of room for improvement. First, the AiRSculpt system currently lacks an expansive set of tools for creating more complex scenes. By combining hand gesture controls with more powerful software tools, such as those provided by Blender, both new and experienced users can enjoy a fun and easy way to interact with the 3D environment while also having a wider range of creative possibilities. While this specific implementation does not include modeling or sculpting controls, future work can add these functionalities and more. Additionally, all tools used in this tutorial are accessible to a wider audience, as opposed to AiRSculpt’s more specialized devices, like an HMD or RGB-D camera.

Other research by Dias et al. also explores mapping hand gestures and movements to the manipulation of virtual 3D objects [3]. In their work, Dias et al. developed two manipulation methods, called “OneHand” and “HandleBar”. In the “OneHand” method, 3D objects are manipulated by simulating object grabbing with one hand. Like in this implementation, cursors are used to indicate where the hands are positioned on the screen. Object resizing, on the other hand, is handled through the use of two GUI buttons. In the “HandleBar” method, two hands are used to rotate, move, and resize objects, as if they were sitting on a bar between the user’s two hands. Through test subjects, the researchers found the “HandleBar” method was overall easier to use. Similar to the implementation presented in this tutorial, the researchers found limitations with detecting hand rotations from a single hand. However, like with the AiRSculpt system, only a limited number of tools were available for users to use and specialized equipment, such as the Microsoft Kinect depth sensor, were needed [3].

For this particular project, the MediaPipe framework by Lugaresi et al. was used to extract hand landmarks and map hand gestures to Blender commands [4]. In addition to hand landmark detection, MediaPipe offers a wide array of machine learning tools, including object detection, and face landmark detection and segmentation pipelines. Many of these pipelines include an integrated visualizer tool that allows users to assess both the behavior and topology of their developed systems. For example, the hand landmark visualization tool is used within this system to show the detected hand landmarks within the webcam footage. The MediaPipe framework is also designed in such a way that it allows for quick real-time feedback, which is incredibly important for systems such as this hand gesture implementation [4].

Another popular computer-vision system is OpenPose by Cao et al. [5]. Like MediaPipe, this system offers multiple keypoint detection tools for human poses, including hand, foot, face, and full body detection. OpenPose detects keypoints within videos and images by using deep convolutional neural networks and part affinity fields (PAFs), which help predict location, orientation, and associations between different body parts [5]. This system was originally going to be used for hand landmark detection, but, like mentioned previously, was replaced by MediaPipe due to its slower runtime and need for upper body keypoints. However, as other online tutorials have shown, OpenPose is still an incredibly useful tool for human pose detection within videos and images, which can then be used to animate characters within Blender. 

Finally, the HOG algorithm by Dalal et al. was used for feature extraction from the hand gesture images [1]. The specifics of how this algorithm works and how it was used for this system can be found in the “Feature Extraction” section of this tutorial. 

# References
[1] N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), San Diego, CA, USA, 2005, vol. 1, pp. 886–893, doi: 10.1109/CVPR.2005.177.

[2] S. A. Jang, H. I. Kim, W. Woo, and G. Wakefield, “AiRSculpt: A wearable augmented reality 3D sculpting system,” in Distributed, Ambient, and Pervasive Interactions. DAPI 2014, N. Streitz and P. Markopoulos, Eds., Lecture Notes in Computer Science, vol. 8530, Cham, Switzerland: Springer, 2014, doi: 10.1007/978-3-319-07788-8_13.

[3] P. Dias, J. Cardoso, B. Ferreira, C. Ferreira, and B. Santos, “Freehand gesture‑based 3D manipulation methods for interaction with large displays,” in Proc. of the International Conference on Distributed, Ambient, and Pervasive Interactions, May 2017, pp. 145–158, doi: 10.1007/978-3-319-58697-7_10.

[4] C. Lugaresi, J. Tang, H. Nash, C. McClanahan, E. Uboweja, M. Hays, F. Zhang, C.-L. Chang, M. G. Yong, J. Lee, W.-T. Chang, W. Hua, M. Georg, and M. Grundmann, "MediaPipe: A Framework for Building Perception Pipelines," arXiv preprint arXiv:1906.08172, 2019. [Online]. Available: https://arxiv.org/abs/1906.08172.

[5] Z. Cao, G. Hidalgo, T. Simon, S.-E. Wei, and Y. Sheikh, “OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields,” arXiv preprint arXiv:1812.08008, 2019. [Online]. Available: https://arxiv.org/abs/1812.08008.

