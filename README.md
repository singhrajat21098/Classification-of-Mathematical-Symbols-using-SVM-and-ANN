# Classification-of-Mathematical-Symbols-using-SVM-and-ANN

The purpose of this project is to present and compare two different types of classifier models: Support Vector Machines (SVM) and Artificial Neural Networks (ANN). Both models are widely used in image classification tasks and have shown promising results. In this report, we focus on how these models perform when trained on three different types of image features: Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), and raw pixel values.
To provide a comprehensive overview, this report first introduces the concept of image classification and explains the importance of feature extraction in this process. Then, we discuss the HOG, LBP, and raw pixel features in detail, highlighting their respective strengths and weaknesses. Next, we provide a brief overview of SVM and ANN classifiers and explain how they work in the context of image classification. Finally, we present the results of our experiments and analyse the performance of both models on each of the three types of image features.



## Download the dataset from the Following Link
https://drive.google.com/drive/folders/1KfqmlqHWDRnQ9_JqNYWJ-8enjR7mmwFn?usp=sharing


## Pip Install all the libraries from requirements.txt before running the code



### Important Questions and Challenges I Faced during this project

The most important discussion points from the report are as follows:- 
Why there is a separate file image_preprocessing.py?
I noticed that some code such as loading dataset, pre-processing images and converting the images to HOG and LBP features would be same for both of the notebooks. Therefore, this python file was created to reuse the functions between these two files.

What is the reason for choosing Block norm as L2-hys for HOG feature extraction?
Since L2-Hys is useful to reduce the effect of illumination and contrast variation in images. Moreover, it also helps to avoid division by zero errors when the L2-norm is very small. 

What was the rationale behind my selection of the number of points and radius for Local Binary Pattern (LBP)?
The values for the number of points and radius for Local Binary Pattern (LBP) were selected through trial and error. While experimenting with different combinations, I settled on these values. However, the confusion matrix suggests that the model is having difficulty distinguishing between the "[" and "]" brackets. This indicates that further optimization of these values might be necessary to improve the model's performance.

What are the reasons for the high value of C in LBP SVM as compared to other models?
In SVM, the C value is a regularization parameter that determines the trade-off between maximizing the margin and minimizing the classification error. A high C value means that the SVM model is less tolerant of misclassifications, which can help to achieve higher accuracy on training data.
Although in LBP has just 26 values when compared to 2304 values in HOG and 2025 values in RAW pixel for each image, but these 26 values have high-dimensionality of data encoded in them. Therefore, a high C value is needed to capture the complex pattern in data.

What factors led me to compromise on accuracy in HOG ANN?
Based on my experiments with HOG ANN, I discovered that increasing the number of nodes in the first hidden layer resulted in a slight increase in accuracy. However, the increase in accuracy was not significant enough to justify the additional CPU power required to support the increased number of nodes. As a result, I decided to compromise on accuracy and reduce the number of hidden layers to conserve CPU power.


What factors contribute to a high number of epochs required for LBP-ANN training?
The accuracy value was decreasing monotonically during the training of the model in each epoch. To reach the highest accuracy value, I began with a very high epoch value and then analyzed the loss function and accuracy of each epoch to gradually decrease the number of epochs down to 400. Therefore, the epoch values were high.

What is the reason for RAW pixels having a large number of nodes in their Artificial Neural Network (ANN) model?
In all the three models, I tired to follow different models to determine the number of nodes for the ANN model. For ANN, the rule that gave me the best result was “use a range of values for the number of nodes in each hidden layer, such as between 2/3 and twice of the number of nodes in the input layer” as mentioned in Stack exchange discussion thread (Stack Exchange contributors, 2010).  

