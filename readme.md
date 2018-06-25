# Dogs vs Cats Kaggle Dataset Study Project

The dataset contains 25,000 images of dogs and cats. 12500 Images in each class.

As a study project, I have implemented a simple MLP network to complex network like Bottleneck feature extraction, fine-tunnig etc. The idea is to implement different architecture and concept and learn the pipeline with actual example and compare loss and accuracy. 

**NOTE1 : I have not focused on hyperparameters tunning much as the main idea was to learn the process flow of each concept. Still I tried to train the models untill I get satisfactory results in each part. I was not able to train for long time and could not fine-tune each models because of resource constraints. (I have only CPU and very basic GPU with limited access.)

**NOTE2 : Change the image path accordingly before you run any notebooks.
Folder structure should look like below.

|-train
|-test1

I haven' included data and saved model/weights files as they are big files. 


I have divided this project in seven notebooks. Below is the brief explanation of each notebook.

**a)	1_Simple Neural Network (MLP) for Dog-Cat dataset Kaggle.ipynb**

In this notebook, I have implemented a neural netowrk with two hidden layers with 768 and 384 nodes respectively.

**b)	2_CNN for Dog-Cat dataset Kaggle.ipynb**

In this notebook, I have implemented Convolutional Neural Network resembled to LeNet Architecture. 

**c)	3_CNN_with_augmented_data for Dog-Cat dataset Kaggle.ipynb**

In this notebook, I have doubled number of images in each class using data augmentation (Vertical flips) and kept the same network as part b.

**d)	4_CNN_Feature_Extraction for Dog-Cat dataset Kaggle.ipynb**

In this notebook, I have used VGGNet architecture pre-trained on ImageNet dataset and extracted the last layer features by freezing all the above layer weights.

**e)	5_CNN_with_bottleneck_features for Dog-Cat dataset Kaggle (Part1).ipynb**

In this notebook, I have used VGGNet architecture pre-trained on ImageNet dataset. All the weights in Convolution layers are freezed and I have appended my own Fully connected layer and trained it. I will use this weights as a starting point in 7th notebook (Fine-tunning the network)

Performance (Loss, Accuracy) in this notebook seems okay to me! I could have worked with hyperparameters tunning but it was too much time for training so I have kept as it is!

**f)	6_CNN_with_bottleneck features for Dog-Cat database Kaggle (Part 2).ipynb**

This notebook is almost same as above notebook. The only difference in this notebook is that I wanted to save all the features upto Freezing layers as this is time-consuming process. Once I saved it, next time I can load these features and train Fully connected layers and fine-tune hyperparameters. This training takes very less time even on CPU (~2min per epoch).

**Here is to mention that, till now I have not explored Keras' ImageDataGenerator class much and the data was not in class-wise subfolders. I have not trained my model using `flow_from_directory` method. After saving `train` and `validation` features in `.npy` file I loaded them again to train FC layer. But then I realized that I don't have labels. Though I have used previous labels while training, I feel that I am doing something wrong here and my training is not correct. **

Even if this is not correct or doubtful I have included this notebook here. Next steps are to find the existing issue in current notebook and then use `ImageDataGenerator`s `flow(X,y)` method to train the model. 

**I would consider this notebook as a pending/incorrect. **

**g)	7_CNN_Fine_Tunning for Dog-Cat dataset Kaggle.ipynb**

In this notebook, I have used VGGNet architecture pre-trained on ImageNet dataset. I have loaded the weights from Part e. Basically, I have used ImageNet dataset weights upto last Convolution layers and trained weights of Fully connected layers. (Here I have saved this complete model in part e). By doing this, I have initialized the proper weights as in fine-tunning process, we cannot initialized random weights. After initializing, I have freezed layers upto 4th Convolution block (Upto 15 layers) and fine-tune last (5th) Convolutional block and FC layers.

**Summary**

| File - Name                                                                | Validation Loss           | Validation Accuracy  | Saved Model                         |
|:---------------------------------------------------------------------------|:--------------------------|:---------------------|:------------------------------------|
| 1_Simple Neural Network (MLP) for Dog-Cat dataset Kaggle.ipynb             | 1.1090                    | 63.7920%             | dog_and_cat_simple_nn.h5            |
| 2_CNN for Dog-Cat dataset Kaggle.ipynb                                     | 0.4101                    | 82.2560%             | dogs_and_cats_CNN_1.h5.             |
| 3_CNN_with_augmented_data for Dog-Cat dataset Kaggle.ipynb                 | 0.4151                    | 81.4400%             | dogs_and_cats_with_data_augmented.h5|
| 4_CNN_Feature_Extraction for Dog-Cat dataset Kaggle.ipynb                  | 0.2330                    | 89.7920%             | dogs_and_cats_CNN_feature_extraction.h5|
| 5_CNN_with_bottleneck_features for Dog-Cat dataset Kaggle (Part1).ipynb.   | 0.2434                    | 90.2080%             | dogs_and_cats_CNN_custom_bottleneck_1_from_part1.h5|  
| 6_CNN_with_bottleneck features for Dog-Cat database Kaggle (Part 2).ipynb  | 0.0308 (Doubtful)         | 99.2000% (Doubtful)	| dogs_and_cats_CNN_custom_bottleneck_1_from_part2.h5
| 7_CNN_Fine_Tunning for Dog-Cat dataset Kaggle.ipynb                        | 0.0790                    | 96.6400%             | dogs_and_cats_CNN_fine_tuning.h5.   |


**Reference https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
