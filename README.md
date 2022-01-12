# Arabic-Calligraphy-Classifier
A Python SVM model to classifiy the 9 Arabic Calligraphies styles.



Arabic is the fourth most spoken language in the world, and it is the first language of
more than 200 million people across the world. Arabic script is the third most widely
used script after Latin script. Similar to other languages, it has grammar, spelling,
punctuation rules, pronunciation, slang, and idioms. Several characteristics beyond the
mere differences between languages make Arabic distinctive, including the number of
variations and the written form. Arabic handwritten text is divided into two main parts:
artistic writing called calligraphy and non-artistic (handwritten or printed) scripts. Arabic calligraphy (AC) is one of the most significant arts in the world. It is written only by hand. The first style of calligraphy was developed at the end of the 7th century.

Calligraphy style recognition (CSR) is considered an important research area for the
following reasons: 
* It helps to read the text content. In other words, knowing the used style reflects the rules used in the writing of different characters, which in turn eases the reading task. 
* It helps to recognize the different parts of a document. Some styles are defined to represent specific parts in the document such as titles, footers, paragraphs, etc.
* It also helps to grasp the history of a document. In paleography, CSR is used to define the era in which the document was written because styles appeared in different eras.
* It also helps to define the origins (i.e., geographical area) in which a document was written.
* It can be used for tutor purposes. Calligraphy learners use CSR systems to judge the quality of their writings in cases of experts’ absence.

# Pipeline
![grid_search_workflow](https://user-images.githubusercontent.com/62334815/149117580-0da1b461-dc7b-479f-a880-4480eb4c3112.png)



# Image Preprocessing 
* Image was binarized and then a threshold was applied on the grayscale version of the image.
* Further preprocessing will be needed to increase the model accuracy by removing background fonts check this link. [Gaussian Threshold](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)

![pre3](https://user-images.githubusercontent.com/62334815/149118048-c6a228c2-9397-4779-bbd8-49e8843f7ba6.JPG)
![pre2](https://user-images.githubusercontent.com/62334815/149118020-4de23e48-31d8-43d8-ae88-0caf13e3428e.JPG)

# Feature Extraction
* Local Phase Quantization was used to extract the features from the images.
 ![1-s2 0-S0020025517309052-gr2](https://user-images.githubusercontent.com/62334815/149119576-f70162ca-8c54-4dba-a620-0a6bbaa22c76.jpg)
![lpq](https://user-images.githubusercontent.com/62334815/149119789-45fad046-69ad-4e45-92cf-3f817c1e499a.JPG)

# Model Selection
* A non-linear SVM was used in this approach but consider using KNN as well or even Neural Networks
* We used SVM as it is better in dealing with ouliers compared to KNNs which were present in the dataset used , unfortunately such models don't have a very high quality datasets but if a one was found consider using KNN.
![svm](https://user-images.githubusercontent.com/62334815/149121437-b6c34dfe-8344-4c52-8d1c-e594d6aa79f9.JPG)

* Cross validation
Kfolds where used  with 5 splits and a max mean accuracy of 97%.
