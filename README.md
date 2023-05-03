# Rahul Reddy Vemparala️
## Data Science Portfolio

## Table of Contents
- [Introduction](#sections)
- [Course Work](#courses)
- [Libraries Used](#technologies-used)
- [Models Implemented](#models)
- [Projects](#projects)
- [Future Projects](#for-the-future)
- [Acknowledgments](#project-maintainers)

# Introduction:
Hello, my name is Rahul R Vemparala and welcome to my data science project portfolio. I am an experienced data engineer, having worked at [Modak Analytics LLC](https://modak.com/). Currently, I am a Master's student at [USF](https://www.usf.edu/), majoring in [Business Analytics and Information Systems](https://www.usf.edu/business/graduate/ms-bais/index.aspx). My passion for data science has driven me to pursue a career as a data scientist. I have developed several projects related to machine learning, natural language processing, and data visualization, and I look forward to sharing them with you. Thank you for visiting my portfolio! Be sure to check out my projects. 


# Course Work:

Following courses have substantially helped me acquire profound understanding of essential concepts of Data Mining and Data Science Programing domains.

1. Data Mining : Laid a foundation in basic concepts in statistics and supervises learning 
2. Data Science Programing: Explored advanced concepts in predictive analytics and given hands on experience on complex train models
3. Big Data for Business: Aided in Understanding the trade-offs between the Scaling resources horizontally and vertically.


# Libraries Used: 
1. **ML Libraries used**: Sklearn, imblearn, scikit, MLlib.
2. **Neural Network libraries**: Tensorflow, keras
3. **Text analytics**: transformers, huggingface
4. **Frameworks**: Hadoop Map-Reduce, Spark
5. **Other modules**: Pandas, numpy, matplotlib, Seaborn


# Models Implemented:
## Regression Models:
1. Linear Regression: for continous target
2. Multi variate Regression : more than one features 
3. Polynomial regression : regression of Higher order

## Classification Models:
1. K-nearest neigbours or K-nn: Binary and Multi class target, can also be applied for regression problems.
2. Decision Tree: Binary or Multi class target
3. Logistic Regression: Binary or Multi class target
4. SVM (Support Vector Machines): Better suited for non-linear relation between target and features. 
5. ANN (Artificial Neural networks) : binary, multiclass and regression problems

## Ensembles:
#### Bagging:
1. Random Forest

#### Boosting: 
1. Gradient Boost:
2. XGBoost 
3. AdaBoost


## Performance tuning:

Fine-tuning models involves adjusting model hyperparameters to find the optimal combination for the specific business problem. This process is crucial for improving model performance and ensuring the model is best suited for the business needs.
The following are the two techniques implemented for improving the performace and find the best suitable model respectively.
1. Random Search Cross Validation: Finding approximate range of parameters 
2. Grid Search Cross Validation : perform an exhaustive search for the best parameters and model.


## Text analytics: 
1. Feature extraction: sklearn or transformers to extract useful features from raw coprus of text.
2. Lemmatization: grouping different varients of words dowon to its root words
3. Simple vector decompostion: convert the features into linear combination of weights to make efficent modeling.



## Neural Networks: 
### Sequential Models:
1. **RNN**: Feedback looped perceptron with a sequential input.
2. **LSTM** ( Long short term memory) : Preserve long and short term memory based on gate operations
3. **GRU**(Gated Recurrent Unit): simplified version of LSTM

### Deep Neural Networks:
Customizable hidden layers, activation functions. and initializers.
1. **Convolutional networks** :image segmentation
2. **Auto Encoder-decoder** : Image denoising, image compression, anomaly detection.


# Projects:

## Breast Cancer Detection: 
### Business Problem:
The increasing cases for TNBC cancer which is an unpredictable varient of breast cancer, there is a difintive need to understand the initial biposy reports and to forcast of potential death due to cancer.
### Models implemented: 
k-nn, Decision Tree and Ensembles like (Random Boost, Gradient Boost).
### Aim: 
Model focus on minimizing FN error which can incur a substantial cost.
### Outcome:
We produced a substantially accurate model and improved Sensitivitly(Recall) of the model by 1% over the previous implementation.




## Chess Game Outcome Prediction Model:
### Business Problem:
**Background information**: Undoubtedly number one platform for online chess players is chess.com known for its wide array of features for studying different gameplays and strategies.
One noticable feature if the move detector which tells if you made a bad move or not. which is quite handy for a beginner level game play.
**Problem:** I think that there is much more need for predicting if a player could win and if so in how many moves will be a game changer for beginner play or it could be intresting to see how seasoned players would perform with this new feature.
My business problem is one such case, with the limited access to chess data, i would only consider one famous gameplays
which is called king-rook-king endgame, we need a model that would predict the number of moves its gonna take to win against black based on the current move.
### Models implemented: 
SVM, Decision Tree and Ensembles like (Random Boost, Gradient Boost) and artificial Neural network.
### Aim: 
How many moves needed to checkmate black ,or will it end in a draw
### Outcome:
With trying various models, the neural network has produced a fairly accurate model with good f1-score(commanly the important metric for multiclass problems)



## Microsoft Stock price prediction:
### Business Problem:
The stock data is a time series data with stock data like open, high,low,close etc.
### Models implemented: 
SVM, Decision Tree and Ensembles like (Random Boost, Gradient Boost) and artificial Neural network.
### Aim: 
Obviously trying to predict the opening price for the next 7 or 8 days
### Outcome:
The model I built is univariate, sequence predictor, using LSTM cell. I was able to predict the open stock price for the next week in future.


## Future Projects:
1. **Phising Prediction**: Fairly new dataset, Aim to do perdictive analyis using different libraries like MLlib.
2. **Ethereum Price Prediction** : Planing to build a multivariate ,sequence predictor for the time series stock data using keras.
3. **Text analytics** : Finding sources for lyrical text documents and build a language model using pretrained model like BERT Or GPT-2 models 



## Acknowledgments
I would like to express my deepest gratitude to Professor Timothy Smith for the invaluable guidance and support throughout my journey as a student of the Business Analytics and Information Systems program at the University of South Florida. Professor Tim's mentorship and teachings have played a significant role in shaping my skills and knowledge in the field of data science.

### Professor Contact Information
1. [Email](mailto:smith515@usf.edu)


## Contacts: 
1. [Email](mailto:rahulreddyvemparala@gmail.com)
2. [LinkedIn](https://linkedin.com/in/rahul-reddy-vemparala-11609924a)