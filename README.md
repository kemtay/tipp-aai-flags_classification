# tipp-aai-flags_classification
To select the best model to classify the “religion” of a country based on its flag colours and shapes.

## Dataset
- The dataset used is flag.data from https://archive.ics.uci.edu/ml/machine-learning-databases/flags .
- The selected target is religion. There are 7 religions, hence this a multi-class classification.
- There are 3 data types of features: Binary (0/1), Discrete (counts of a shape) and Categorial (colours).
- All discrete features are binarized to 0/1 as count=0 outweighs other counts for all features except ‘colours’ which count<=4 dominates.
- All categorical features are label-encoded for model training purposes. 

## The python program explained:
The python program is designed in accordance to the common ML processes as the followings:
- Reads the dataset of 194 samples from a file into an pandas dataframe.
- Performs Exploratory Data Analysis by plotting histogram and bar charts to identity noises/outliers to be dropped from the dataset.
- Pre-processes the data by dropping data with ‘religion=7(others)’ as they are considered noise since they “unclassifiable”. Features are encoded as necessary. Not outliers can be determined since the dataset is very small.
- All 23 features are used in training the 7 Machine Learning models. Dataset are split with train_test_split(test_size=0.3) for training (fit->predict) the models. Accuracy scores and confusion matrices are collected and plotted.
- Cross validation (cv=30) are applied for all the models with the same dataset. Validation and Training scores are collected to detect any overfitting and underfitting of the trained models.
- Features selected with RFE (7-22 features and step=1) are trained with all models to identify the possibilities of reducing the number of features. The exhaustive RFE is possible as the dataset is small. Another set of features are selected based on their Correlations scores.
- Hyper-parameters of all models are tuned with GridSearchCV (cv=20) with the features selected from Correlations matrix. The models resulted from GridSearchCV (cv=20) are trained against the features selected from Correlations matrix with accuracy scores and confusion matrices collected. The highest accuracy scored model is selected at this stage.
- Further validates the GridSearchCV selected model by collecting accuracy scores with “StratifiedShuffleSplit(n_splits=10, test_size=0.3)” for all models. Correlations selected features are used to train the models.

## Results Summary
- RandomForestClassifier is the model with the highest accuracy score of 0.596 with hyper-parameters: max_depth=6, n_estimators=20. 
- The models are trained with features selected based on correlations. 
- The hyperparameters are discovered through GridSearchCV. 
- Learning curve of cross-validations do not show any consistency between the train and validation scores, this could be due to the unbalanced dataset. 
- The considerably low accuracy score could be due to the small dataset with only 194 samples with little explanatory features.
