# Simple Regression
Collection of Simple Regression Analyses

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Goal
This project was undertaken to learn more about regression analyses using the scikit-learn library. It consists of five notebooks analyzing three datasets.

## Heating and Cooling Load Predictor

This was the first analysis undertaken. The purpose was to get some practice exploring a dataset, pre-processing and to use the lasypredict library for the first time. In hindsight, I now realize that two of the features are categorical and should have been encoded before running machine learning (ML) on them. Although the data was not formatted properly and the models were not tuned at all, they still achieved an $R^2$ score of 0.90. This indicates that this is likely a relatively easy regression task, so it was a good choice to start with.

## Power Plant Output Predictor

### Data preprocessing and EDA

I performed data cleaning and a brief exploratory data analysis (EDA), where I observed the relationships between features and the effects of whitening them. The pairplots clearly showed how whitening eliminates skewness, resulting in much more Gaussian distributions after the transformation. Subsequently, the lazypredict library was used to estimate the best model, and a hyperparamter search was performed using manual looping (this is improved upon by utilizing sklearn's GridSearchCV in the next analysis). The effects of changing one parameter (while holding all others at the optimal values) is illustrated. It appears the effects of the hyperparameters is not particularly strong (i.e. many models perform relatively well), so there is not a clear maximum in these curves. 

### Neural Network Analysis

I implemented a basic neural network, with three hidden layers. This was probably overkill, although the fact that both the training and test losses are similar indicates it is probably not overfitting. However, the neural network does not perform better than the other machine learning models assessed. Given that the dataset seems to be fairly predictable, I would have thought that a neural network would be able to make nearly a perfect regression prediction. This likely indicates that I am not constructing the model correctly, and need to get more experience in the practicalities of building and training these networks, or that I just do not understand the limitations of neural networks well enough.

## Forest Fire Burn Area Dataset

I divided the work on this datasest into three parts.

### 1. Data Preprocessing and EDA

- Checked for NaNs and missing values
- Looked at the duplicate entries
- Converted categorical features to numerical values
- Ran a pandas profile_report()
- Data exploration (pairplot and correlation plot)

### 2. Feature Selection

- Used the sklearn function SelectKBest to score and rank the features.
- Trained a random forest regressor, and used the feature_importances_ attribute to compare these rankings to those found using SelectKBest. Both ranked temp strongly as the most important feature, but after that, there is little agreement between the two methods.
- Feature importance is investigated using recursive feature selection. There was a sharp drop in the $R^2$ score after removing DMC in both sequences. Therefore, DMC is likely an important feature in these models.
    - In hindsight, I should have used adjusted-$R^2$ to properly compared between models with different numbers of features
- The standard deviation of cross-validation scores was quite large, so any conclusions made about $R^2$ changes after removing features should be viewed with skepticism. The point I took from this is these drops in performance (after removing a feature) are the type of indicators I should look for in future analyses. 
- I would imagine that feature importance depends on the model class being used, and if the results are more important, this analysis should be part of a more comprehensive feature selection process. 
- I also inadvertently discovered that failing to shuffle the data before performing cross-validation greatly reduced the model performance. This could indicate the data is not well mixed with regard to the target variable, or that there are some ordinal characteristics in the data that bias the model when not accounted for. 

### 3. Burn Area Prediction

#### Further preprocessing
- Incorporated all modelling aspects to this point
- The categorical features are binary encoded (wrote a custom binary encoding function)
- Whitened the data (wrote custom whitening function)
- Balanced the dataset with regard to the "month" feature in three ways
    1. Upsampling
    2. Dividing the majority class into subsets
    3. A mixture of the two above methods

#### Model Selection
- Three different model classes trained (sklearn's GridSearchCV using k-fold cross-validation)
    - Upsampling -> overfitting (does not significantly improve test performance)
    - Splitting the majority class -> worse performance (relatively small training sets after splitting)
- The best model had an $R^2 \approx 0$ (trivial fit)
    - Predicting the mean target value performs as well as the trained model
![](https://github.com/marshineer/SimpleRegression/blob/main/readme_images/forest_fire_prediction_results.png?raw=true "Confusion Matrices")

#### Model Improvement
- Split the model into two parts:
    1. A classification model to identify whether a fire occurs
    2. A regression model to estimate the size of a fire, if one is predicted to occur
- The data is fairly evenly split between the "fire" and "no fire" classes
    - Accounts for the original, zero-skewed target variable (burn area)
- Unfortunately, the classification models performed little better than chance
- Since the classification task fails, the regression model was abandoned

Note: In order to determine whether there were better ways to perform this analysis, I looked online for other people's analyses of this dataset. MWhat I found fit into one of the following categories:
- Analyses where they agreed with my assessment that the data does not contain enough information for proper classification
- Analyses where they performed some kind of transformation or relabelling of the data before applying maching learning
- Analyses where there was either an obvious oversight or what I would consider flawed methodology

A detailed description of my findings can be read in the notebook.

## Minnesota I-94 Traffic Volume Analysis

This dataset was also analyzed in three parts

1. Data pre-processing and EDA
2. Model selection
3. Hyperparameter tuning

### 1. Preprocessing and Exploratory Data Analysis

#### Preprocessing steps
- Checked for NaNs and missing values
- Investigated suspicious values in "temp", "rain_1h" and "snow_1h"
- Removed duplicated date-time rows (occurred when multiple weather categories applied)
- Dropped detailed weather descriptions
- Holidays are only marked at the midnight hour. These were expand to fill all hours of the day
- Added "hour" and "day of year" categorical features
- Encoded all categorical features using either cyclical or binary encoding
- Normalized the numerical data

#### EDA Takeaways
- People drive mostly during the day on weekdays. Traffic volume peaks at rush hour during the week and mid-afternoon on weekends
![](https://github.com/marshineer/SimpleRegression/blob/main/readme_images/traffic_weekday_weekend_split.png?raw=true "Weekday/Weekend Hourly Distributions")

- It doesn't seem to matter what the weather is like, people drive at the same rate regardless
    - Variance is important. Averages would indicate counterintuitive conclusions, like people drive more on hazy days
    - The high variance indicates weather is not a major factor in determining whether people drive
![](https://github.com/marshineer/SimpleRegression/blob/main/readme_images/traffic_weather_effect.png?raw=true "Effects of Weather on Traffic Volume")

- The night/day split captures the target variable's bimodality
![](https://github.com/marshineer/SimpleRegression/blob/main/readme_images/traffic_day_night_split.png?raw=true "Day/Night Split of Hourly Distributions")

### 2. Model Selection

Goal: Machine learning models are trained and compared to learn more about how each model class reacts to different preprocessing techniques. These models are later compared to the performance of a neural network, which should be the most flexible (and least interpretable) model class tested. 

#### Further Preprocessing
- Split the data into train and test sets in two different ways
    1. Shuffle all the data, and split by a train:test fraction
    2. Leave the data ordered and pull out a single year
- Normalize data to the range [0, 1]
    - Normalization required after splitting, because test data must be normalized using training values

#### Training
- Three different model classes are trained:
    1. Random Forest
    2. SVR
    3. XGBoost
- Models are trained using sklearn's RandomsearchCV
- Results compared between the four preprocessing combinations (shuffled/unshuffled and normalized/unnormalized)

#### Conclusions and Lessons Learned
- SVM methods don't scale well (long training and prediction times)
    - Limits hyperparameter tuning possible
- Data should not be whitened if the original data's distributions are non-Gaussian
    - Whitening reduced the SVR and XGBoost performance (not shown here)
- $R^2$ does not tell the whole story (see image below)
    - The random forest has a high $R^2$ but is likely overfitting
    - SVR has a much lower $R^2$, but predictions look more realistic
    - This is a reminder that results should be evaluated in more than one way
![](https://github.com/marshineer/SimpleRegression/blob/main/readme_images/traffic_prediction_distributions.png?raw=true "Model Prediction Distributions")

- Depending on the model family, normalization may be a critical step
    - Random forest and XGBoost are not affected by normalization (these models are agnostic to the underlying distribution)
    - SVR's performance is greatly improved with normalization
- Encoding the categorical features has a negligible effect on the random forest model
    - Maybe the model utilized the original categorial data as well as the encoded features
    - Or, the categorical features are not important for the predictions (indicated by EDA)
- I am not sure how expert models should work
    - It was expected that the individual expert models would have better performance than the single model
    - However, the individual day/night model's was worse than the single model
    - Maybe because there is less training data?
![](https://github.com/marshineer/SimpleRegression/blob/main/readme_images/traffic_day_night_predictions_ml.png?raw=true "Day/Night Expert Model Prediction Distributions")

#### Ideas for future improvements and analysis
- Unsupervised clustering on the target variable
    - Train expert models on these clusters
- Randomized grid search followed by hyperparameter tuning
- Removing features with high zero frequency (eg. rain_1h)
    - Use adjusted $R^2$ to measure this
- Calculating the variance explained by a day/night boolean feature


#### Neural Network (NN) Model
Can a simple neural network perform better than other ML methods, with minimal tuning?

In most resources I found regarding neural networks for multimodal regression, it was suggested the data should be divided into its component modalities. Here, this is done by finding a natural division in the data (day/night in this case), and two separate "expert" models are trained.

#### Conclusions and Lessons Learned
- The single NN model performed nearly as well as the XGBoost model
![](https://github.com/marshineer/SimpleRegression/blob/main/readme_images/traffic_nn_predictions.png?raw=true "Neural Network Prediction Distributions")

- The separate day/night models evaluate similarly to those of the XGBoost model
    - Something seems off here: The separate model $R^2$ scores are quite low, but the concatenated output and prediction vectors calculate to a much better $R^2$. This seems like it's somehow accounting for the missing data, although I'm not sure how this is possible.
- Similar performance between the combined day/night expert models and the baseline neural network
![](https://github.com/marshineer/SimpleRegression/blob/main/readme_images/traffic_day_night_predictions_nn.png?raw=true "Day/Night Neural Network Prediction Distributions")


### 3. Hyperparameter Tuning

#### Comparison of Random Parameter Search
- Look at the parameter ranges corresponding to well-performing models
    - Which parameters have strict ranges for good performance and which are more robust?
- Few parameters show a clear trend in performance
- Most other parameters show general robustness to the parameter range
- Some parameters show differences in performance, depending on the way the data is split
- Once a suitable range is determined, the parameters could be further tuned with an exhaustive grid search
- Further statistical tests are possible for model selection

#### Conclusions and Lessons Learned
- There are three patterns to the parameter distributions:
    1. An even distribution of parameter values and consistent $R^2$ score (model is robust to the value)
    2. An clear best parameter across most models (eg. "min_samples_split" in the random forest model)
    3. A clear correlation between parameter value and performance (eg. "epsilon" in the SVR model)

Note: Parameters evaluated using cross-validation, so there may be some over-fitting. Where there is no clear consensus, one should lean towards regularization.
