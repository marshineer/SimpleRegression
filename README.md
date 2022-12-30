# Simple Regression
Collection of Simple Regression Analyses

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Goal
This project was undertaken to learn more about regression analyses using the scikit-learn library. It consists of five notebooks analyzing three datasets.

## Project Descriptions

### Heating and Cooling Load Predictor

This was the first analysis undertaken. The purpose was to get some practice exploring a dataset, pre-processing and to use the lasypredict library for the first time. In hindsight, I now realize that two of the features are categorical and should have been encoded before running machine learning on them. Although the data was not formatted properly and the models were not tuned at all, they still achieved an $R^2$ score of 0.90. This indicates that this is likely a relatively easy regression task, so it was a good choice to start with.

### Power Plant Output Predictor

#### Data preprocessing and EDA:

I performed data cleaning and a brief exploratory data analysis (EDA), where I observed the relationships between features and the effects of whitening them. The pairplots clearly showed how whitening eliminates skewness, resulting in much more Gaussian distributions after the transformation. Subsequently, the lazypredict library was used to estimate the best model, and a hyperparamter search was performed using manual looping (this is improved upon by utilizing sklearn's GridSearchCV in the next analysis). The effects of changing one parameter (while holding all others at the optimal values) is illustrated. It appears the effects of the hyperparameters is not particularly strong (i.e. many models perform relatively well), so there is not a clear maximum in these curves. 

#### Neural Network Analysis

I implemented a basic neural network, with three hidden layers. This was probably overkill, although the fact that both the training and test losses are similar indicates it is probably not overfitting. However, the neural network does not perform better than the other machine learning models assessed. Given that the dataset seems to be fairly predictable, I would have thought that a neural network would be able to make nearly a perfect regression prediction. This likely indicates that I am not constructing the model correctly, and need to get more experience in the practicalities of building and training these networks, or that I just do not understand the limitations of neural networks well enough.

### Forest Fire Burn Area Dataset

I divided the work on this datasest into three parts.

1. Data pre-processing and EDA:
	- Checked for NaNs and missing values.
	- Looked at the duplicate entries. They seemed reasonable so did not remove due to lack of domain knowledge.
	- Converted categorical features to numerical values. Was not yet aware of categorical feature encoding for regression analsysis, so this is missing.
	- Ran a pandas profile_report(), which identified that "rain" and "area" have high proportions of zeros (addressed in a later analysis).
	- Data exploration consists of just a pairplot and correlation plot, showing basic relationships between features. 
	- A more creative analysis can be found [here](https://doi.org/10.3390/su141610107). This is an area of data analysis that I would like to improve on.

2. Feature selection:
	- Used the sklearn function SelectKBest to score and rank the features.
	- Trained a random forest regressor, and used the feature_importances_ attribute to compare these rankings to those found using SelectKBest. Both ranked temp strongly as the most important feature, but after that, there is little agreement between the two methods.
	- Feature importance is investigated using recursive feature selection. There was a sharp drop in the $R^2$ score after removing DMC in both sequences. Therefore, DMC is likely an important feature in these models.
		- In hindsight, I should have used adjusted-$R^2$ to properly compared between models with different numbers of features
	- The standard deviation of cross-validation scores was quite large, so any conclusions made about $R^2$ changes after removing features should be viewed with skepticism. The point I took from this is these drops in performance (after removing a feature) are the type of indicators I should look for in future analyses. 
	- I would imagine that feature importance depends on the model class being used, and if the results are more important, this analysis should be part of a more comprehensive feature selection process. 
	- I also inadvertently discovered that failing to shuffle the data before performing cross-validation greatly reduced the model performance. This could indicate the data is not well mixed with regard to the target variable, or that there are some ordinal characteristics in the data that bias the model when not accounted for. 

3. Burn area prediction:
	- This incorporated all modelling aspects that I have worked on up to this point.
	- The categorical features are encoded using a binary encoding. Since scikit-learn does not have a built-in binary encoding function, I wrote one.
	- I wrote functions to whitening the data, as well as balancing the dataset with regard to particular features, in this case "month".
	- The feature is balanced three ways: 
		1. Upsampling
		2. Dividing the majority class into subsets
		3. A mixture of the two above methods
	- Three different model classes are trained using k-fold cross-validation and the hyperparameters tuned using scikit-learn's GridSearchCV method.
		- Upsampling results in overfitting during training, but does not significantly improve test performance, which makes sense.
		- Splitting the majority class resulted in worse performance. This could be due to the relatively small training sets that remained after splitting the data.
	- None of the models performed particularly well, with the best hovering around $R^2 \approx 0$, indicating a trivial fit. This means that simply always predicting the mean target value would perform as well as the trained models.
	- In an attempt to improve performance, the model was split into two parts: a classification model to identify whether a fire occurs, and a regression model to estimate the size of a fire, if one is predicted to occur.
		- The data is fairly evenly split between the "fire" and "no fire" classes, accounting nicely for the original, zero-skewed target variable (burn area).
		- Unfortunately, the models trained to classify on whether a fire occurs performed little better than chance.
		- Since the classification task fails, the regression model is abandoned.
	Note: In order to determine whether there were better ways to perform this analysis, I looked online for other people's analyses of this dataset. Most of what I found either consisted of analyses where they agreed with my assessment that the data does not contain enough information for proper classification, those where they performed some kind of transformation or relabelling of the data before applying maching learning, or those where there was either an obvious oversight or what I would consider flawed methodology.

A detailed description of my findings can be read in the notebook.

### Minnesota I-94 Traffic Volume Analysis

This dataset was also analyzed in three parts

1. Data pre-processing and EDA:
	- This was a more comprehensive and informed preprocessing than performed on the forest fire dataset.
	- 

Preprocessing the data consisted of the following steps:
- Checked for NaNs and missing values.
- Investigated suspicious values in "temp", "rain_1h" and "snow_1h".
    - Zero Kelvin temperatures were linearly interpolated.
    - Snow only recorded for a span of ~2.5 weeks (over 6 year total time period), which does not match historical weather data. Therefore, the snow feature was dropped.
    - There is one entry with 9831mm of rain recorded for the hour. This value was reduced by a factor of $10^4$, as historical weather data shows trace amounts of rain on this day. Perhaps it makes sense to drop the rain category as well, since it has so many zeros (future work).
- Removed duplicated hours. Some hour rows were repeated (apparently when multiple weather categories applied.
- Detailed weather descriptions do not seem to provide more information than could be gleaned from the simple description plus the weather data. Therefore, dropped the feature.
- Holidays are only marked at the midnight hour. These were expand to fill all hours of the day.
- Added "hour" and "day of year" categorical features, derived from date_time, since this data seems relevant in predicting traffic volume. 
- Encoded all categorical features using either cyclical or binary encoding.
- Normalized the numerical data. Since none of the variables are particularly Gaussian, no standardization is performed.

Notes:
- The dataset only contains data for the first half of 2014 and last half of 2015 (about a year is missing).
- The target variable is multimodal. Tree regressors (random forest and XGBoost) are agnostic to this, but this affects SVM methods, such as SVR.

Takeaways
As would be expected, people tend to drive mostly during the day, and traffic volume peaks at rush hour. The distribution on weekends is smoother, peaking in mid-afternoon. Overall, there is more traffic volume on weekdays than on weekends. This all passes the smell test. 

Surprisingly, it doesn't seem to matter what the weather is like. People drive at the same rate regardless of weather conditions (with the exception of squalls, although this could just be a small sample size issue). There are some differences in the average rates that people drive during each type of weather condition, but the variance for each category is so large that it would be specious to attribute meaning to these relatively small differences in mean volume. 

This is a good example of why looking at variance is important when making these comparisons. If only the averages were displayed, it would look like people were more likely to drive during than on a clear day, which would be highly counterintuitive. The high variance likely means that there are other more important factors which determine when people drive, for example, going to work. 

The night/day split captures much of the target variable's bimodality. Therefore, it may be possible to train two "expert" models for night and day separately, then combine their predictions to improve performance.



Machine Learning
Methods:
- Split the data into train and test sets in two different ways
    1. Shuffle all the data, and split by a train:test fraction
    2. Leave the data ordered and pull out a single year
    - Since the data corresponds to a time series, it is possible there is some temporal information that can be gleaned by leaving the data in order. Pulling out a contiguous year allows the model to train on data in this way.
- Since the numerical data is not Gaussian distributed, it does not make sense to standardize it, as this would change the distributions (although tree methods such as random forests and XGBoost are agnostic to this). Therefore, the data is only normalized to the range [0, 1]. The target variable is also normalize to this range.
- The models are trained using a randomized parameter search, and the results are compared across the four combinations of data preprocessing:
    1. Shuffled and unnormalized
    2. Unshuffled and unnormalized
    3. Shuffled and normalized
    4. Unshuffled and normalized

Conclusions and Lessons Learned
- SVM methods don't scale well with large amounts of data (long training and prediction times). This limits the hyperparameter tuning possible (although the SVR model does not have that many hyperparameters anyway).
- Whitening the data (not shown here) ruined the SVR and reduced XGBoost performance. Upon looking into the cause of this, it was discovered that if the original data's distributions are non-Gaussian, then the data should only be normalized and not whitened.
- Although the random forest has a relatively good $R^2$ score (explained variance), you can see from the distribution of predictions that it is probably overfitting, as it is only predicting a small set of traffic flow volumes. Therefore, the overall performance is not actually that good. On the other hand, SVR has a much lower explained variance, but its prediction distributions look more realistic. This is a good reminder that models should be evaluated in more than one way, to determine which one performs best. 
- As expected, the random forest and XGBoost performance scores are not affected by normalization (since these models are agnostic to the underlying distribution of the data). However, SVR's performance is greatly improved with the addition of normalization preprocessing.
- Encoding the categorical features gives a negligible improvement to the random forest model. This may indicate that the model was able to utilize the simply encoded categorial data nearly as well as the encoded features. It is also possible that the categorical features (the weather description in particular) are not important for the predictions. Evidence for the weather category's lack of importance can be found in the EDA.
- Splitting the data into day/night sets and training expert models actually reduces the individual model's predictive ability. This may be because it reduces the amount of training data available, and because XGBoost is already agnostic to the underlying distributions. Perhaps this would have been more effective if tested with a model class that does not handle multimodal target variables well. This will be explored when looking at training a neural network to solve the task. 

Ideas for future improvements and analysis
- Could use unsupervised clustering, such as Guassian mixture modelling or Kernel Density Estimation, to divide the target variable into its modalities, and then train models based on these clusters.
- After performing a randomized grid search to find the range of best parameters, the model could be further improved with hyperparameter tuning using a grid search. 
- Look at effects of removing rain_1h feature (many zero values perhaps act as noise). Must use adjusted-$R^2$ to measure this.
- Perhaps calculating the variance explained by a day/night boolean feature would help explain the results of splitting the model.


Neural Network Model
This model is implemented to determine whether a simple neural network can achieve performance similar to or better than other machine learning methods, with minimal tuning. It is also meant to help me learn more about what types of problems simple neural networks can be applied to.

In most resources that I found addressing the use of neural networks on multimodal target regression variables, it was suggested that the data be divided into its component modalities. This can either be done as above (finding a natural division in the data such as day/night), or by unsupervised methods such as Gaussian mixture modelling or Kernel estimation. In order to try this, the data is again split into night and day, and two separate models are trained.

Conclusions and Lessons Learned
- The neural network model performed nearly as well as the XGBoost model, with minimal parameter tuning. It is a little surprising how well it worked, since I was under the impression that neural networks do not handle multimodal targets well. 
- The individual day/night models evaluate similarly to those of the XGBoost model. I find it odd that the $R^2$ scores are quite low for the individual models, but if I simply take the output and prediction vectors and concatenate them, then recalculate the combined $R^2$ score, the result is a much better score. I think that there is either some error in how I am performing the calculations, or something I don't understand about $R^2$. It seems like the individual $R^2$ scores somehow take into account the missing data, although I'm not sure how this is possible. I will have to look at the math to figure this out.
- There is little difference in performance between the combined day/night models and the baseline neural network (which incorporates all the data simultaneously).




Comparison of Random Parameter Search
The model selection notebook runs random parameter searches and saves the results of all parameter combinations. Since there may be several good (nearly equal) models identified by the parameter search, it makes sense to look at the parameter ranges associated with the well performing models, to ensure a value is chosen that corresponds to a robust range of model performance

The plots below show the parameters for each model, as a function of their associated $R^2$ score. This is a way to visualize the distribution of parameters, and more easily identify the range in which models tend to perform best. The colours represent the various data splits, and the individual points are plotted translucent, so that areas where there are many overlapping points can be identified by how dark they are. Note that all of the XGBoost models have a very high score, so the limits of the x-axis are zoomed to allow one to distinguish between parameter values.

Once a range of working values is known for each parameter, a grid search can then be used to gain some incremental improvement in performance. However, model selection doesn't have to stop there, as there are futher statistical tests that can be performed in choosing a model. Some of these test are outlined [here](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html). Although I leave the tuning here for this project, further tuning methods will be investigated in my Hockey Analytics project.

Conclusions and Lessons Learned
Visualizing the randomized parameter search results shows how robust the model is to certain parameter selections. There are three identifiable patterns to the parameter distributions:

1. There is a relatively even distribution of parameter values, and a relatively constant $R^2$ score associated with all values. This indicates that the parameter does not have a strong effect on model performance. However, there is still usually a trend that indicates what range the parameter should lie in. 
2. The best parameter is the same across most models, as with the "min_samples_split" parameter in the random forest model. Nearly all of the best performing models used a value of 0.2, so it is clear what the optimal choice is.
3. There is a clear correlation between the parameter value and performance. The best example of this is epsilon in the SVR model, where smaller values lead to better training scores. 

However, it should be noted that these parameters are evaluated using cross-validation, so there may be some effects due to over fitting. Therefore, in cases where there is not a clear consensus in what value of the parameter is best, one should lean towards regularization.
