DS GA 1001: Intro to Data Science
Capstone Project
Team members: Anshika Gupta, Syed Hussain

## Introduction and data preprocessing
We’re given two datasets, spotify52k dataset and starRatings dataset which contain data on various different songs like the album name, genre, duration, valence, etc along with multiple user ratings given to the first 5k songs. 

We set the RNG seed to Syed’s N#17951972.

For data preprocessing, we observed duplicates in song track names wherein a song could have multiple songNumbers due to the change in either one of the other columns. We dropped the duplicates wherever required to maintain the standard of the results. To deal with the missing data, we filled the ‘NaN’ values in the starRating dataset with 0 to not introduce bias.

1. Is there a relationship between song length and popularity of a song? If so, is it positive or negative?
We analyze the dataset by creating histograms for song durations (in milliseconds and minutes) and popularity scores. Then we calculate and correlation coefficient between song duration (in minutes) and popularity, providing insights into the relationship between these two variables in the dataset. The obtained correlation is -0.054651 which suggests a very weak negative correlation. 

2. Are explicitly rated songs more popular than songs that are not explicit?
We conduct a Mann-Whitney U test to compare the mean popularity of songs marked as explicit and non-explicit in a Spotify dataset. The Mann-Whitney U test is chosen here because it doesn't assume normality in the data. 
The obtained P-value: 1.53e-19 is extremely small, indicating strong evidence against the null hypothesis. The Mann-Whitney U test suggests a significant difference in popularity between explicit and non-explicit songs. The test statistic (u = 139361273.5) supports the conclusion that explicit songs tend to have higher popularity in the Spotify dataset.

3. Are songs in major key more popular than songs in minor key?
The null hypothesis in your case states that there is no difference in the mean popularity between songs in major keys and songs in minor keys. Essentially, it suggests that any observed difference in mean popularity in your sample is due to random chance and not because of the key of the song.  The p-value you obtained from ttest(we are not dealing with median here, so decided to go ahead with t-test), approximately 1.66*10^−6, is a very small number. This p-value represents the probability of observing a difference as large as the one in your sample data (or larger) if the null hypothesis were true. Since this p-value is significantly lower than the common alpha level of 0.05 (or 5%), it indicates that the probability of observing such a difference due to random chance (if the null hypothesis were true) is extremely low. This leads you to reject the null hypothesis and accept the alternative hypothesis. In other words, the test suggests that there is a statistically significant difference in the mean popularity of songs in major keys compared to songs in minor keys, and this difference is not likely due to random chance.

4. Which of the following 10 song features: duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence and tempo predicts popularity best? How good is this model?
We perform linear regression analysis for each feature in the 'features' list to predict the 'popularity' of songs in a Spotify dataset. The graphs represent absolute values of the coefficients, RMSE values and R squared values obtained from the model. ‘Instrumentalness’ results as the best predictor on these metrics as we can see there’s dip in RMSE plot and spike in R2 plot for the same. Even after the obtained results, the model is not the best fit to predict popularity. We can further build a model with all the features in the following question 5. However, we suspect the absence of linearity in the model and hence we continue this process in the next question.





5. Building a model that uses *all* of the song features mentioned in question 1, how well can you predict popularity? How much (if at all) is this model improved compared to the model in question 4). How do you account for this? What happens if you regularize your model?

When comparing these metrics to the ones from the Lasso and Ridge regression models, it seems that all models are struggling to capture the underlying pattern in the data, as indicated by the low R² values across all models. The MSE for the non-regularized model is significantly higher than the RMSE of the Lasso and Ridge models, suggesting more pronounced errors in prediction.

These observations might suggest that:

The linear model, regardless of regularization, may not be suitable for your data. It could be that the relationship between the variables is not linear or there are interactions or non-linear patterns that the model is not capturing. Essential predictors or important variable interactions might be missing from the model. The data might be inherently noisy, or there could be issues with data quality. It may be beneficial to explore other types of models, especially ones that can capture non-linear relationships, such as decision trees, random forests, or neural networks, depending on the nature and amount of your data.

Without Regularization:  
Mean Squared Error: 445.68902465840165
R-squared: 0.0523541198267119

With Regularization:
Ridge Reg: RMSE 21.11171092235081
COD 0.05232155970166208

Lasso Reg: RMSE 21.11134824865985
COD 0.052354119363632545


6. When considering the 10 song features in the previous question, how many meaningful principal components can you extract? What proportion of the variance do these principal components account for? Using these principal components, how many clusters can you identify? Do these clusters reasonably correspond to the genre labels in column 20 of the data?




The number of meaningful principal components we got is 8 as shown in the plot above. And the portion of variance is 8-1 = 7. Using the price components, we applied the Silhouette method to determine the number of clusters = 2 as shown in the plot. As shown in the above plot “Distribution of Genres across Clusters”, the NMI came out to be 0.0785. The NMI shows that there’s low correlation and it could be because of the choice of features. It illustrates the distribution of song genres across different clusters, revealing that some clusters predominantly consist of one or two genres, indicating a degree of effective grouping by the clustering algorithm. However, other clusters show a uniform distribution of multiple genres, suggesting that the features used for clustering do not differentiate well between all genres. The presence of genres spread across multiple clusters could point to diversity within those genres or a lack of distinctive features that align with cluster boundaries. The lighter colors for certain genres imply fewer songs from those categories in the dataset or a less distinct feature set. This visualization, in conjunction with the previously mentioned NMI score of approximately 0.0785, confirms that while there is some correlation between clusters and genres, it is relatively weak, suggesting room for improvement in the clustering process, possibly by refining feature selection or the clustering methodology.

7. Can you predict whether a song is in major or minor key from valence using logistic regression or a support vector machine? If so, how good is this prediction? If not, is there a better one?
To predict whether a song is in major or minor key from valence using logistic regression first. Start with merging the datasets on the common song identifier and extracting relevant features i.e., valence and the target variable ‘major/minor’. Handle any missing values in the data by dropping them. We split the dataset into training and testing and train the logistic regression model. The following results were obtained:


The accuracy and precision are pretty low, hence we decided to try out SVM also. Following a similar approach as above, these are the results:


We can say the results are exactly the same and not satisfactory to be considered a good model.

As next steps, we try out other models such as,
GBM 


And random forest:

Determining the best model depends on our case. Here, we can say GBM performed better than others.

8. Can you predict genre by using the 10 song features from question 4 directly or the principal components you extracted in question 6 with a neural network? How well does this work?
Here we train a neural network model on a Spotify dataset with 10 song features to predict track genres. The dataset is split into training and testing sets, standardized, and used to train the model. The accuracy and a classification report are printed to evaluate the model's performance on the testing set. The model performs badly with an accuracy of 0.24 and equally worse on other evaluation metrics like precision and recall.

9. In recommender systems, the popularity based model is an important baseline. We have a two part question in this regard: a) Is there a relationship between popularity and average star rating for the 5k songs we have explicit feedback for? b) Which 10 songs are in the “greatest hits” (out of the 5k songs), on the basis of the popularity based model?

To explore the relationship between popularity and average star rating for the 5k songs with explicit feedback, we perform a correlation analysis. We calculate the Pearson correlation coefficient between the 'popularity' and 'avgstar_rating' columns. The correlation coefficient ranges from -1 to 1, where 1 indicates a perfect positive correlation, -1 indicates a perfect negative correlation, and 0 indicates no correlation. 

Correlation between Popularity and Average Star Rating: 0.57

For the top 10 tracks, as discussed before, we encountered duplicates with the same track name associated with multiple songNumbers. The following dataframe shows the top 10 unique track names obtained from the entire dataset. We did not remove any duplicates as described in the preprocessing stage.


10. You want to create a “personal mixtape” for all 10k users we have explicit feedback for. This mixtape contains individualized recommendations as to which 10 songs (out of the 5k) a given user will enjoy most. How do these recommendations compare to the “greatest hits” from the previous question and how good is your recommender system in making recommendations?
To generate personalized playlists for all the 10k users, we use cosine similarity to find the nearest neighbors for each user and recommend songs based on the preferences of similar users.
To compare these personalized recommendations with the "greatest hits" from the previous question, we calculate metrics such as precision, recall, or mean squared error. The following are the obtained results for the first 2 users:



The value ‘Popularity’ in the brackets shows the popularity score of that song on the general greatest hits lists.

We observe there exists  quite a huge difference in the top 10 for specific users and the popularity scores for generalized top songs.
