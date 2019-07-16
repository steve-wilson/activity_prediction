# Human Activity Prediction
Code and pretrained models for predicting human activities using language features from Twitter data.

The code is this repository was used to proeduce the data and results from this paper:
> "Predicting Human Actities from User-Generated Content" 
> Steven R. Wilson and Rada Mihalcea. 
> ACL 2019.

There is a copy of the paper in this repository in the file called `Wilson_ACL_2019.pdf`.

## Contents

Separate README files in each subdirectory will describe the files included there.

### [/data_collection](./data_collection)

Code used to collect data, and a list of the Twitter user IDs associated with the dataset.

### [/clustering](./clustering)

Code used to embed and cluster the activity phrases along with sample clustering results with different values of k (for k-means).

### [/prediction](./prediction)

Code used to run the prediction experiments described in the paper. That is, given a user, their profile and previous tweets, predict which activity cluster they are likely to tweet about performing an activity from.
