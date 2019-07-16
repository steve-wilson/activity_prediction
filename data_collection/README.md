# Data Collection

Code used to collect data, and a list of the Twitter user IDs associated with the dataset.

## Contents

There are quite a few scripts here that were used for various purposes in collecting and processing the Twitter data. They are organized here by their general purpose.

### Getting data from Twitter

- `search.py` : contains functions to do things like:
    - find Tweets matching the queries defined by sets of human activities,
    - get user descriptions based on their profile IDs, and
    - get users' previous tweets based on their profile IDs.
- `valid_user_ids.txt` : a list of the "valid" user IDs used in the experiments from the paper (see `get_valid_user_list.py` below.

### Data wrangling and modification

- `convert_to_present.py` : converts activity phrases from past to present tense in order to match the format expected as input by the pretrained activity embedding models.
- `get_valid_user_list.py` : finds the set of users that are deemed "valid", that is, they have a non-empty profile, at least 5 tweets containing activities, and at least 10 total tweets.
- `make_datasets.py` : divide users into train/dev/test, and move their profiles, labels, and tweets into the correct directory structure so that they can be used in the prediction task.

### Augmenting data with additional features

- `compute_values_from_profiles.py` : compute the values scores for a given user based on their profile description (based on the Distributed Dictionaries Representation method).
- `get_labels_for_users.py` : assigns cluster labels to users based on activities contained in their tweets.
