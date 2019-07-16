# Prediction

Code used to run the prediction experiments described in the paper. That is, given a user, their profile and previous tweets, predict which activity cluster they are likely to tweet about performing an activity from.

## Contents

### Training

- See `Modified_Infersent/`. This code starts with the based from Infersent, but it was modified to change the data loading, models, task, etc.
    - The main file to view here is `train_ap.py`, which runs the full training process using the rest of the files in this subdirectory.
    - `LICENSE` in this subdirectory contains the original Apache 2 License provided with Infersent, and all files that have been modified have been explicitely marked as such by Steve Wilson.

### Evaluation

- `evaluate_predictions.py` : gets evaluation scores based on the sets of predictions
- `get_baseline.py` : creates predictions for baseline models like majority, random guessing, etc., so that they can be passed through the `evalute_predictions.py` script along with the actual predictions.
