# dnt-semantic-similarity

*Much of the code in this subdirectory here was written/adapted by Li Zhang in 2018*

## Acknowledgements
The codes in /InferSent-master are modified from the original [InferSent](https://github.com/facebookresearch/InferSent) implementation copyrighted by Facebook, Inc.

## Dependencies

This code is written in python 3. Dependencies include:

* Python 3
* [Pytorch](http://pytorch.org/) (recent version)
* NLTK >= 3

## Examples

To load models using InferSent+NT on Human Activities, for example, on the SIM relation:
1. Put model file (i.e. sim_nt.pt) in /InferSent-master
2. In /InferSent-master/dataset run
```
./get_data.bash 
```
to download word embedding file

3. In /DNT, run python DNT.py --model infersent --dataset activities --dimension 1 --transfer NT --save no --load_model sim_nt.pt

This would evaluate the pre-trained model on the Human Activities validation and test set. To produce embeddings, modify the following code in /InferSent-master/train_nli.py:
```python
# model forward
outputs = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
```

## Note for extension for Activity Prediction work

`train_nli_custom.py` is a modified version of `train_nli.py` which will also allow for accessing the hidden states of the network, and leverages some other changes to the model to do things like adding an additional layer that reduces the dimensionality of that hidden state in order to make it easier to work with. -Steve Wilson
