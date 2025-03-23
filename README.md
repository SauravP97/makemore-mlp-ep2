# makemore - A character level Language Model
## Episode 2: Multi-Layer Perceptron (MLP) implementation

Makemore is an auto-regressive character-level language model. It can take a text file as an input, where each line is assumed to be one training example and it generates more examples like it. New and Unknown...

This repo covers the work of Andrej's [makemore](https://github.com/karpathy/makemore/) repo with a more detailed documentation of each implementations. :smile:

This is the 2nd repo in the series where we are covering multiple implementations of **makemore**. In our previous episode we covered the [**Bigram Model**](https://github.com/SauravP97/makemore) through **Probabilities Count** and **Neural Network** implementation.We also discussed how the calculated Loss for both the implementations converged to a similar value.

The best loss achieved through our Bigram model was around `2.5` which was not that great. Hence sampling more words from that implementation sadly resulted in the meaning-less series of letters.

This time we plan to do better!

We will use the `Multi-Layer Perceptron` to build our probabilistic model to sample words and this time instead of keeping our context window of `1` word, we will increase it to `3` words.

The approach we will follow has been covered in this paper: MLP, following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).


## Implementation

To implement our model, we will need a training dataset. We build our training dataset with the help of a collection of [names](https://github.com/SauravP97/makemore/blob/master/datasets/names.txt).

***Goal The end goal is to generate / make more names similar to those present in the names dataset, but potentially never seen before in the dataset.***

Our current names dataset with approx 32K examples looks somewhat like this.

```
emma
olivia
ava
isabella
sophia
charlotte
...
```

As mentioned earlier, we will use a block-size of `3` in this approach. The context window of 3 block size will help the model to predict the next letter. This is how the datasets will look like.

```
emma
... ---> e
..e ---> m
.em ---> m
emm ---> a
mma ---> .

olivia
... ---> o
..o ---> l
.ol ---> i
oli ---> v
liv ---> i
ivi ---> a
via ---> .
```

In our dataset we will transform all these 27 characters to their indices with the help of a map.

![map](/media/makemore-2-1.png)

We will use the above mapping to represent characters everywhere in our model. Let's also visualize how our initial character friendly dataset can be mapped to the actual set.

![dataset mapping](/media/makemore-2-2.png)

We will build a character embedding vector which will have 2 embeddings for each character. Hence our embedding vector will be a `27 x 2` matrix.

```python
# embedding vector: C
C = torch.rand((27, 2))
```

```
[       
    [0.3229, 0.7459],
    [0.3204, 0.6386],
    [0.7159, 0.8892],
    [0.4156, 0.7040],
    [0.1160, 0.0299],
    [0.8790, 0.1160],
    [0.6731, 0.6735],
    [0.6363, 0.6873],
    [0.6101, 0.0808],
    [0.1732, 0.2006],
    [0.0301, 0.8121],
    [0.9094, 0.5033],
    [0.3327, 0.8181],
    [0.3731, 0.8226],
    [0.2618, 0.3873],
    [0.7068, 0.2900],
    [0.2119, 0.2677],
    [0.5422, 0.5946],
    [0.4434, 0.9632],
    [0.9237, 0.9008],
    [0.1595, 0.8789],
    [0.0381, 0.0772],
    [0.2151, 0.0597],
    [0.6294, 0.0514],
    [0.1176, 0.2957],
    [0.4325, 0.8694],
    [0.9811, 0.4930]
]
```

Now, we will index our dataset with the embeddings so that each character in the dataset will be transformed into their 2 dimensional embedding values.

```python
C[X]
```

Our embedded vector will be of dimension `len(X) x 3 x 2` since each character in the training dataset will now have 2 embedding values picked from the embedding map `C`.

![Embedding Vector](/media/makemore-2-3.png)
