# Sentiment Analysis of Movie Reviews: A Comparison of Deep Learning Architectures and Embeddings

By Louis Magowan (@louismagowan), Aashrit Ahuja and Alberto Agudo (@alberto-agudo).

A comparison of 80 combinations of deep learning models, embeddings, regularization and batch normalization layers for sentiment analysis on the Large Movie Reviews Dataset.

**1. What is the problem that you want to solve?** 

The explosion in the availability of text data has made sentiment analysis an increasingly important and necessary task. Using various deep learning architectures, we aim to develop a classifier that is capable of determining whether a movie review is positive or negative, with the hope of generalising its application into other domains.

**2. What deep learning methodologies do you plan to use in your project?** 

We will look to create a baseline model using MLP/CNN/RNNs and then look to build upon this by employing more complex constructions such as LSTMs or GRUs. This will allow us to compare different architectures and determine which would be optimal in this context, compared with other literature benchmarks.

**3. What dataset will you use? Provide information about the dataset, and a url for the dataset if available. Briefly discuss suitability of the dataset for your problem.** 

We will rely on the Stanford AI Large Movie Reviews dataset. The dataset is available [here](https://ai.stanford.edu/~amaas/data/sentiment/). It is comprised of 50,000 highly polar movie reviews.

We consider that this dataset is suitable for the purpose of classifying movie reviews. The first reason for this is that there is plenty of data, since there is a good number of reviews which include the text (around 120 MB of data). On the other hand, every review is already labeled as positive or negative by virtue of their score out of 10, so there is no need for manually labelling the reviews. Secondly, the classes are balanced, with 25,000 positive reviews and 25,000 negative reviews.

The Readme file of the dataset specifies that it was compiled by these authors: Maas, A., Daly, R., Pham, P., Huang, D., Ng, A. and Potts, C., 2011. Learning Word Vectors for Sentiment Analysis. In: Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies. Portland, Oregon, USA: Association for Computational Linguistics, pp.142-150.

**4. List key references (e.g. research papers) that your project will be based on?** 
- Maas, A., Daly, R., Pham, P., Huang, D., Ng, A. and Potts, C., 2011. Learning Word Vectors for Sentiment Analysis. In: Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies. Portland, Oregon, USA: Association for Computational Linguistics, pp.142-150. [URL](https://www.aclweb.org/anthology/P11-1015)

- Potts, Christopher. 2011. On the negativity of negation. In Nan Li and David Lutz, eds., Proceedings of Semantics and Linguistic Theory 20, 636-659. [URL](https://semanticsarchive.net/Archive/2M2NTY0O/potts-salt20-negation.pdf)
