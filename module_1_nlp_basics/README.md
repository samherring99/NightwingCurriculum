# Module 1: Natural Language Processing Basics

![alt text](https://codesrevolvewordpress.s3.us-west-2.amazonaws.com/revolveai/2022/05/15110810/natural-language-processing-techniques.png)

## Introduction

This module aims to cover the basics of NLP (natural language processing), covering tokenization, vector embeddings, and simple language modeling. At the end of this module, you should fully understand tokenizers, word2vec, and the BERT language model, as well as how to implement them from scratch in the provided notebooks.

## Projects

### Project 1 - Tokenization

Project 1 is to rebuild the GPT tokenizer, following Karpathy's (very insightful) video and [exercises](https://github.com/karpathy/minbpe/blob/master/exercise.md). The video and repository covers everything in depth, much better than I could here, and the accompanying readings cover lesser discussed pieces like stemming and lemmatization.

#### Goals: 

- Preprocess text data by tokenizing, removing stop words, and performing stemming or lemmatization as necessary.
- Understand the basics and purpose of tokenization in NLP
- Understabnd basic spproaches to stemming and lemmatization
- Understand the different approaches to tokenization

#### Readings:
- 📖 [Tokenization in NLP](https://neptune.ai/blog/tokenization-in-nlp)
- 📖 [Standford Stemming Lemmatization lecture](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)

#### Videos:
- 📺 [NLP Basics](https://www.youtube.com/watch?v=8F_ERPqN0T0)
- 📺 [Text preprocessing](https://www.youtube.com/watch?v=hhjn4HVEdy0)
- 📺 [Karpathy GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)

💻 [Notebook](https://github.com/samherring99/NightwingCurriculum/blob/main/module_1_nlp_basics/module_1_project_1.ipynb)

### Project 2 - Word2Vec

Project 2 is to implement word2vec from scratch, and to fully understand the importance and use of vector embeddings for semantic similarity. This piece is integral to understanding why language models like GPT work the way they do. Below readings also cover GloVe and FastText, alternative vector embedding techniques.

#### Goals: 

- Understand Word2Vec, GloVe, and FastText.
- Undertsand the importance of vector embeddings and similarity
- Implement Word2Vec from scratch using [this tutoria](https://towardsdatascience.com/word2vec-with-pytorch-implementing-original-paper-2cd7040120b0
)

#### Readings:
- 📖 [TowardsDataScience](https://towardsdatascience.com/word2vec-explained-49c52b4ccb71)
- 📖 [GloVe Stanford](https://nlp.stanford.edu/projects/glove/)

#### Videos:
- 📺 [Word2Vec Explained](https://www.youtube.com/watch?v=UqRCEmrv1gQ)

💻 [Notebook](https://github.com/samherring99/NightwingCurriculum/blob/main/module_1_nlp_basics/module_1_project_2.ipynb)

### Project 3 - BERT

Project 3 is to implement BERT (Bi-directional Encoder Representations from Transformers) from scratch, and to fully understand the usage of contextual word embeddings in language models.

#### Goals: 

- Understand contextual word embeddings (e.g., ELMo, BERT).
- Study BERT architecture and pre-training objectives.

#### Readings:
- 📖 [TowardsDataScience](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
- 📖 [BERT Paper](https://arxiv.org/abs/1810.04805)

#### Videos:
- 📺 [BERT Explained](https://www.youtube.com/watch?v=xI0HHN5XKDo)

💻 [Notebook](https://github.com/samherring99/NightwingCurriculum/blob/main/module_1_nlp_basics/module_1_project_3.ipynb)

### Conclusion:

Now that you have completed this module, you should be able to implement basic tokenizers, vector embedding methods, and simple encoder based language models with full understanding of their internals.
