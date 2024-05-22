# Nightwing's AI/ML Curriculum #

## Purpose ##

- This repository contains the project-based learning and notes for the Nightwing AI/ML Curriculum.
- It is intended to be a higher-level refresher for a large range of the developments in AI and machine learning since 2017 and their uses in production today.
-  Upon completing this course, you should be prepared to utilize LLMs and diffusion models in your future projects.
-  This repo also serves as my collection of notes as I went through these implementations, most of which are based off one or more repositories I found (which are linked below)

## Course Outline ##

# Module 0: Data Collection

## Introduction

This module aims to cover best data collection practices for finding, retrieving, and preprocessing data for modern AI/ML tasks, specifically those with LLMs and diffusion models. After completing this, you should be able to organize and prepare a dataset for virtually any machine learning task rewuired today. This is meant to be a simple, higher-level overview, follow the code samples provided and the resources below.

## Projects

### Project 1 - Data Collection

Project 1 will be to implement a data collection script for a data source of your choosing. In the provided notebook, we are using Project Gutenberg to retrieve the full content of various books as a single text file. We will be using this website to gather data throughout this course.

#### Goals: 

- Identify relevant datasets for NLP tasks.
- Acquire datasets from sources like Kaggle, academic repositories, or web scraping.
- Ensure data quality and relevance to chosen tasks.

#### Readings:
- 📖 [Data Collection Methods](https://labelyourdata.com/articles/data-collection-methods-AI)
- 📖 [Data Collection Pt 2](https://www.altexsoft.com/blog/data-collection-machine-learning/)
- 📖 [How to collect data](https://medium.com/codex/how-to-collect-data-for-a-machine-learning-model-2b152752a15b)

#### Videos:
- 📺 [How is data prepared for ML?](https://www.youtube.com/watch?v=P8ERBy91Y90)
- 📺 [Data Collection Strategy - RapidAPI](https://www.youtube.com/watch?v=G7W1LzhbfGE)

💻 [Notebook](https://github.com/samherring99/NightwingCurriculum/blob/main/module_0_data_collection/module_0_project_1.ipynb)

### Project 2 - Data Preprocessing

Project 2 will be to implement a full data preprocessing pipeline using `sklearn`. The intent here is to learn how to use things like imputation and scaling to help normalize our data. The project aims to capture methods of handling categorical variables such as one hot encoding. This is why random data is chosen instead of a text file like we did in Project 1.

#### Goals: 

- Explore advanced preprocessing techniques such as handling imbalanced datasets, dealing with noisy text, and incorporating domain-specific knowledge.
- Consider techniques like data augmentation to increase the diversity of training data.
- Implement preprocessing pipelines using libraries like NLTK, spaCy, or scikit-learn.

### Readings:
- 📖 [TowardsDataScience](https://towardsdatascience.com/introduction-to-data-preprocessing-in-machine-learning-a9fa83a5dc9d)
- 📖 [GeeksForGeeks](https://www.geeksforgeeks.org/data-preprocessing-machine-learning-python/)
- 📖 [sklearn guide](https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9)
- 📖 [Data Augmentation](https://www.datacamp.com/tutorial/complete-guide-data-augmentation)

#### Videos:
- 📺 [Preprocessing](https://www.youtube.com/watch?v=4i9aiTjjxHY)
- 📺 [Preprocessing Pt 2](https://www.youtube.com/watch?v=h1BnRBzYjYY)
- 📺 [sklearn preprocessing pipeline](https://www.youtube.com/watch?v=ZddUwo4R5ug)

💻 [Notebook](https://github.com/samherring99/NightwingCurriculum/blob/main/module_0_data_collection/module_0_project_2.ipynb)

## Conclusion

Now that this module is completed, you should have a solid understanding of different sources to collect data for machine learning, different methods of organizing and preprocessing data, and ways to optimize this process in future efforts.

# Module 1: Natural Language Processing Basics

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

# Module 2: Advanced NLP and Transformers

## Introduction

This module aims to cover advanced techniques in NLP and the Transformer architecture in language models, including underlying mechanisms like attention and residual connections. By the end of this module, you should be able to implement basic versions of GPT, T5, and MoE models, as well as getting started with PeFT (Parameter Efficient Finetuning), SFT (Supervised Finetuning), and LoRA (Low Rank Adaptation) for finetuning models.

## Projects

### Project 1 - Generative Pretrained Transformers

Project 1 is to recreate nanoGPT following Karpathy's video and [exercises](https://github.com/karpathy/ng-video-lecture). After this is completed, you should understand the inner working of the Transformer architecture well enough to implement from scratch.

#### Goals: 

- Understand self-attention mechanisms
- Study the Transformer architecture

#### Readings:
- 📖 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 📖 [Transformers Introduction](https://www.turing.com/kb/brief-introduction-to-transformers-and-their-power)
- 📖 [Attention Mechanism](https://machinelearningmastery.com/the-transformer-attention-mechanism/)

#### Videos:
- 📺 [Karpathy GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

💻 [Notebook](https://github.com/samherring99/NightwingCurriculum/blob/main/module_2_advanced_nlp_and_transformers/module_2_project_1.ipynb)

### Project 2 - Text-to-text Transfer Transformers

Project 2 is to implement T5 (text-to-text transfer transformers) from scratch. This project illustrates how to build an encoder-decoder based language model, and the differences between this and a decoder-only architecture like GPT.

#### Goals: 

- Explore XLNet, T5, and other variants
- Understand their arcitectures and differences from classic GPT

#### Readings:
- 📖 [XLNet](https://towardsdatascience.com/what-is-xlnet-and-why-it-outperforms-bert-8d8fce710335)
- 📖 [T5 Introduction](https://blog.research.google/2020/02/exploring-transfer-learning-with-t5.html)
- 📖 [T5 Deep Dive](https://cameronrwolfe.substack.com/p/t5-text-to-text-transformers-part)

#### Videos:
- 📺 [T5 Continued](https://www.youtube.com/watch?v=91iLu6OOrwk)

💻 [Notebook](https://github.com/samherring99/NightwingCurriculum/blob/main/module_2_advanced_nlp_and_transformers/module_2_project_2.ipynb)

### Project 3 - Mixture of Experts

Project 3 is to implement a simple MoE (Mixture of Experts) model from scratch. After completing this project, you should be able to understand the benefits and optimizations brought from replacing the classical FFN layer with an MoE layer, as well as how to implement MoE in future model architectures.

#### Goals: 

- Understand the MoE architecture and its applications.
- Study recent advancements and variants.

#### Readings:
- 📖 [Mixture of Experts Introduction](https://machinelearningmastery.com/mixture-of-experts/)
- 📖 [Mixture of Experts Explained](https://www.tensorops.ai/post/what-is-mixture-of-experts-llm)
- 📖 [Mixture of Experts Paper](https://arxiv.org/abs/1312.4314)

#### Videos:
- 📺 [Mixture of Experts Explained](https://www.youtube.com/watch?v=mwO6v4BlgZQ)

💻 [Notebook](https://github.com/samherring99/NightwingCurriculum/blob/main/module_2_advanced_nlp_and_transformers/module_2_project_3.ipynb)

### Project 4 - Finetuning

Project 3 is to learn finetuning techniques with modern LLMs and their benefits. The readings cover PeFT and LoRA in depth, and the notebooks illustrate how they are implemented in theory and in practice.

#### Goals: 

- Learn techniques for fine-tuning pre-trained models for specific tasks
- Understand model distillation: compressing large models into smaller ones
- Parameter Efficient Finetuning - PeFT
- Learn how LoRA works abnd how to implement it from scratch

#### Readings:
- 📖 [Finetuning Introduction](https://www.turing.com/resources/finetuning-large-language-models)
- 📖 [LoRA](https://towardsdatascience.com/understanding-lora-low-rank-adaptation-for-finetuning-large-models-936bce1a07c6)
- 📖 [LLM Distillation](https://snorkel.ai/llm-distillation-techniques-to-explode-in-importance-in-2024/)

#### Videos:
- 📺 [Finetuning with examples](https://www.youtube.com/watch?v=eC6Hd1hFvos)

💻 [Notebook Part 1 - CPU](https://github.com/samherring99/NightwingCurriculum/blob/main/module_2_advanced_nlp_and_transformers/module_2_project_4_1.ipynb)
💻 [Notebook Part 2 - GPU](https://github.com/samherring99/NightwingCurriculum/blob/main/module_2_advanced_nlp_and_transformers/module_2_project_4_2.ipynb)

### Conclusion:

Now that you have completed this module, you should understand GPT, T5, MoE models, and finetuning techniques enough to re-use them across your next projects. These concepts are foundational to modern LLMs and understanding their uses and differences is important!

# Module 3: Prompt Engineering

## Introduction

This module aims to cover the different techniques used in prompt engineering at a high and low level. The below projects cover basic prompt engineering, RAG (retrieval augmented generation), CoT (Chain of Thought), ReAct (Reason + Action), and the DSPy prompt programming framework. Once you have completed this module you should fully understand the state of the art in prompt engineering, how to apply these technniques to your own prompts, and the benefits and drawbacks of each method.

## Projects

### Project 1 - Basic Prompt Engineering

Project 1 is basic prompt engineering. The goal is to create a list of prompts, iterate over them, and prompt an LLM. This is to show an introduciton to automated prompting, which we will build on later.

#### Goals: 

- Clearly define the task objectives and desired model outputs.
- Identify key prompts that provide necessary context and guidance to the model.
- Consider the target audience, domain-specific requirements, and potential biases when formulating prompts.

#### Readings:
- 📖 [Prompt Engineering Basics](https://medium.com/academy-team/prompt-engineering-formulas-for-chatgpt-and-other-language-models-5de3a922356a)
- 📖 [Prompt Engineering Continued](https://www.insidr.ai/advanced-guide-to-prompt-engineering/)

#### Videos:
- 📺 [Prompt Engineering Explained](https://www.youtube.com/watch?v=BzIF4hrEgyk)

💻 [Notebook](https://github.com/samherring99/NightwingCurriculum/blob/main/module_3_prompt_engineering/module_3_project_1.ipynb)

### Project 2 - Retrieval Augmented Generation

Project 2 is to implement a basic RAG (Retrieval Augmented Generation) pipeline with a PDF document to illustrate how to build a knowledge base for accurate LLM inference. After completing this project, you should understand how RAG works and different use cases for it.

#### Goals: 

- Understand techniques to improve LLM output so it remains relevant, accurate, and useful in various contexts.
- Implement RAG to augment prompt-based generation with information retrieval, enhancing the model's contextual understanding and response coherence.

#### Readings:
- 📖 [RAG Explained](https://www.smashingmagazine.com/2024/01/guide-retrieval-augmented-generation-language-models/)
- 📖 [RAG Continued](https://www.promptingguide.ai/techniques/rag)

#### Videos:
- 📺 [Prompt Engineering Continued](https://www.youtube.com/watch?v=1c9iyoVIwDs)
- 📺 [RAG Part 1](https://www.youtube.com/watch?v=2uMuqD4UvkA&pp=ygUecmV0cmlldmFsIGF1Z21lbnRlZCBnZW5lcmF0aW9u)
- 📺 [RAG Part 2](https://www.youtube.com/watch?v=XctooiH0moI)

💻 [Notebook](https://github.com/samherring99/NightwingCurriculum/blob/main/module_3_prompt_engineering/module_3_project_2.ipynb)

### Project 3 - Reason + Action = ReAct

Project 3 is to implement a basic ReAct prompt framework that breaks the model's answer into Thought, Observation, and Action. After completing this project, you should understand how to incorporate ReAct into your prompting and for what tasks this framework would be desirable.

#### Goals: 

- Explore advanced prompt engineering techniques such as ReAct and Chain-of-Thought
- Explore Chain-of-Thought frameworks to create prompts that scaffold sequential thinking and reasoning processes within the model.
- Evaluate the impact of Chain-of-Thought techniques and tool usage with LangChain model performance and user interaction.

#### Readings:
- 📖 [Chain Of Thought](https://www.promptingguide.ai/techniques/cot)
- 📖 [Advanced Prompt Engineering](https://www.altexsoft.com/blog/prompt-engineering/)
- 📖 [ReAct](https://www.promptingguide.ai/techniques/react)
- 📖 [ReAct Part 2](https://medium.com/@jainashish.079/build-llm-agent-combining-reasoning-and-action-react-framework-using-langchain-379a89a7e881)
- 📖 [LangChain ReAct Docs](https://python.langchain.com/docs/modules/agents/agent_types/react)
- 📖 [LlamaIndex ReAct Agent](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/)

#### Videos:
- 📺 [CoT Explained](https://www.youtube.com/watch?v=b210W3JWOxw)
- 📺 [Advanced Prompt Engineering Continued](https://www.youtube.com/watch?v=j320H2LFx-U)

💻 [Notebook](https://github.com/samherring99/NightwingCurriculum/blob/main/module_3_prompt_engineering/module_3_project_3.ipynb)

### Project 4 - DSPy

Project 4 is a simple example of DSPy to get started with understanding the framework. After completing this project, you shoul dbe able to understand what DSPy is, what makes it special, and how to implemebnt basic uses of it.

#### Goals: 

- Explore advanced prompt tuning techniques and their impact on top of CoT
- Implement DSPy as another implementation of a RAG/CoT based retrieval system

#### Readings:
- 📖 [DSPy Github](https://github.com/stanfordnlp/dspy) 
- 📖 [QDrant DSPy Docs](https://qdrant.tech/documentation/frameworks/dspy/)
- 📖 [DSPy Website](https://dspy-docs.vercel.app/)
- 📖 [DSPy Paper](https://arxiv.org/abs/2310.03714)

#### Videos:
- 📺 [DSPy Overview](https://www.youtube.com/watch?v=njVKMqs9lxU)
- 📺 [DSPy Explained](https://www.youtube.com/watch?v=41EfOY0Ldkc)

💻 [Notebook](https://github.com/samherring99/NightwingCurriculum/blob/main/module_3_prompt_engineering/module_3_project_4.ipynb)

### Project 5 - Automated Evaluation

Project 5 is the culmination of all the last projects in this module. The goal is to implement an advanced, automated prompt iteration and evaluation pipeline using DSPy. The provided notebook illustrates how this would be done for an automated coding assistant, with steps planned out as Signatures and clearly explained. After  ompleting this project, you should be fully versed in all main techniques used for prompt engineering lately and their differences, as well as how to implent these for future projects.

#### Goals: 

- Evaluate the effectiveness of prompts through quantitative metrics and qualitative analysis
- Assess how well prompts guide the model towards desired outcomes and user expectations
- Consider the integration of RAG and Chain-of-Thought techniques
- Collect feedback from users and domain experts to iteratively improve prompt design and effectiveness

#### Readings:
- 📖 [LLM Evaluation](https://quickstarts.snowflake.com/guide/prompt_engineering_and_llm_evaluation/index.html#0)
- 📖 [LLM Evaluation Cnntinued](https://docs.humanloop.com/docs/evaluate-your-model)
- 📖 [Iterative Prompt Engineering](https://betterprogramming.pub/steering-llms-with-prompt-engineering-dbaf77b4c7a1)
- 📖 [LangChain DSPy](https://python.langchain.com/docs/integrations/providers/dspy)

#### Videos:
- 📺 [Iterative Prompt Engineering](https://www.youtube.com/watch?v=1c9iyoVIwDs)

💻 [Notebook](https://github.com/samherring99/NightwingCurriculum/blob/main/module_3_prompt_engineering/module_3_project_5.ipynb)

### Conclusion:

Now that you have completed this module, you should fully understand the meaning of prompt engineering and different techniques used in production to validate and improve outputs. 

# Module 4: Language-Image Pretraining and Diffusion Models

## Introduction

This module aims to cover LIP (language-image pretraining) and diffusion models at a high level to introduce multimodal modeling. At the end of this project, you should be able to utilize and implement CLIP and other LIP models in your projects, as well as the basics of diffusion modeling to generate images from text. Note, this is only a high level overview, while both CLIP and StableDiffusion are implemented from scratch here, there is more to cover in depth.

## Projects

### Project 1 - Contrastive Language Image Pretraining

Project 1 is to implement CLIP from scratch, using a Flickr images dataset to train the embedding model. After this project is completed, you should understand the uses and internals of any language image pretraining model, and should be able to re-use them in future projects. This implementation is based off [this repo](https://github.com/moein-shariatnia/OpenAI-CLIP).

#### Goals: 

- Review concepts of probabilistic modeling and diffusion
- Understand embeddings like CLIP

#### Readings:
- 📖 [OpenAI CLIP](https://openai.com/research/clip)
- 📖 [Multimodal Embeddings](https://towardsdatascience.com/clip-model-and-the-importance-of-multimodal-embeddings-1c8f6b13bf72)

#### Videos:
- 📺 [OpenAI CLIP Explained](https://www.youtube.com/watch?v=T9XSU0pKX2E)

💻 [Notebook](https://github.com/samherring99/NightwingCurriculum/blob/main/module_4_diffusion_models/module_4_project_1.ipynb)

### Project 2 - Diffusion Models

Project 2 is to implement a basic form of StableDiffusion from scratch following [this tutorial](https://levelup.gitconnected.com/building-stable-diffusion-from-scratch-using-python-f3ebc8c42da3). After this project is completed, you should understand the basics of diffusion modeling, and how to implement these techniques into future projects.

#### Goals: 

- Review the architecture of Stable Diffusion and how CLIP could be involved
- Understand how diffusion models are constructed and utilized

#### Readings:
- 📖 [OpenAI DALLE](https://openai.com/research/dall-e)
- 📖 [StableDiffusion Illustrated](https://jalammar.github.io/illustrated-stable-diffusion/)
- 📖 [Diffusion Models Introduction](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)
- 📖 [Harvard StableDiffusion Tutorial](https://scholar.harvard.edu/sites/scholar.harvard.edu/files/binxuw/files/stable_diffusion_a_tutorial.pdf)

#### Videos:
- 📺 [Computerphile AI Image Generation](https://www.youtube.com/watch?v=1CIpzeNxIhU)
- 📺 [StableDiffusion Explained](https://www.youtube.com/watch?v=RGBNdD3Wn-g)

💻 [Notebook](https://github.com/samherring99/NightwingCurriculum/blob/main/module_4_diffusion_models/module_4_project_2.ipynb)

### Conclusion:

Now that you have completed this module, you should understand the importance of multimodal embeddings, the uses and improvements it offers, as well as the techniques used for diffusion modeling and why this works. You should also understand how to re use these project implementations for future work.

