# Nightwing's AI/ML Curriculum #

## Purpose ##

This repository holds the projects and course outline for the Nightwing AI/ML Curriculum. It covers a large range of the developments in AI and machine learning since 2017 including:

- Transformer models
- Diffusion models
- Data preprocessing and evaluation
- Model deployment, serving, and monitoring

## Course Outline ##

### Week 0: Data Collection, Organization, and Preprocessing ###
#### __Day 1-2: Data Collection:__ ####
Identify relevant datasets for ML tasks
Acquire datasets from sources like Kaggle, academic repositories, or web scraping.
Ensure data quality and relevance to chosen tasks.

#### Readings: ####
https://labelyourdata.com/articles/data-collection-methods-AI 
https://www.altexsoft.com/blog/data-collection-machine-learning/ 
https://medium.com/codex/how-to-collect-data-for-a-machine-learning-model-2b152752a15b 

#### Project:
Implement a data collection pipeline from a sample source.

#### __Day 3-7: Data Organization and Preprocessing:__ ####
Organize acquired data into appropriate formats (e.g., CSV, JSON).
Perform data cleaning tasks such as removing duplicates, handling missing values, and standardizing.
Split data into training, validation, and test sets.
Explore advanced preprocessing techniques such as handling imbalanced datasets, dealing with noisy text, and incorporating domain-specific knowledge.
Consider techniques like data augmentation to increase the diversity of training data.
Implement preprocessing pipelines using libraries like NLTK, spaCy, or scikit-learn.

#### Readings: ####
https://neptune.ai/blog/data-preprocessing-guide
https://towardsdatascience.com/introduction-to-data-preprocessing-in-machine-learning-a9fa83a5dc9d
https://www.geeksforgeeks.org/data-preprocessing-machine-learning-python/
https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9
https://www.datacamp.com/tutorial/complete-guide-data-augmentation

#### Project: ####
Implement a data preprocessing pipeline from sklearn, use advanced preprocessing techniques with data augmentation on a sample dataset

### Week 1: Foundations of NLP and Word Embeddings ###
#### __Day 1-2: Review basics of NLP: tokenization, stemming, lemmatization.__ ####

#### Readings: ####
https://neptune.ai/blog/tokenization-in-nlp
https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html

#### Project: ####
Recreate tokenizer following Karpathy video


#### __Day 3-4: Study word embeddings:__ ####
Understand Word2Vec, GloVe, and FastText.
Implement Word2Vec using Gensim or TensorFlow.

#### Readings: ####
https://towardsdatascience.com/word2vec-explained-49c52b4ccb71
https://nlp.stanford.edu/projects/glove/
https://www.analyticsvidhya.com/blog/2017/07/word-representations-text-classification-using-fasttext-nlp-facebook/

#### Project: ####
Implement word2vec for embeddings and understand the outputs


#### __Day 5-7: Explore recent advancements:__ ####
Contextual word embeddings (e.g., ELMo, BERT).
Study BERT architecture and pre-training objectives.

#### Readings: ####
https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

#### Project: ####
Implement BERT and play around with it


### Week 2: Transformer Models and Advanced NLP Techniques ###
#### __Day 1-3: Dive deeper into transformer models:__ ####
Understand self-attention mechanisms.
Study the Transformer architecture.

#### Readings: ####
https://arxiv.org/abs/1706.03762 - attention is all you need
https://www.turing.com/kb/brief-introduction-to-transformers-and-their-power
https://machinelearningmastery.com/the-transformer-attention-mechanism/

#### Project: ####
Follow Karpathy GPT video, rebuild GPT and play around with it


#### __Day 4-5: Explore advanced transformer-based models:__ ####
GPT (Generative Pre-trained Transformer) series.
XLNet, T5, and other variants.

#### Readings: ####
https://blog.research.google/2020/02/exploring-transfer-learning-with-t5.html
https://cameronrwolfe.substack.com/p/t5-text-to-text-transformers-part
https://towardsdatascience.com/what-is-xlnet-and-why-it-outperforms-bert-8d8fce710335

#### Project: ####
Implement a T5 model and play around with it

#### __Day 6-7: Learn about fine-tuning and distillation:__ ####
Techniques for fine-tuning pre-trained models for specific tasks.
Model distillation: compressing large models into smaller ones. - LoRA

#### Readings: ####
https://www.turing.com/resources/finetuning-large-language-models
https://towardsdatascience.com/understanding-lora-low-rank-adaptation-for-finetuning-large-models-936bce1a07c6
https://snorkel.ai/llm-distillation-techniques-to-explode-in-importance-in-2024/


#### Project: ####
Implement a model and finetune it with some data


### Week 3: Prompt Engineering Techniques ###
#### __Day 1-2: Task Formulation:__ ####
Clearly define the task objectives and desired model outputs.
Identify key prompts that provide necessary context and guidance to the model.
Consider the target audience, domain-specific requirements, and potential biases when formulating prompts.

#### Readings: ####
https://medium.com/academy-team/prompt-engineering-formulas-for-chatgpt-and-other-language-models-5de3a922356a
https://www.insidr.ai/advanced-guide-to-prompt-engineering/

#### Project: ####
Experiment with different prompt types


#### __Day 3-4: Prompt Design:__ ####
Craft prompts that effectively guide the model's responses towards desired behaviors.
Experiment with different prompt structures, including open-ended questions, fill-in-the-blank statements, or multiple-choice prompts.
Optimize prompt length, complexity, and specificity to elicit desired model behaviors.

#### Readings: ####
https://www.insidr.ai/advanced-guide-to-prompt-engineering/
https://www.altexsoft.com/blog/prompt-engineering/

#### Project: ####
Same as above, play around with different prompt types 


#### __Day 5-6: Advanced Prompt Engineering Techniques:__ ####

Explore advanced prompt engineering techniques such as RAG (Retrieval-Augmented Generation) and Chain-of-Thought:
Implement RAG to augment prompt-based generation with information retrieval, enhancing the model's contextual understanding and response coherence.
Explore Chain-of-Thought frameworks to create prompts that scaffold sequential thinking and reasoning processes within the model.
Evaluate the impact of RAG and Chain-of-Thought techniques on model performance and user interaction.

#### Readings: ####
https://www.promptingguide.ai/techniques/rag
https://www.smashingmagazine.com/2024/01/guide-retrieval-augmented-generation-language-models/
https://www.promptingguide.ai/techniques/cot

#### Project: ####
Implement a model with RAG and CoT prompting


#### __Day 7: Prompt Evaluation:__ ####
Evaluate the effectiveness of prompts through quantitative metrics and qualitative analysis.
Assess how well prompts guide the model towards desired outcomes and user expectations, considering the integration of RAG and Chain-of-Thought techniques.
Collect feedback from users and domain experts to iteratively improve prompt design and effectiveness.

#### Readings: ####
https://quickstarts.snowflake.com/guide/prompt_engineering_and_llm_evaluation/index.html#0
https://docs.humanloop.com/docs/evaluate-your-model
https://betterprogramming.pub/steering-llms-with-prompt-engineering-dbaf77b4c7a1

#### Project: ####
Implement a prompt engineering loop to iteratively gauge success


### Week 4: Advanced Topics in NLP and Generative AI ###
#### __Day 1-2: Study diffusion models:__ ####
Review concepts of probabilistic modeling and diffusion.
Understand recent diffusion models like DALL-E and CLIP.

#### Readings: ####
https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/
https://towardsdatascience.com/clip-model-and-the-importance-of-multimodal-embeddings-1c8f6b13bf72
https://openai.com/research/dall-e

#### Project: ####
Implement SD or DALLE and play around with it


#### __Day 3-4: Explore Mixture of Experts (MoE):__ ####
Understand the MoE architecture and its applications.
Study recent advancements and variants.

#### Readings: ####
https://machinelearningmastery.com/mixture-of-experts/
https://www.tensorops.ai/post/what-is-mixture-of-experts-llm

#### Project: ####
Implement an MoE model and play around with it


#### __Day 5-7: Hands-on projects and practical applications:__ ####
Implement a project using transformer-based models.
Experiment with fine-tuning, distillation, or diffusion models.

Project: Do something cool :-) 
