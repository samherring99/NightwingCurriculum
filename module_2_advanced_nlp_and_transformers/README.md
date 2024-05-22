# Module 2: Advanced NLP and Transformers

![alt text](https://github.com/samherring99/NightwingCurriculum/blob/main/module_2_advanced_nlp_and_transformers/image/image.jpeg)

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
