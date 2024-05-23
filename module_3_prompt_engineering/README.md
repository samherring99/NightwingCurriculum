# Module 3: Prompt Engineering

![alt text](https://media.licdn.com/dms/image/D5612AQExQRgbkb_S2w/article-inline_image-shrink_400_744/0/1693748745179?e=1721260800&v=beta&t=1iema6XN8Z1wI0AdBS4iKkAqBD7pAPBYHdfgHOcUvkk)

## Introduction

This module aims to cover the different techniques used in prompt engineering at a high and low level. The below projects cover basic prompt engineering, RAG (retrieval augmented generation), CoT (Chain of Thought), ReAct (Reason + Action), and the DSPy prompt programming framework. Once you have completed this module you should fully understand the state of the art in prompt engineering, how to apply these technniques to your own prompts, and the benefits and drawbacks of each method.

## Projects

### Project 1 - Basic Prompt Engineering

Project 1 is basic prompt engineering. The goal is to create a list of prompts, iterate over them, and prompt an LLM. This is to show an introduciton to automated prompting, which we will build on later.

#### Goals: 

- Clearly define the task objectives and desired model outputs
- Identify key prompts that provide necessary context and guidance to the model
- Consider the target audience, domain-specific requirements, and potential biases when formulating prompts

#### Readings:
- 📖 [Prompt Engineering Basics](https://medium.com/academy-team/prompt-engineering-formulas-for-chatgpt-and-other-language-models-5de3a922356a)
- 📖 [Prompt Engineering Continued](https://www.insidr.ai/advanced-guide-to-prompt-engineering/)

#### Videos:
- 📺 [Prompt Engineering Explained](https://www.youtube.com/watch?v=BzIF4hrEgyk)

💻 [Notebook](https://github.com/samherring99/NightwingCurriculum/blob/main/module_3_prompt_engineering/module_3_project_1.ipynb)

### Project 2 - Retrieval Augmented Generation

Project 2 is to implement a basic RAG (Retrieval Augmented Generation) pipeline with a PDF document to illustrate how to build a knowledge base for accurate LLM inference. After completing this project, you should understand how RAG works and different use cases for it.

#### Goals: 

- Understand techniques to improve LLM output so it remains relevant, accurate, and useful in various contexts
- Implement RAG to augment prompt-based generation with information retrieval, enhancing the model's contextual understanding and response coherence

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
- Explore Chain-of-Thought frameworks to create prompts that scaffold sequential thinking and reasoning processes within the model
- Evaluate the impact of Chain-of-Thought techniques and tool usage with LangChain model performance and user interaction

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
