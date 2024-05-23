# Module 0: Data Collection

![alt text](https://images.prismic.io/turing/65980bec531ac2845a272689_Machine_Learning_4_11zon_0415e7dfea.webp?auto=format,compress)

## Introduction

This module aims to cover best data collection practices for finding, retrieving, and preprocessing data for modern AI/ML tasks, specifically those with LLMs and diffusion models. After completing this, you should be able to organize and prepare a dataset for virtually any machine learning task rewuired today. This is meant to be a simple, higher-level overview, follow the code samples provided and the resources below.

## Projects

### Project 1 - Data Collection

Project 1 will be to implement a data collection script for a data source of your choosing. In the provided notebook, we are using Project Gutenberg to retrieve the full content of various books as a single text file. We will be using this website to gather data throughout this course.

#### Goals: 

- Identify relevant datasets for NLP tasks
- Acquire datasets from sources like Kaggle, academic repositories, or web scraping
- Ensure data quality and relevance to chosen tasks

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

- Explore advanced preprocessing techniques such as handling imbalanced datasets, dealing with noisy text, and incorporating domain-specific knowledge
- Consider techniques like data augmentation to increase the diversity of training data
- Implement preprocessing pipelines using libraries like NLTK, spaCy, or scikit-learn

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
