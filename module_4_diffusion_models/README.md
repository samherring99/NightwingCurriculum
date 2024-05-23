# Module 4: Language-Image Pretraining and Diffusion Models

![alt text](https://learnopencv.com/wp-content/uploads/2023/02/stable-diffusion-high-level-working.png)

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
