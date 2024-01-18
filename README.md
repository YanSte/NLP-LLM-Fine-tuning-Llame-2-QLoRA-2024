# | NLP | LLM | Fine-tuning | Llame 2 QLoRA |

## Natural Language Processing (NLP) and Large Language Models (LLM) with Fine-Tuning LLM and make Chatbot with LoRA and Flan-T5 

![Learning](https://t3.ftcdn.net/jpg/06/14/01/52/360_F_614015247_EWZHvC6AAOsaIOepakhyJvMqUu5tpLfY.jpg)

# <b>1 <span style='color:#78D118'>|</span> Overview</b>

In this notebook we're going to Fine-Tuning LLM:

<img src="https://github.com/YanSte/NLP-LLM-Fine-tuning-Trainer/blob/main/img_2.png?raw=true" alt="Learning" width="50%">

Many LLMs are general purpose models trained on a broad range of data and use cases. This enables them to perform well in a variety of applications, as shown in previous modules. It is not uncommon though to find situations where applying a general purpose model performs unacceptably for specific dataset or use case. This often does not mean that the general purpose model is unusable. Perhaps, with some new data and additional training the model could be improved, or fine-tuned, such that it produces acceptable results for the specific use case.

<img src="https://github.com/YanSte/NLP-LLM-Fine-tuning-Trainer/blob/main/img_1.png?raw=true" alt="Learning" width="50%">

Fine-tuning uses a pre-trained model as a base and continues to train it with a new, task targeted dataset. Conceptually, fine-tuning leverages that which has already been learned by a model and aims to focus its learnings further for a specific task.

It is important to recognize that fine-tuning is model training. The training process remains a resource intensive, and time consuming effort. Albeit fine-tuning training time is greatly shortened as a result of having started from a pre-trained model. 

<img src="https://github.com/YanSte/NLP-LLM-Fine-tuning-Trainer/blob/main/img_3.png?raw=true" alt="Learning" width="50%">

### The Power of Fine-Tuning: An Overview

Fine-tuning, a crucial aspect of adapting pre-trained models to specific tasks, has witnessed a revolutionary approach known as Low Rank Adaptation (LoRA). Unlike conventional fine-tuning methods, LoRA strategically freezes pre-trained model weights and introduces trainable rank decomposition matrices into the Transformer architecture's layers. This innovative technique significantly reduces the number of trainable parameters, leading to expedited fine-tuning processes and mitigated overfitting.

**Here some definitions:**
<br/>
<details>
  <summary><b>What is LoRA?</b></summary>

  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/0*kzZ2_LZqBO9_hTi3.png" alt="Learning" width="30%">

  LoRA represents a paradigm shift in fine-tuning strategies, offering efficiency and effectiveness. By reducing the number of trainable parameters and GPU memory requirements, LoRA proves to be a powerful tool for tailoring pre-trained large models to specific tasks. This article explores how LoRA can be employed to create a personalized chatbot.

  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*SJtZupeQVgp3s5HOBymcQw.png" alt="Learning" width="40%">

</details>
<br/>
<details>
  <summary><b>What is QLoRA?</b></summary>

  QLoRA (Quantized Low-Rank Adaptation) is an extension of LoRA (Low-Rank Adapters) that uses quantization to improve parameter efficiency during fine-tuning. QLoRA is more memory efficient than LoRA because it loads the pretrained model to GPU memory as 4-bit weights, compared to 8-bits in LoRA. This reduces memory demands and speeds up calculations.

  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/0*oV_KwvWnFYzuWzlz.png" alt="Learning" width="40%">

</details>
<br/>
<details>
  <summary><b>What is Model Quantization?</b></summary>

  Quantization is a technique used to reduce the size of large neural networks, including large language models (LLMs) by modifying the precision of their weights. Instead of using high-precision data types, such as 32-bit floating-point numbers, quantization represents values using lower-precision data types, such as 8-bit integers. This process significantly reduces memory usage and can speed up model execution while maintaining acceptable accuracy.
</details>
<br/>
<details>
  <summary><b>What is Bitsandbytes?</b></summary>

  Hugging Face’s Transformers library is a go-to choice for working with pre-trained language models. To make the process of model quantization more accessible, Hugging Face has seamlessly integrated with the Bitsandbytes library. This integration simplifies the quantization process and empowers users to achieve efficient models with just a few lines of code.

  <img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*O4RAzlQkbrcCPiPPD9JIYw.jpeg" alt="Learning" width="50%">

  ```BitsAndBytesConfig```

</details>
<br/>
<details>
  <summary><b>What is Loading a Model in 4-bit Quantization?</b></summary>

  One of the key features of this integration is the ability to load models in 4-bit quantization. This can be done by setting the `load_in_4bit=True` argument when calling the `.from_pretrained` method. By doing so, you can reduce memory usage by approximately fourfold.

  ```python
  load_in_4bit=True
  ```
</details>
<br/>
<details>
  <summary><b>What is Supervised fine-tuning?</b></summary>

  Supervised fine-tuning (SFT) is a key step in reinforcement learning from human feedback (RLHF). The TRL library from HuggingFace provides an easy-to-use API to create SFT models and train them on your dataset with just a few lines of code. It comes with tools to train language models using reinforcement learning, starting with supervised fine-tuning, then reward modeling, and finally proximal policy optimization (PPO).

  We will provide SFT Trainer the model, dataset, Lora configuration, tokenizer, and training parameters.

  ```SFTTrainer```
</details>
<br/>
<details>
  <summary><b>Trainer vs SFTTrainer</b></summary>  

  **Trainer:**
  - **General-purpose training:** Designed for training models from scratch on supervised learning tasks like text classification, question answering, and summarization.
  - **Highly customizable:** Offers a wide range of configuration options for fine-tuning hyperparameters, optimizers, schedulers, logging, and evaluation metrics.
  - **Handles complex training workflows:** Supports features like gradient accumulation, early stopping, checkpointing, and distributed training.
  - **Requires more data:** Typically needs larger datasets for effective training from scratch.


  **SFTTrainer:**
  - **Supervised Fine-tuning (SFT):** Optimized for fine-tuning pre-trained models with smaller datasets on supervised learning tasks.
  - **Simpler interface:** Provides a streamlined workflow with fewer configuration options, making it easier to get started.
  - **Efficient memory usage:** Uses techniques like parameter-efficient (PEFT) and packing optimizations to reduce memory consumption during training.
  - **Faster training:** Achieves comparable or better accuracy with smaller datasets and shorter training times than Trainer.


  **Choosing between Trainer and SFTTrainer:**
  - **Use Trainer:** If you have a large dataset and need extensive customization for your training loop or complex training workflows.
  - **Use SFTTrainer:** If you have a pre-trained model and a relatively smaller dataset, and want a simpler and faster fine-tuning experience with efficient memory usage.

</details>
<br/>

### Prompt Datasets

The utilization of chat prompts dataset during the fine-tuning of the model holds crucial significance due to several inherent advantages associated with the conversational nature of such data. Here is a more detailed explanation of using chat prompts in this context:

1. **Simulation of Human Interaction:** Chat prompts enable the simulation of human interactions, mirroring the dynamics of a real conversation. This approach facilitates the model's learning to generate responses that reflect the fluidity and coherence inherent in human exchanges.

2. **Contextual Awareness:** Chat prompts are essential for capturing contextual nuances in conversations. Each preceding turn of speech influences the understanding and generation of responses. The use of these prompts allows the model to grasp contextual subtleties and adjust its responses accordingly.

3. **Adaptation to Specific Language:** By incorporating chat prompts during fine-tuning, the model can adapt to specific languages, unique conversational styles, and even idiosyncratic expressions. This enhances the model's proficiency in generating responses that align with the particular expectations of end-users.

4. **Diversity in Examples:** Conversations inherently exhibit diversity, characterized by a variety of expressions, tones, and linguistic structures. Chat prompts inject this diversity into the training process, endowing the model with the ability to handle real-world scenarios and adapt to the richness of human interactions.

In summary, the use of chat prompts during the fine-tuning of a T5 model represents a potent strategy to enhance its capability in understanding and generating conversational texts. These prompts act as a bridge between training data and real-life situations, thereby strengthening the model's performance in applications such as chatbot response generation, virtual assistant systems, and other natural language processing tasks.


## Learning Objectives

By the end of this notebook, you will gain expertise in the following areas:

1. Learn how to effectively prepare datasets for training.
2. Understand the process of fine-tuning the Llama 2 on QLoRA with Trainer module.
3. Using Monitoring with WandB
