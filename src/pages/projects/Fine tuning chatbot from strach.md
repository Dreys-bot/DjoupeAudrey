We will see how to generate text with models based on the Transformers architecture, and we will use this knowledge to demonstrate how to create fake news. The objective is to demonstrate the operation and use of these models through this practical example.

First, we will present a theoretical introduction to text generation models, followed by a presentation to HuggingFace Transformers, the Python library that we will use in the rest of the post. Then, we will focus on the GPT-2 model, and how to use the interface available in HuggingFace Transformers, both to generate text with the pre-trained models, as well as to re-train them with their own text. Finally, we will see the ethical risks associated with the use of these models without caution,since they have been trained with text from the internet and have learned the same biases present on the web.

## Text generation models

### 1.1 Introduction to text genration models

Text generation models began to be developed decades ago, long before the deep learning boom. The purpose of his type of models is to be able to predict a word or sequence of words given a text. The bottom diagram is a implified representation of what these models do, using a text as input, the model is capable of generating a robability distribution over the dictionary of words it knows, and choose based on it.

![[Pasted image 20231026110400.png]]

Early text generation models were trained using Markov chains, where each word was a state of the chain and the probability of the next word (based on the previous one) is calculated based on the number of occurrences of both words consecutively in the training text. Subsequently, recurrent neural networks (RNN) began to be used, which were capable of retaining a greater context of the text introduced, and Long Short-Term Memory (LSTM), which are a type of RNN that have a better long-term memory. Nevertheless, these type of networks are limited in what they can remember and they are also difficult to train, so they are not good for generating long texts.

In 2017 Google proposes a new architecture called the Transformer in its paper [“Attention Is All You Need”] ([https://arxiv.org/abs/](https://arxiv.org/abs/) 1706.03762), on which different text generation models are based today, such as GPT-2 and GPT-3, BERT or Transformer XL.

In this post we are going to focus on how to generate text with GPT-2, a text generation model created by OpenAI in February 2019 based on the architecture of the Transformer. It should be noted that GPT-2 is an autoregressive model, this means that it generates a word in each iteration. In addition, the model is available in different sizes depending on the embedding:

![[Pasted image 20231026110525.png]]

### 1.2 Setup

First, let’s import all the packages we are going to use. Specifically, the versions of these packages are:

- transformers==4.4.2
- datasets==1.5.0
- nlp
- colorama==0.4.4
- torch==1.9.1

## 3. Text generation with GPT-2

### 3.1 Model and tokenizer loading

The first step will be to load both the model and the tokenizer the model will use. We both do it through the interface of the GPT2 classes that exist in Huggingface Transformers `GPT2LMHeadModel` and`GPT2Tokenizer` respectively. In both cases, you must specify the version of the model you want to use, and the 4 dimensions of the model published by OpenAI are available:

- `'gpt2'`
- `'gpt2-medium'`
- `'gpt2-large'`
- `'gpt2-xl'`

**Model architecture of gpt-2** 

![[Pasted image 20231026113224.png]]

**Tokenizer architecture of GPT-2 model**

The tokenizer has three functions:

- It separates the input text into tokens, which do not necessarily have to coincide with words, and encodes and decodes those tokens to the input ids to the model and vice versa.
- It allows adding new tokens to the vocabulary
- It manages special tokens, such as masks, beginning of text, end of text, special separators, etc.

Through the tokenizer instance we can explore the vocabulary (`get_vocab`) and see its size, as well as explore and play tokenizing (`tokenize`) different texts to understand how it works.


### 3.2 Decoding methods and parameters

With all of the above we can already generate text. We have a tokenized text and a pre-trained model, we can call the `generate` function passing the tokenized text as input.


````python 
text = "I work as a data scientist"
text_ids = base_tokenizer.encode(text, return_tensors = 'pt')

generated_text_samples = base_model.generate(
    text_ids
)
generated_text_samples

````

As the output is again a tensor, we will have to decode the output using the tokenizer token by token:

```python
for i, beam in enumerate(generated_text_samples):
    print(f"{i}: {base_tokenizer.decode(beam, skip_special_tokens=True)}")
    print()

```

However, it is important to mention the relevance of decoding methods (ways of choosing the next word or words given a phrase), since the quality of the obtained text will vary significantly. They can be configured based on the parameters that are passed to the generation function.

## 4. Fine Tunning : How to generate fake news

GPT-2 has been trained with generic text downloaded from the internet (Wikipedia, Reddit, etc.), so if we want the text structure to be in a certain way or the content to focus on one theme, it is not enough to just use the pre-trained model available in Transformers. To do this, a fine-tuning of the model can be done, which consists of adding some layers to the architecture and retraining the model with a dataset containing the desired theme or text structure.

Fine-tuning allows you to control both the structure and the theme of the text that is generated, based on the input dataset. In addition, you do not need as large a data volume as used for the GPT-2 scratch training, making it more affordable.

To do the fine-tuning, you have to follow three steps:

1. Get the data
2. Process it to add the start and end tokens of the text (or those that are needed according to the type of text that you want to generate)
3. Train the base model with this new data

We are going to  generate text in news format (fake news): headline + article. For this, 2 models are necessary:

1. **Headline generation model**, by fine-tuning GPT-2 small with headlines from various newspapers
2. **Article generation model**, by fine-tuning GPT-2 small with headlines and articles so that, given a headline, it generates the first sentences of an article.

We will use the following dataset with headlines and news articles from 2016: [https://www.kaggle.com/snapcrack/all-the-news?select=articles1.csv](https://www.kaggle.com/snapcrack/all-the-news?select=articles1.csv). Most of the news are about Trump, Obama and Hillary Clinton.

### 4.1 Fine-tunning to generate headlines

**Loading the tokenizer and model with special tokens**

We define the start and end tokens of the headlines and add them:

- to the tokenizer as special tokens and
- to pre-trained model configuration when loaded

```python

# the eos and bos tokens are defined
bos = '<|endoftext|>'
eos = '<|EOS|>'
pad = '<|pad|>'

special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad}

# the new token is added to the tokenizer
num_added_toks = base_tokenizer.add_special_tokens(special_tokens_dict)

# the model config to which we add the special tokens
config = AutoConfig.from_pretrained('gpt2', 
                                    bos_token_id=base_tokenizer.bos_token_id,
                                    eos_token_id=base_tokenizer.eos_token_id,
                                    pad_token_id=base_tokenizer.pad_token_id,
                                    output_hidden_states=False)

# the pre-trained model is loaded with the custom configuration
base_model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)

# the model embedding is resized
base_model.resize_token_embeddings(len(base_tokenizer))

```

#### Data loading and processing

The data processing in this case consists of three steps:

1. Clean up the dataset
2. Add the start and end tokens to the headlines
3. Generate the tokenized datasets that we can pass to the model to train

It is vitally important to clean and process the texts before training the model, since the presence of noise will make the quality of the text generated by the re-trained model worse than that of the default model. Therefore we are going to filter the headlines:

- Empty or null
- We remove the name of the publication from those headlines in which it appears
- We discard headlines with less than 8 words
- We discard duplicate headlines
- We keep the first 100 words of the articles

#### Training

We train in the same way as we did with the articles. In this case, the training was stopped without having finished, and continued from the last saved checkpoint.

```python

model_articles_path = './news-articles_v4'

training_args = TrainingArguments(
    output_dir=model_articles_path,          # output directory
    num_train_epochs=2,              # total # of training epochs
    per_device_train_batch_size=5,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=200,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir=model_articles_path,            # directory for storing logs
    prediction_loss_only=True,
    save_steps=10000
)
```


Source:
Déploiement: https://www.youtube.com/watch?v=WM6_fGABaw8
https://spotintelligence.com/2023/04/21/fine-tuning-gpt-3/
https://botpress.com/blog/how-can-i-train-my-own-gpt-model
https://www.modeldifferently.com/en/2021/12/generaci%C3%B3n-de-fake-news-con-gpt-2/
https://blog.propelauth.com/rapid-prototype-gpt-app/


https://dlmade.medium.com/trip-planner-end-to-end-gpt-project-with-flask-and-lambda-6ff756053e3f