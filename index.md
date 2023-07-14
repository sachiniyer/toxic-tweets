# Toxic Tweet Project

## What is this project?

This project uses Bert to and the toxic tweet dataset to figure the
classes of toxicity that a tweet falls under

The classes of toxicity are as follows:

-   Toxic
-   Severe Toxic
-   obscene
-   threat
-   insult
-   identity hate

## Where is this project?

### Github

The code is hosted on
[github](https://github.com/sachiniyer/toxic-tweets)

#### Huggingface

The model is deployed on
[huggingface](https://huggingface.co/spaces/sachiniyer/toxic-tweets)

## What is the accuracy?

About 90% according to [this
notebook](https://github.com/sachiniyer/toxic-tweets/blob/main/accuracy.ipynb).
I don\'t have access to a dataset that has labels for all types of
toxicity, but my validation dataset also gave scores of around 90%.

## How does it work?

### A high level video explanation

### A technical textual explanation

#### I will go through a more in depth explanation of the code (however I will be skipping stuff like imports and sanity checks)

1.  First you must create the tokenizer. This is what is going to take
    the text that we have and actually embed it into tokens for us to
    train on.

                        
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', \
                                            truncation=True, do_lower_case=True)
                        
                    

2.  Then we do some data wrangling in order to make our dataframe easier
    to work with. This mostly means getting rid of columns we don\'t
    want as well as one-shot encoding the labels

                        
        classes = list(df.columns)[2:]
        df['labels'] = df.iloc[:, 2:].values.tolist()
        df['text'] = df['comment_text']
        df.drop(['id', 'comment_text'], inplace=True, axis=1)
        df.drop(classes, inplace=True, axis=1)
        df.head()
                        
                    

3.  We then create a class which is used for encoding. We will break
    this up into multiple parts
    1.  First we have our initialization function which includes the
        tokenizer that we created before, as well as the dataframe we
        are operating on, and the max-length we would like our input_id
        vector to be.

             
            def __init__(self, df, tokenizer, max_len):
                self.tokenizer = tokenizer
                self.df = df
                self.texts = df.text
                self.targets = self.df.labels
                self.max_len = max_len
                                 

    2.  Then we do three things. First we get rid of bad whitespace
        (newlines, tabs, etc.). Then we create a tokenizer that will
        parse the inputs and return all of the values that we want back.
        Then we convert those values into tensors, and finally, we
        return them from the function We will create both validation and
        training datasets from this class.

                                    
            def __getitem__(self, index):
                text = str(self.texts[index])
                text = " ".join(text.split())

                inputs = self.tokenizer.encode_plus(
                    text,
                    None,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    pad_to_max_length=True,
                    return_token_type_ids=True,
                    truncation=True,
                    return_attention_mask=True,
                )
                ids = torch.LongTensor(inputs['input_ids'])
                mask = torch.LongTensor(inputs['attention_mask'])
                token_type_ids = torch.LongTensor(inputs['token_type_ids'])
                targets = torch.FloatTensor(self.targets[index])

                return {
                    'ids': ids,
                    'mask': mask,
                    'token_type_ids': token_type_ids,
                    'targets': targets,
                    'text': text
                }

                                    
                                

4.  Then we actually create the model. I decided to use a sequence
    classification version of bert for my model so that it would more
    accurately represent the problem. I also defined the number of
    labels, and possibly most importantly, I defined the problem_type.
    This is essential to working with the transformers library, because
    otherwise you will have all of your values softmaxed when you don\'t
    want them to be.

                        
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6, problem_type="multi_label_classification")
        model.to(device)
                        
                    

5.  Now, we actually create the optimizer and loss functions to use
    during our training. The optimizer is used to update the model
    parameters in accordance with the loss function that we have. The
    specific optimizer choosen - AdamW also takes into account weight
    decay which is useful to make sure that we don\'t overfit on our
    data. The loss function is also for binary cross entropy instead of
    regular cross entropy so that means it can handle the binary data
    better.

                        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        loss = torch.nn.BCEWithLogitsLoss()
                        
                    

6.  Then we actually train our model. A few things happen here. First we
    get the data from the dataloader we created earlier (this is
    actually a batch of data and contains multiple samples). Then we
    extract all the information that we will use to train our model -
    input_ids, attention_masks, and the optimal labels. We can then use
    this to train our model, and update our optimizer in accordance with
    the loss function.

                        
        def train(epoch):
            model.train()
            for i,data in tqdm(enumerate(df_train, 0)):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)
                optimizer.zero_grad()
                out = model(input_ids=ids, attention_mask=mask, labels=targets)

                if i%100==0: print(f'epoch: {epoch} | loss: {out[0].item()}')

                out[0].backward()
                optimizer.step()
                        
                    

7.  Lastly, we can do some validation and exporting of the model, but
    that is somewhat verbose code, and does not seem like it needs
    explation. After this previous step you can save the model and get a
    very similar result as I did in my project.

## Can I try it?

::: iframe
You need to enable JavaScript to run this app.

::: {#root}
:::
:::
