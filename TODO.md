### Tasks
* Next steps

* **Main streams**
- [x] Finetuning on all three datasets - resources?
    * So I have it for all combinations of datasets for both ordinal and multiclass settings except for multiclass pairs (fb+csfd, fb+mall, csfd+mall)
- [ ] Preprocessing of the emails
    * Send preprocessed emails back to Stepan - he might arrange a labelling experiment
- [ ] Data
    * Get train, val, test splits from mac
- [ ] Adaptation
- [ ] Inference (interface, interaction, output structure)

* **Other**
- [ ] Gradient clipping https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961
    * Norm clipping https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/tricks/gradclipping_mlp.ipynb
- [ ] Put all params to config
    * If arguments not passed from CLI, take the defaults from config
- [ ] Try stronger model - Robeczech
    * Just for one dataset to see the possible improvement
- [ ] Train the models
    * Use balanced datasets (in terms of source and labels) for the multitask
      learning?
    * Rather use as many samples as possible
- [ ] Think about main.py
    * What will be the main functionalities of the whole code?
    * Inference, bulk inference, train,...
    * Based on that define what and how will be exposed to the user - what will
      the user eventually call

##### Open questions
- [ ] Separator between conversations in mails.txt?
- [ ] Encoding of the mails.txt? Replace special chars for czech letters? Or just fix encoding somehow?
- [ ] Distribution of the results - are these emails closed set, so should the results cover the whole range between 0 and 1? Or do we want "absolute" sentiment (most of the values would probably be somewhere around 0.5)? If the results are input for next mode, it doesn't really matter.
- [ ] What is the desired output? Answer with score? Any ID?
- [ ] Inference time/requirements - how heavy can the final model be?
- [ ] Interaction of the user with the tool - API? Bulk inference? Adaptation on target data?

#### On hold
- [ ] API with inference for one email
    * Agree first on how the output should look like
- [ ] Preprocess the emails
    * Wait for reply with additional info about the super messy file
- [ ] Preprocess data
	* How? Inspiration here https://is.muni.cz/th/n0lnb/Sentiment_Analysis_cz.pdf + other Czech papers dealing with sentiment analysis
	* Maybe I can have preprocessing pipeline, start with nothing and just add elements step by step
	* Do I really need big preprocessing if I am using BERT-like tokenizers? Well not really, I just need to preprocess the emails, but if I get cleaned email body, I am good to go with the tokenizers

#### Done
- [x] Change splitting finetuning data to train:val:test 7:1:2
- [x] Implement CORN ordinal classification
	- [x] Encoding labels
    	- [x] Change last layer of the ClassificationHead
    	- [x] Change loss
    	- [x] Revisit step 2b of adaptation - distillation with the new labels and loss
		* https://openreview.net/forum?id=3jBXX9Xb1iz
- [x] If adaptation is performed on dataset with labels, I want to save the
  train and val spilts
    * There are no splits, everything can be train and validation at the same time - I can use the whole dataset for adaptation
- [x] To be able to skip the validation, I need to save the validation datasets first to evaluate the models on them later
- [x] Inference
    * Predict method of the AdaptiveSentimentClassifier
- [x] Look for a lighter model
    * https://huggingface.co/Seznam/small-e-czech
- [x] Adaptation method
    * Finish the method and follow the same structure as the finetuning - same
      input and output, saving the models
- [x] How to connect the finetuning with adaptation? Mainly how to ensure that
  the datasets are passed correctly, that the source_train and source_val for
  adaptation are the same as they were for finetuning of the selected
  classification head
- [x] Model to CUDA - do I have to put there also every single input tensor?
- [x] Create loading of the Czech sentiment classification datasets
	* What datasets? CSFD, FB, MALL
	* Loading from url as well as loading local files from raw
- [x] Get pretrained models
	* What models? Robeczech, Czert, XLM-RoBERTa, XLM-RoBERTa-large, FERNETs
    1. Robeczech should perform better then Czert + much more downloaded from huggingface
    2. XLM-RoBERTa-large is just too big
    3. FERNETs are pretrained on less formal data, but that is probably something I don't really want
    * Overall Robeczech seems like the most reasonable option
- [x] Finetune selected models on sentiment datasets
    * No further pretraining - I don't want the encoder to match the domain perfectly
    * Decide on the parameters and finetune - doesn't have to be serious but to have the "finetuned" model ready
    * Decide on what parameters might be changing during the training so that I can write the finetune method
    * Parameters as per finetuning BERT paper
    	* Layerwise learning rate decay = 0.95
	* Dropout = 0.1
	* Adam optimizer (b1 = 0.9, b2 = 0.999)
	* linear schedule with warmup (warmup proportion = 0.1)
	* base learning rate = 2e-5
	* n_epochs = 4
    * Create a nice training script - modularize the current spaghetti
    	- Probably it would be good to have one class for my final "classifier" with methods like:
		- "finetune on source"
		- "finetune on target"
		- appropriate init
		- inference
	- First, go through the current code and try to split it to reasonable parts
    * On all at once? One by one?
    * First, try to replicate the results from the Robeczech paper, if I can get there, then I might try improving it
    * Output of the final model should be on scale 0-1 (~negative-positive) so I can either drop the neutral class and make it binary classification problem or I could use "distilation loss function" and give the neutral class value 0.5
- [x] Learn how to run the code on Metacentrum
- [x] Create the ClassificationDataset class
    * It will be able to create and store the torch dataset instance
    * It will be able to create and store the torch dataloader instance
    * It will be able to preprocess and tokenize its inputs
    * It will be able to transform labels/take only subset of the data based on labels
    * Splitting to train, val, test will not be its method, will be a separate function
- [x] Model classes
    * Classifiers
    * Encoders - just to load correct models by default and simplify the forward method to output only the embeddings
    * Tokenizer - just to load correct models by default


### Continuous
- [ ] Orchestrate the pipeline with makefile
