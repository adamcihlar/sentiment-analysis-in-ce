### Tasks
- [x] Create loading of the Czech sentiment classification datasets
	* What datasets? CSFD, FB, MALL
	* Loading from url as well as loading local files from raw
- [x] Get pretrained models
	* What models? Robeczech, Czert, XLM-RoBERTa, XLM-RoBERTa-large, FERNETs
    1. Robeczech should perform better then Czert + much more downloaded from huggingface
    2. XLM-RoBERTa-large is just too big
    3. FERNETs are pretrained on less formal data, but that is probably something I don't really want
    * Overall Robeczech seems like the most reasonable option
- [ ] Finetune selected models on sentiment datasets
    * Create a nice training script - modularize the current spaghetti
    	- Probably it would be good to have one class for my final "classifier" with methods like:
		- "finetune on source"
		- "finetune on target"
		- appropriate init
		- inference
	- First, go through the current code and try to split it to reasonable parts
    * On all at once? One by one?
    * First, try to replicate the results from the Robeczech paper, if I can get there, then I might try improving it
- [x] Learn how to run the code on Metacentrum
- [ ] Preprocess data
	* How? Inspiration here https://is.muni.cz/th/n0lnb/Sentiment_Analysis_cz.pdf + other Czech papers dealing with sentiment analysis
	* Do I really need big preprocessing if I am using BERT-like tokenizers?
	* Maybe I can have preprocessing pipeline, start with nothing and just add elements step by step
- [ ] Create the ClassificationDataset class
    * It will be able to create and store the torch dataset instance
    * It will be able to create and store the torch dataloader instance
    * It will be able to preprocess and tokenize its inputs
- [ ] Model classes
    * Classifiers
    * Encoder


### Continuous
- [ ] Orchestrate the pipeline with makefile
