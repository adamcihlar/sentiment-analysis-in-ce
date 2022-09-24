### Tasks
- [x] Create loading of the Czech sentiment classification datasets
	* What datasets? CSFD, FB, MALL
	* Loading from url as well as loading local files from raw
- [ ] Get pretrained models
	* What models? Robeczech, Czert, XLM-RoBERTa, XLM-RoBERTa-large, FERNETs
- [ ] Finetune selected models on sentiment datasets
    * On all at once? One by one?
- [ ] Preprocess data
	* How? Inspiration here https://is.muni.cz/th/n0lnb/Sentiment_Analysis_cz.pdf + other Czech papers dealing with sentiment analysis
	* Do I really need big preprocessing if I am using BERT-like tokenizers?
	* Maybe I can have preprocessing pipeline, start with nothing and just add elements step by step

### Continuous
- [ ] Orchestrate the pipeline with makefile
