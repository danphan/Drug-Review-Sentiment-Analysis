# drug-review-project

In this repo, we perform exploratory data analysis and sentiment analysis of drug reviews, taken from [UCI ML Drug Review Dataset](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29). To model the topics in the drug reviews, we apply term-frequency-inverse-document-frequency (TF-IDF), Latent Semantic Analysis (LSA) and k-means clustering. We also perform Latent Dirichlet Allocation (LDA) to model the topics in the different reviews.

For sentiment analysis, we apply the following models:
1. Logistic Regression
2. Support Vector Machines (SVMs)
3. Naive Bayes (Multinomial)
4. Random Forests
5. Fine-tuned transformers (BERT)
