# Flirt Detection Model


Tested out 15 different hyperparameter configurations for DistilBERT model. Most of them showed very bad generalization on the test set. 

The model i trained firstly turned out to be the best performing with 78% accuracy on test dataset. It is stored in 01_detection_training.ipynb.

TRIED NEW CONFIGURATIONS:

ALL MODELS (sorted by test accuracy):
    model_type                                            model  accuracy  f1_weighted
   Transformer cardiffnlp/twitter-roberta-base-sentiment-latest  0.856631     0.856623
   Transformer                        microsoft/deberta-v3-base  0.842294     0.841215
   Transformer                                     roberta-base  0.838710     0.838701
Traditional ML                               LogisticRegression  0.772401     0.772079
Traditional ML                                        LinearSVC  0.770609     0.770465
Traditional ML                                 GradientBoosting  0.761649     0.759946


======================================================================
🏆 BEST OVERALL MODEL
======================================================================
Type: Transformer
Model: cardiffnlp/twitter-roberta-base-sentiment-latest
Test Accuracy: 0.8566
Test F1: 0.8566
Improvement over DistilBERT (78.67%): +6.99%
