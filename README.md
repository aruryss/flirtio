# flirtio

### preprocessing done
i gathered everything in process_all.py file and as result got:  
=== Processing Detection Data ===  
Removed 4669 duplicates  
Merged dataset size: 4368  
Class distribution:  
label  
0.0    2397  
1.0    1971  
Name: count, dtype: int64  
Balanced dataset size: 3942  
Train size: 2758  
Val size: 592  
Test size: 592  

all processed data is in: data\\processed


### flirt detection model
This project implements a flirt detection model using DistilBERT. The goal is to accurately detect flirtatious behavior in text data.


Tested out 15 different hyperparameter configurations for DistilBERT model. Most of them showed very bad generalization on the test set. 

The model i trained firstly turned out to be the best performing with 78% accuracy on test dataset. It is stored in 01_detection_training.ipynb.

Maybe I will try different configurations later. 

To install the required dependencies, run:

```
pip install -r requirements.txt
```
