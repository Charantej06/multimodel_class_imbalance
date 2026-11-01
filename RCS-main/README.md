# Rebalancing with Calibrated Sub-classes (RCS): An enhanced approach for Robust Imbalanced Classification

This methods generates synthetic samples using the statistics of the majority and the intermediate class samples.

## Usage
```
Python 3.10.12
Pytorch 2.0.1
```

## Run the following scripts

- Extract training data and store in the data folder by running the script img_extract.py in data/dataset folder

- Run the scripts img_extract_val.py and sample_extract.py in data/dataset folder for the validation data

- run model_training_dataset.py (mention the dataset) 

- run sample_generation_dataset.py (mention the dataset)

- run classification.py (write the name of the dataset and the path for the oversampled and validation data)

- The t-SNE plot for generated samples are included in sample_generation_dataset.py script
