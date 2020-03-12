# HLA Allele Imputation with Deep Convolutional Neural Network (CNN)

Scripts for running HLA allele imputation with deep convolutional neural network. Assume the directory structure is as follows

```
current
  - scripts
  - results
  - models
  - data
```

where python scripts are under the `scripts/` folder.

## Hyperparameter tuning
	
To perform random search hyperparameter tuning (search space specified in the script)

```
python tune.py
```

results of hyperparameter tuning saved in `results/` folder.

## Train CNN

To train CNN, (1) save model and (2) save HLA imputation accuracy results

```
python ConvNet.py -t [training_data] -v [test_data] -m [model_directory] -r [result_directory]
```

Where the all paths (i.e. `training_data`) are specified in the same level as `scripts/` directory. For example, `-t data/train/T1DGC_REF_Train.txt`. 

## Generate error bars 

To generate error bars for test accuracy by HLA allele via bootstrapping the test dataset. 

```
python bootstrap.py -t [test_data] -m [model_directory] -r [result_directory]
```

