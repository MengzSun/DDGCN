# DDGCN
Code for the AAAI 2022 paper "[DDGCN: Dual Dynamic Graph Convolutional Networks for Rumor Detection on Social Media](https://www.aaai.org/AAAI22Papers/AAAI-6370.SunM.pdf)"



Make sure the following files are present as per the directory structure before running the code,
```
DDGCN
├── README.md
├── preprocess
|   ├──  *.py
|   └──  *.txt
├── models
|   ├── *.py 
|   └── model_saved
|        └── ckpt_nn.model
├── data
|   └── pheme
|        ├── all-rnr-annotated-threads
|        │   ├── ebola-essien-all-rnr-threads
|        │   ├── charliehebdo-all-rnr-threads
|            ├── ......
|        │   └── sydneysiege-all-rnr-threads
|        ├── pheme_clean
|        ├── pheme_concept_yago
|        ├── pheme_entity
|        ├── pheme_temporal_data
|        ├── mid2text.txt
|        ├── mid2token.csv
|        ├── node2idx.txt
|        ├── node2idx_mid.txt
|        ├── node2idx_test.txt
|        └── pheme_id_label.txt
└── requirement.txt

```
pheme_clean, pheme_concept_yago, pheme_entity and pheme_temporal_data these four folders are packed into a zip file, and can be obtained from https://www.dropbox.com/s/xwn5dvqgx2n2vsd/pheme_peocessed_data.zip?dl=0.
The Raw Pheme dataset can be obtained from https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078 (or https://www.dropbox.com/s/j8x105s60ow997f/all-rnr-annotated-threads.zip?dl=0)


# Dependencies

Our code runs with the following packages installed:
```
python 3.6
torch 1.4.0+cu100
torch-cluster 1.5.2
torch-geometric 1.7.2
torch-scatter 2.0.3
torch-sparse 0.5.1
tqdm
numpy
pandas
matplotlib
scikit_learn
```

install the virtual environment with 
```
pip install -r requirement.txt
```



# Run

Train and test,
```
python train_dynamic.py --dataset pheme --model completed --cuda 1 --batch 32 --epoch 5 --lr 0.001
```


