# DDGCN

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
└───data
    └── pheme
         ├── all-rnr-annotated-threads
         │   ├── ebola-essien-all-rnr-threads
         │   ├── charliehebdo-all-rnr-threads
             ├── ......
         │   └── sydneysiege-all-rnr-threads
         ├── pheme_clean
         ├── pheme_concept_yago
         ├── pheme_entity
         ├── pheme_temporal_data
         ├── mid2text.txt
         ├── mid2token.csv
         ├── node2idx.txt
         ├── node2idx_mid.txt
         ├── node2idx_test.txt
         └── pheme_id_label.txt


```
# Dependencies

Our code runs with the following packages installed:
```
python 3.6
torch 1.3.1
nltk 3.2.5
tqdm
numpy
pandas
matplotlib
scikit_learn
xlrd (pip install xlrd)
```



# Run

Train and test,
```
python train_dynamic.py --dataset pheme --model completed --cuda 1 --batch 32 --epoch 5 --lr 0.001
```



# Citation
If you make advantage of our model in your research, please cite the following in your manuscript:
```
```
