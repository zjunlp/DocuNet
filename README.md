
# Document-level Relation Extraction as Semantic Segmentation

This repository is the official implementation of [Document-level Relation Extraction as Semantic Segmentation](). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Datasets

The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data). The CDR and GDA datasets can be obtained following the intructions in [edge-oriented graph](https://github.com/fenchri/edge-oriented-graph). The expected structure of files is:
```
ATLOP
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json        
 |    |    |-- train_distant.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |-- cdr
 |    |    |-- train_filter.data
 |    |    |-- dev_filter.data
 |    |    |-- test_filter.data
 |    |-- gda
 |    |    |-- train.data
 |    |    |-- dev.data
 |    |    |-- test.data
 |-- meta
 |    |-- rel2id.json
```

## Training
### DocRED
Train the DocuNet model on DocRED with the following command:

```bash
>> bash scripts/run_docred_bert.sh  # for BERT
>> bash scripts/run_docred_roberta.sh  # for RoBERTa
```

### CDR and GDA
Train DocuNet model on CDA and GDA with the following command:
```bash
>> bash scripts/run_cdr.sh  # for CDR
>> bash scripts/run_gda.sh  # for GDA
```
 

## Evaluating

To evaluate the trained model in the paper, you setting the `--load_path` argument in training scripts. The program will log the result of evaluation automatically. And for DocRED  it will generate a test file `result.json` in the official evaluation format. You can compress and submit it to Colab for the official test score.

