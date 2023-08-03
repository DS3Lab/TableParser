# TableParser

Repo for TableParser based on DocParser

- [TableParser](#tableparser)
  - [1. Project goal](#1-project-goal)
  - [2. env](#2-env)
  - [3. Testing (eval and demo) and finetuning](#3-testing-eval-and-demo-and-finetuning)
    - [3.1 Preparation: Conversion from DocParser to MScoco](#31-preparation-conversion-from-docparser-to-mscoco)
    - [3.2 Finetuning for ModernTableParser](#32-finetuning-for-moderntableparser)
    - [3.2 Run inference demo on the dataset `ZHYearbook-Excel-Test`](#32-run-inference-demo-on-the-dataset-zhyearbook-excel-test)
    - [3.3 Get evaluation](#33-get-evaluation)
    - [3.4 Pre-trained model with `ZHYearbook-Excel-WS`](#34-pre-trained-model-with-zhyearbook-excel-ws)
    - [3.5 ModernTableParser](#35-moderntableparser)
    - [3.6 HistoricalTableParser](#36-historicaltableparser)
  - [4. Google Vision OCR => TableParser](#4-google-vision-ocr--tableparser)

## 1. Project goal
To design a TableParser based on the DocParser, which deals with all sorts of tables (scanned, automatically generated) in a robust way. 
1. We also make the [TableAnnotator GUI](https://github.com/susierao/doc_annotation) (write to srao@ethz.ch to ask for access) openly accessible where users can visualize their outputs easily, make adjustments, and export in a json/csv file. For the open-source dataset, see [here](https://drive.google.com/file/d/1gaaHMG6f7sIH1DK4Ybg13_lBHNS2wbbn/view?usp=sharing) for download. For our ExcelAnnotator, see [here](https://anonymous.4open.science/r/ExcelAnnotator-D8E5/Readme.md). 
2. We also provide our pretrained model for new domain adaptation, so that users can fine tune the model with their in-domain tables (under `pretrained_models').


## 2. env 
Make sure to set up an env with `conda` using the configuration in `environment.yaml` and install `detectron2`, `opencv` and [`docparser`](https://github.com/DS3Lab/DocParser).


## 3. Testing (eval and demo) and finetuning

### 3.1 Preparation: Conversion from DocParser to MScoco
  
Run `python docparser/utils/create_dataset_groundtruths_exact.py`. 

### 3.2 Finetuning for ModernTableParser 
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url 'tcp://127.0.0.1:5362' --resume --config-file ./<git-home>/TableParser/detectron2/configs/arxivdocs-Detection/docparser_yearbooks_tables_4gpu_thesis_v1_swisstext_finetune.yaml MODEL.WEIGHTS pretrained_models/docparser_tables_4gpu_ws/model_final.pth OUTPUT_DIR tools/docparser_outputs/docparser_yearbooks_ws_tables_4gpu_thesis_v1_swisstext_finetune_v2_from_arxivdocs
```

### 3.2 Run inference demo on the dataset `ZHYearbook-Excel-Test`
```sh
CUDA_VISIBLE_DEVICES=0 python demo/demo.py --config-file ./<git-home>/TableParser/detectron2/configs/arxivdocs-Detection/docparser_yearbooks_tables_4gpu_thesis_v1_swisstext_finetune.yaml --input ./<input-image-dir>/* --output "demo/yearbooks_swisstext/outputs_automatictrain_yearbooks_ws/" --opts MODEL.WEIGHTS pretrained_models/docparser_yearbooks_ws_tables_4gpu_thesis_v1_swisstext_finetune_v2_from_arxivdocs/model_final.pth
```
### 3.3 Get evaluation
```sh
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url 'tcp://127.0.0.1:5362' --resume --config-file ./<git-home>/TableParser/detectron2/configs/arxivdocs-Detection/docparser_yearbooks_tables_4gpu_thesis_v1_swisstext_finetune_v2.yaml MODEL.WEIGHTS pretrained_models/docparser_tables_4gpu_ws/model_final.pth OUTPUT_DIR tools/docparser_outputs/docparser_yearbooks_ws_tables_4gpu_thesis_v1_swisstext_finetune_v2_from_arxivdocs
```
**Note**: TableParser M1 (ModernTableParser) and M2 (HistoricalTableParser) can be downloaded from [this Google Drive link,](https://drive.google.com/file/d/1HxILaFrymyjuUtqyqcz3fyS5TrLhf-05/view?usp=sharing) and put under `.TableParser/TableParser/detectron2/tools/docparser_outputs`. 

### 3.4 Pre-trained model with `ZHYearbook-Excel-WS` 
Model under `./TableParser/detectron2/tools/docparser_outputs/docparser_tables_4gpu_ws` 

### 3.5 ModernTableParser 
Model under `./TableParser/detectron2/tools/docparser_outputs/docparser_yearbooks_ws_tables_4gpu_thesis_v1_swisstext_finetune_v2_from_arxivdocs`

### 3.6 HistoricalTableParser 
Model under `./TableParser/detectron2/tools/docparser_outputs/manual_docparser_Au_tables_4gpu_finetune_v2`

## 4. Google Vision OCR => TableParser
- Getting OCR bounding boxes, see`./preprocessing/google-vision-bbox.ipynb` 
- Adding row/column: We also used the postprocessing script to add row/columns and then manually curated the files. see `./preprocessing/adding_col_row_ranges_after_manual.ipynb` for details.
