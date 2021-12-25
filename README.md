# TableParser
Repo for "TableParser: Automatic Table Parsing with Weak Supervision from Spreadsheets" at SDU@AAAI-22 

- [TableParser](#tableparser)
    - [1. Clone repositories](#1-clone-repositories)
    - [2. System components of TableParser](#2-system-components-of-tableparser)
    - [3. References](#3-references)

### 1. Clone repositories
Download and install a git client and clone this repository:
```batch
git clone git@github.com:DS3Lab/TableParser.git
```
into `<git-home>` directory. (home directory is denoted as git-home furtheron).

### 2. System components of TableParser
- System overview of the TableParser pipeline 
    <object data="https://github.com/DS3Lab/TableParser/blob/main/figures/TableParser.drawio.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/DS3Lab/TableParser/blob/main/figures/TableParser.drawio.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/DS3Lab/TableParser/blob/main/figures/TableParser.drawio.pdf">Mask-RCNN</a>.</p>
    </embed>

- Model overview of Mask RCNN in DocParser

    <object data="https://github.com/DS3Lab/TableParser/blob/main/figures/mask-rcnn.drawio.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/DS3Lab/TableParser/blob/main/figures/mask-rcnn.drawio.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/DS3Lab/TableParser/blob/main/figures/mask-rcnn.drawio.pdf">Mask-RCNN</a>.</p>
    </embed>
</object>

- TableAnnotator: refer to [this repo.](https://anonymous.4open.science/r/doc_annotation-AAAI22-SDU/README.md) 
  - [Demo](https://github.com/DS3Lab/TableParser/blob/main/demo/2021-06-15%2002-05-58.gif) of annotating a table using TableAnnotator
- ExcelAnnotator: `./ExcelAnnotator`.
- TableParser pipelines: `./TableParser`.
- Data: Download from [this Google Drive link.](https://drive.google.com/file/d/1gaaHMG6f7sIH1DK4Ybg13_lBHNS2wbbn/view?usp=sharing)
- TableParser M1 (ModernTableParser) and M2 (HistoricalTableParser) can be downloaded from [this Google Drive link,](https://drive.google.com/file/d/1HxILaFrymyjuUtqyqcz3fyS5TrLhf-05/view?usp=sharing) and put under `.TableParser/TableParser/detectron2/tools/docparser_outputs`. 

### 3. References
To cite TableParser, refer to these items:
```bibtex
@inproceedings{rausch2021docparser,
  title={DocParser: Hierarchical Document Structure Parsing from Renderings},
  author={Rausch, Johannes and Martinez, Octavio and Bissig, Fabian and Zhang, Ce and Feuerriegel, Stefan},
  booktitle={35th AAAI Conference on Artificial Intelligence (AAAI-21)(virtual)},
  year={2021}
}
```
```bibtex
@article{rao2022tableparser,
  title={TableParser: Automatic Table Parsing with Weak Supervision from Spreadsheets},
  author={Rao, Susie Xi and Rausch, Johannes and Egger, Peter and Zhang, Ce},
  booktitle={Scientific Document Understanding Workshop (\tt SDU{@}AAAI-22)(virtual)},
  year={2022}
}
```