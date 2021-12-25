# Getting Started with ExcelAnnotator

This document provides brief steps for the usage of ExcelAnnotator.
It has been tested on Windows 10 machines, especially this works for Windows 10 Virtual machines.
These steps have been tested with Windows 10 build 19042.572 (64-bit), Excel version with Microsoft 365 (3.12.2020).

---
## 1. Document domain
ExcelAnnotator works on Excel source files (xls or xlsx). It requires the ability to manipulate the source files, thus e.g. password protected files are not possible to process.

ExcelAnnotator generates from the Excel source file `example.xlsx` the following directory structure as output:

`example.xlsx` is replaced by a index, say 0. Each sheet of `example.xlsx` also gets numbered.

Assuming `example.xlsx` has one sheet, following output is generated:

```bash
output_directory
|
+--0-0
|    |
|    +-- 0-0.json
|    +-- 0-0.pdf
|    +-- 0-0-0.png
|    +-- 0-0-automated.json
|
+--1-0
|    |
|    ...
...
```

- **0-0.json** contains document related information.
- **0-0.pdf** is the generated PDF using Excel export of the worksheet.
- **0-0-0.png** is the image file of page 0, if the table is longer than one page, files 0-0-i.png exists for all pages i.
- **0-0-automated.json** contains the generated annotations.

---
## 2. Prerequisites

### **2.1 Clone Repository**
Download and install a git client and clone this repository:
```batch
git clone git@github.com:DS3Lab/docparser-raoschi.git
```
into `<git-home>` directory. (home directory of docparser-raoschi is denoted as git-home furtheron).

### **2.2 Excel**
Download and install Office (We used Microsoft 365 on 3.12.2020).

### **2.3 Python**
Download and install python for Windows. We installed and utilized Python 3.9.0.

#### **2.3.1 Python requirements**
Load provided venv and install required tools:

Activate virtual environment:
```batch
cd <venv directory>
./Scripts/activate.bat
```

Set PythonPath:
```batch
cd <git-home>/doc_parser
set PYTHONPATH=%CD%
```

Make sure pip is installed, if not install pip:
```batch
py -m pip install --upgrade pip
```
*(Hint: Make sure pip is on path.)*

Install required tools, python plugins:
```batch
pip install pdf2image pywin32
```

### **2.4 DeExcelerator**
Setup and run DeExcelerator on all documents of the document domain. **ExcelAnnotator requires DeExcelerator being run beforehand**.

---
## 3. Set up environment dependent variables
It is recommended to use **IntelliJ** (IDEA Community Edition) on Windows to setup the running environment. For missing jar packages, check out the directory `./jar`. 

The environment dependent variables can be found in the main python script `<git-home>\doc_parser\__main__.py`

- **poppler_path**: Path pointing to exe of poppler installation. If default poppler used no changes are needed ( `<git-home>\doc_parser\Util\poppler\poppler-0.68.0\bin`).
- **metadata_directory**: Path pointing to previously genereated DeExcelerator output
- **intermediate_directory**: Path pointing to directory where intermediate data is stored
- **source_directory**: Path pointing to directory where source files are stored
- **shuffled_files_list**: Path pointing to json file which stores the shuffled files list. Either a provided list or if `load_shuffled_list = False` the shuffled list is stored into this file. Overwrites previous document.
- **output_directory**: Path pointing to directory where output data will be stored


---
## 4. Run ExcelAnnotator

```batch
cd <git-home>\doc_parser
python doc_parser\__main__.py
```

ExcelAnnotator runs single-threaded due to the limitation of win32com. On our test setup, the fastest machine needed about 15 seconds per file for porcessing. A possibility to speed up is to use several virtual machines in parallel.
ExcelAnnotator has been prepared for this:
`num_file_chunks` and `machine_index` can be used to process only a partition of all files once and distribute them among several machines.

---
## 5. Extract successfully annotated files

Using bash integration for Windows 10, with following command one can extract all the files of fully completed annotations into a new folder `<DESTINATION>`. (This is helpful for usage with the Annotator GUI, as empty folders throw exceptions there).

```bash
for dir in find . -type d -exec test -e {}/{}-automated.json -a -e {}/{}.json \; -print`; do cp -r $dir <DESTINATION> ; done
```

According batch commands are defenitely also available.

---

The generated annotations can then be used for training a Neural network or can directly be viewed in the Annotator GUI of [Doc_annotation](https://github.com/j-rausch/doc_annotation/tree/master/doc-anno-client).
