# Integrating higher-level semantics into robust biomedical name representations

This directory contains source code for the following paper:

`Scalable Few-Shot Learning of Robust Biomedical Name Representations.` \
Pieter Fivez, Simon Šuster and Walter Daelemans. *BioNLP (NAACL)*, 2021.


## License

GPL-3.0

## Requirements

All requirements are listed in **requirements.txt**. 

You can run `pip install -r requirements.txt`, preferably in a virtual environment.

The fastText model used in the paper can be downloaded from the following link: 
https://drive.google.com/file/d/1B07lc3eeW_zughHguugLBR4iJYQj_Wxz/view?usp=sharing \
Our example script requires a path to this downloaded model.

## Data

We demonstrate our code using the ICD-10 data. We have extracted this data using source files which can be found at
 https://github.com/kamillamagna/ICD-10-CSV.

The script **data/icd_chapters.py** has used these source files to create **data/icd10.json**.

## Code

We provide a script to run our training objectives from the paper using fastText embeddings as input.

**main_dan.py** trains our proposed encoder on **data/icd10.json** and reports results on the benchmarks in **data/benchmarks.json** \
Please run `python main.py --help` to see the options, or check the script. \
The default parameters are the best parameters reported in our paper.
