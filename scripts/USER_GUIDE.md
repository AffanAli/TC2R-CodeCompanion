# User Guide: Run the Document Pipeline

## Before you start
You need:
- Python 3.11
- the files in the `scripts` folder
- a folder of documents to process
- a trained model file named `ml_classifier_gbc.pkl`

## Folder setup
Put your files in this structure:

```text
repo/
└── scripts/
    ├── pipeline.py
    ├── helper_functions.py
    ├── unit_tests.py
    ├── requirements.txt
    ├── Dockerfile
    ├── tests/
    │   ├── docs/
    │   ├── output/
    │   └── model/
```

Put your new documents in `scripts/tests/docs/`.
Put your trained model file in `scripts/tests/model/`.

## Step 1: Install dependencies
Open Terminal in the `scripts` folder and run:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Also install Tesseract OCR on your machine.

## Step 2: Run the pipeline
Excel output:

```bash
python pipeline.py --input_folder ./tests/docs --output_folder ./tests/output --model_folder ./tests/model --output_format excel
```

Pickle output:

```bash
python pipeline.py --input_folder ./tests/docs --output_folder ./tests/output --model_folder ./tests/model --output_format pickle --output_name model_results.pkl
```

SQLite output:

```bash
python pipeline.py --input_folder ./tests/docs --output_folder ./tests/output --model_folder ./tests/model --output_format sqlite --output_name model_results.db
```

## Step 3: Check the results
Look in `scripts/tests/output/` for:
- `model_results.xlsx` or chosen output file
- `data_df.pkl`
- `OCR_quality_distribution.png`
- `text_files/`

The Excel output includes a `parsed_days` column for non-technical users.

## Troubleshooting
### Error: No module named helper_functions
Run the command from inside the `scripts` folder, or use package-safe imports already included in the updated files.

### Error: Tesseract not found
Install Tesseract OCR and confirm it is available on your PATH.

### Output file is empty
Check:
- documents contain readable text
- OCR succeeded
- model file exists in `tests/model`
- threshold is not too high
