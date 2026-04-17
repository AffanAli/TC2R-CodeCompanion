import json
import logging
import os
import pickle
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence

import fitz
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
import spacy
from bs4 import BeautifulSoup
from PIL import Image
from pytesseract import Output

logger = logging.getLogger(__name__)

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_NLP_CACHE: Dict[tuple, Any] = {}


def get_spacy_model(exclude: Optional[List[str]] = None):
    """Load and cache a spaCy model instance."""
    key = tuple(sorted(exclude or []))
    if key not in _NLP_CACHE:
        _NLP_CACHE[key] = spacy.load("en_core_web_sm", exclude=exclude or [])
    return _NLP_CACHE[key]


def _words_to_number(text: str) -> Optional[int]:
    parts = re.split(r"[-\s]+", text)
    total = 0
    for part in parts:
        if part not in _NUMBER_WORDS:
            return None
        total += _NUMBER_WORDS[part]
    return total if total > 0 else None


# def parse_days_from_context(context: str) -> Optional[int]:

def parse_days_from_context(context: str) -> Optional[int]:
    """Extract the first explicit day count from text."""
    if context is None or not isinstance(context, str):
        return None

    try:
        if pd.isna(context):
            return None
    except Exception:
        pass

    text = context.strip().lower()
    if not text:
        return None

    # Numeric forms: 30 days, 14-day, 5 business days, 30 calendar days
    numeric_match = re.search(
        r"\b(\d+)\s*-?\s*(?:business\s+|calendar\s+)?day(?:s)?\b",
        text,
    )
    if numeric_match:
        return int(numeric_match.group(1))

    # Word-number forms: ten days, twenty-one days, thirty calendar days
    number_words_pattern = (
        r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
        r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
        r"eighty|ninety)"
    )

    word_match = re.search(
        rf"\b({number_words_pattern}(?:[-\s]{number_words_pattern})*)\s+"
        r"(?:business\s+|calendar\s+)?day(?:s)?\b",
        text,
    )
    if not word_match:
        return None

    return _words_to_number(word_match.group(1))
    """Extract the first explicit day count from text.

    Supported examples:
    - "within 30 days" -> 30
    - "a 14-day notice" -> 14
    - "after ten days" -> 10
    - "within 5 business days" -> 5
    - "payment due in thirty calendar days" -> 30
    """
    if context is None or pd.isna(context) or not isinstance(context, str):
        return None

    text = context.strip().lower()
    if not text:
        return None

    numeric_match = re.search(
        r"\b(\d+)\s*-?\s*(?:business\s+|calendar\s+)?day(?:s)?\b",
        text,
    )
    if numeric_match:
        return int(numeric_match.group(1))

    word_match = re.search(
        r"\b((?:[a-z]+(?:[-\s][a-z]+)*))\s+(?:business\s+|calendar\s+)?day(?:s)?\b",
        text,
    )
    if not word_match:
        return None

    return _words_to_number(word_match.group(1))


def pdf_to_text_with_ocr(pdf_path: str, output_txt_path: str) -> None:
    """Run OCR on a PDF and save extracted text and OCR positions.

    Inputs:
        pdf_path: Local path to a PDF file.
        output_txt_path: Output folder where .txt and .xlsx files will be written.

    Outputs:
        None. Side effects:
        - <filename>.txt
        - <filename>.xlsx
    """
    start = time.perf_counter()
    try:
        os.makedirs(output_txt_path, exist_ok=True)
        pdf_document = fitz.open(pdf_path)
        pdf_name = os.path.basename(pdf_path) + ".txt"
        excel_name = os.path.basename(pdf_path) + ".xlsx"

        text_content = ""
        positions_df = pd.DataFrame()

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            image = page.get_pixmap()
            img = Image.frombytes("RGB", (image.width, image.height), image.samples)

            page_text = pytesseract.image_to_string(img, lang="eng")
            positions = pytesseract.image_to_data(img, output_type=Output.DATAFRAME)

            text_content += page_text
            positions_df = pd.concat([positions_df, positions], ignore_index=True)

        with open(os.path.join(output_txt_path, pdf_name), "w", encoding="utf-8") as txt_file:
            txt_file.write(text_content)

        positions_df.to_excel(os.path.join(output_txt_path, excel_name), index=False)
        logger.info(
            "OCR completed for %s in %.2f seconds across %d pages",
            os.path.basename(pdf_path),
            time.perf_counter() - start,
            pdf_document.page_count,
        )
    except OSError as err:
        logger.exception("OS error during OCR for %s: %s", pdf_path, err)
        raise
    except ValueError as err:
        logger.exception("Invalid path during OCR for %s: %s", pdf_path, err)
        raise
    except Exception as err:
        logger.exception("Unexpected OCR error for %s: %s", pdf_path, err)
        raise


def pull_text_from_html(file_list: Sequence[str]) -> List[Optional[str]]:
    """Extract and clean text from HTML files.

    Returns one entry per input file. Files that cannot be read return None.
    """
    clean_texts: List[Optional[str]] = []
    for file_path in file_list:
        html_string = ""
        try:
            with open(file_path, "r", encoding="windows-1252") as file:
                html_string = file.read()
        except Exception as e1:
            logger.debug("windows-1252 read failed for %s: %s", file_path, e1)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    html_string = file.read()
            except Exception as e2:
                logger.warning("Could not read HTML file %s: %s", file_path, e2)

        if html_string:
            clean_text = " ".join(BeautifulSoup(html_string, "html.parser").stripped_strings)
            clean_texts.append(clean_text)
        else:
            clean_texts.append(None)
    return clean_texts


def read_text_files(folder_path: str):
    """Read .txt files from a folder, ignoring other file types.

    Returns:
        texts: list[str]
        files: list[str]
    """
    texts: List[str] = []
    files: List[str] = []

    if not os.path.isdir(folder_path):
        logger.warning("Text folder does not exist: %s", folder_path)
        return texts, files

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                texts.append(file.read())
                files.append(filename)
        else:
            logger.info("Skipping non-text file during text read: %s", filename)
    return texts, files


def calculate_ocr_quality(texts: Sequence[str]) -> List[float]:
    """Estimate OCR quality using recognized English-like tokens / total tokens."""
    nlp = get_spacy_model()
    scores: List[float] = []

    for text in texts:
        doc = nlp(text)
        english_word_count = sum(1 for token in doc if token.is_alpha and token.text.lower() in nlp.vocab)
        total_word_count = len(doc)
        quality_score = english_word_count / total_word_count if total_word_count > 0 else 0.0
        scores.append(float(quality_score))

    return scores


def plot_ocr_quality_histogram(ocr_quality_scores: Sequence[float], output_folder: str) -> Optional[str]:
    """Plot and save a histogram of OCR quality scores."""
    if not ocr_quality_scores:
        logger.warning("No OCR quality scores provided; histogram will not be generated.")
        return None

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "OCR_quality_distribution.png")

    plt.figure()
    plt.hist(ocr_quality_scores, bins=20, range=(0, 1), edgecolor="black", alpha=0.7)
    plt.title("OCR Quality Histogram")
    plt.xlabel("OCR Quality Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info("OCR histogram saved to %s", output_path)
    return output_path


def process_texts_to_dataframe(texts: Sequence[str], filenames: Sequence[str]) -> pd.DataFrame:
    """Split texts into sentences and return a normalized DataFrame."""
    if len(texts) != len(filenames):
        raise ValueError("texts and filenames must have the same length")

    nlp = get_spacy_model(exclude=["parser"])
    if "sentencizer" not in nlp.pipe_names:
        config = {"punct_chars": ["\n\n", ".", "?", "!"]}
        nlp.add_pipe("sentencizer", config=config)

    rows = {"filename": [], "sentence_index": [], "sentence_text": []}
    for filename, text in zip(filenames, texts):
        if text is None:
            continue
        for i, sentence in enumerate(nlp(str(text)).sents):
            sent_text = sentence.text.strip()
            if sent_text:
                rows["filename"].append(filename)
                rows["sentence_index"].append(i)
                rows["sentence_text"].append(sent_text)

    return pd.DataFrame(rows)


def run_classification_model(
    df: pd.DataFrame,
    model_folder: str,
    model_name: str = "ml_classifier_gbc.pkl",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Run a classifier over sentence embeddings and keep the top sentence per file."""
    required_cols = {"filename", "sentence_index", "sentence_text", "Embedding"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {sorted(missing)}")

    model_path = os.path.join(model_folder, model_name)
    with open(model_path, "rb") as f:
        clf_model = pickle.load(f)

    working_df = df.copy()
    working_df["Probability"] = clf_model.predict_proba(working_df["Embedding"].to_list())[:, 1]
    top_doc_preds_indices = working_df.groupby("filename")["Probability"].idxmax()
    top_doc_preds = working_df.loc[top_doc_preds_indices].reset_index(drop=True)

    mask = top_doc_preds["Probability"] < threshold
    cols_to_mask = ["sentence_index", "sentence_text", "Embedding", "Probability"]
    top_doc_preds.loc[mask, cols_to_mask] = pd.NA
    top_doc_preds["below_threshold"] = mask.astype(bool)
    return top_doc_preds


def _serialize_value(value: Any) -> Any:
    """Safely convert complex values for SQLite/JSON storage."""
    if value is None:
        return None

    # Handle pandas missing values safely
    if value is pd.NA:
        return None

    # Handle numpy arrays / pandas arrays before pd.isna
    if hasattr(value, "tolist"):
        return json.dumps(value.tolist())

    # Handle Python containers
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)

    # Handle scalar missing values
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    # Keep normal scalar types as-is
    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)

def prepare_results_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Create an export-friendly copy of the results DataFrame."""
    export_df = df.copy()
    for col in export_df.columns:
        export_df[col] = export_df[col].apply(_serialize_value)
    return export_df


def save_results_to_excel(df: pd.DataFrame, output_folder: str, output_name: str = "model_results.xlsx") -> str:
    """Save results DataFrame to Excel."""
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_name)
    prepare_results_for_export(df).to_excel(output_path, index=False)
    logger.info("Excel results saved to %s", output_path)
    return output_path


def save_results_to_pickle(df: pd.DataFrame, output_folder: str, output_name: str = "model_results.pkl") -> str:
    """Save results DataFrame to pickle."""
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_name)
    with open(output_path, "wb") as f:
        pickle.dump(df, f)
    logger.info("Pickle results saved to %s", output_path)
    return output_path


def save_results_to_sqlite(
    df: pd.DataFrame,
    db_path: str,
    table_name: str = "model_results",
    if_exists: str = "replace",
) -> str:
    """Save results DataFrame to a SQLite database table."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        prepare_results_for_export(df).to_sql(table_name, conn, if_exists=if_exists, index=False)
    logger.info("SQLite results saved to %s [table=%s]", db_path, table_name)
    return db_path


def save_run_summary(summary: Dict[str, Any], output_folder: str, output_name: str = "run_summary.json") -> str:
    """Save a lightweight JSON summary of a pipeline run."""
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Run summary saved to %s", output_path)
    return output_path
