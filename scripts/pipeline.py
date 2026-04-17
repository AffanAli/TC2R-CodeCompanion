import argparse
import glob
import logging
import os
import pickle
from typing import Dict, Optional, Tuple

from sentence_transformers import SentenceTransformer

try:
    from .helper_functions import (
        calculate_ocr_quality,
        parse_days_from_context,
        pdf_to_text_with_ocr,
        plot_ocr_quality_histogram,
        process_texts_to_dataframe,
        pull_text_from_html,
        read_text_files,
        run_classification_model,
        save_results_to_excel,
        save_results_to_pickle,
        save_results_to_sqlite,
        save_run_summary,
    )
except ImportError:
    from helper_functions import (
        calculate_ocr_quality,
        parse_days_from_context,
        pdf_to_text_with_ocr,
        plot_ocr_quality_histogram,
        process_texts_to_dataframe,
        pull_text_from_html,
        read_text_files,
        run_classification_model,
        save_results_to_excel,
        save_results_to_pickle,
        save_results_to_sqlite,
        save_run_summary,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_INPUT_EXTENSIONS = {".pdf", ".html", ".htm"}
SUPPORTED_OUTPUT_FORMATS = {"excel", "pickle", "sqlite"}

try:
    sent_emb_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    logger.info("SentenceTransformer model loaded successfully.")
except Exception as e:
    logger.error("Error loading SentenceTransformer model: %s", e)
    sent_emb_model = None


def _discover_files(input_folder: str) -> Tuple[list, list, list]:
    pdf_files = sorted(glob.glob(os.path.join(input_folder, "*.pdf")))
    html_files = sorted(glob.glob(os.path.join(input_folder, "*.html")) + glob.glob(os.path.join(input_folder, "*.htm")))
    all_files = sorted(glob.glob(os.path.join(input_folder, "*")))
    skipped_files = [
        f for f in all_files
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() not in SUPPORTED_INPUT_EXTENSIONS
    ]
    return pdf_files, html_files, skipped_files


def _save_results(df_model_results, output_folder: str, output_format: str, output_name: Optional[str] = None):
    if output_format == "excel":
        file_name = output_name or "model_results.xlsx"
        return save_results_to_excel(df_model_results, output_folder, file_name)
    if output_format == "pickle":
        file_name = output_name or "model_results.pkl"
        return save_results_to_pickle(df_model_results, output_folder, file_name)
    if output_format == "sqlite":
        file_name = output_name or "model_results.db"
        db_path = os.path.join(output_folder, file_name)
        return save_results_to_sqlite(df_model_results, db_path, table_name="model_results")
    raise ValueError(f"Unsupported output format: {output_format}")


def _build_run_summary(
    input_folder: str,
    output_folder: str,
    model_folder: str,
    output_format: str,
    threshold: float,
    pdf_files: list,
    html_files: list,
    skipped_files: list,
    texts: list,
    filenames: list,
    df_sentences,
    df_model_results,
    output_path: str,
) -> Dict[str, object]:
    return {
        "input_folder": input_folder,
        "output_folder": output_folder,
        "model_folder": model_folder,
        "output_format": output_format,
        "threshold": threshold,
        "discovered_pdf_count": len(pdf_files),
        "discovered_html_count": len(html_files),
        "discovered_skipped_count": len(skipped_files),
        "processed_text_file_count": len(texts),
        "processed_filenames": filenames,
        "sentence_row_count": len(df_sentences),
        "document_result_count": len(df_model_results),
        "parsed_days_count": int(df_model_results["parsed_days"].notna().sum()) if "parsed_days" in df_model_results else 0,
        "below_threshold_count": int(df_model_results["below_threshold"].sum()) if "below_threshold" in df_model_results else 0,
        "final_output_path": output_path,
    }


def process_and_classify_files(
    input_folder: str,
    output_folder: str,
    model_folder: str,
    sent_emb_model=sent_emb_model,
    threshold: float = 0.5,
    output_format: str = "excel",
    output_name: Optional[str] = None,
):
    """End-to-end document processing pipeline.

    Inputs:
        input_folder: folder containing .pdf/.html/.htm documents
        output_folder: folder where outputs should be written
        model_folder: folder containing ml_classifier_gbc.pkl
        sent_emb_model: sentence embedding model instance
        threshold: probability threshold for accepting a document result
        output_format: excel, pickle, or sqlite
        output_name: optional output filename

    Output:
        A pandas DataFrame containing one top result row per document.
    """
    if sent_emb_model is None:
        raise RuntimeError("SentenceTransformer model is not available.")
    if not os.path.isdir(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")
    if not os.path.isdir(model_folder):
        raise ValueError(f"Model folder does not exist: {model_folder}")
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {sorted(SUPPORTED_OUTPUT_FORMATS)}")

    logger.info("Starting file processing for input folder: %s", input_folder)
    os.makedirs(output_folder, exist_ok=True)
    text_dir = os.path.join(output_folder, "text_files")
    os.makedirs(text_dir, exist_ok=True)
    logger.info("Ensured text files directory exists at: %s", text_dir)

    pdf_files, html_files, skipped_files = _discover_files(input_folder)
    logger.info("Discovered %d PDF files, %d HTML files, %d skipped files.", len(pdf_files), len(html_files), len(skipped_files))
    if pdf_files:
        logger.info("PDF files to process: %s", [os.path.basename(x) for x in pdf_files])
    if html_files:
        logger.info("HTML files to process: %s", [os.path.basename(x) for x in html_files])
    if skipped_files:
        logger.info("Unsupported files skipped: %s", [os.path.basename(x) for x in skipped_files])

    logger.info("Ingesting files...")
    for pdf_file in pdf_files:
        try:
            pdf_to_text_with_ocr(pdf_file, text_dir)
            logger.info("Successfully processed PDF file: %s", pdf_file)
        except Exception as e:
            logger.error("Error processing PDF file %s: %s", pdf_file, e)

    html_texts = pull_text_from_html(html_files)
    html_saved_count = 0
    for html_file, text_content in zip(html_files, html_texts):
        if text_content:
            file_name = os.path.splitext(os.path.basename(html_file))[0] + ".txt"
            output_file_path = os.path.join(text_dir, file_name)
            try:
                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
                html_saved_count += 1
                logger.info("Successfully processed HTML file: %s and saved to %s", html_file, output_file_path)
            except Exception as e:
                logger.error("Error saving HTML content from %s to %s: %s", html_file, output_file_path, e)
        else:
            logger.warning("No content extracted from HTML file: %s. Skipping save.", html_file)
    logger.info("Saved %d/%d HTML-derived text files.", html_saved_count, len(html_files))

    logger.info("Reading ingested text files...")
    texts, filenames = read_text_files(text_dir)
    logger.info("Read %d text files.", len(texts))
    if not texts:
        raise ValueError("No text files were generated/read from the input folder.")

    df = process_texts_to_dataframe(texts, filenames)
    logger.info("Processed texts into DataFrame with %d sentence rows.", len(df))

    logger.info("Generating sentence embeddings for %d rows...", len(df))
    df["Embedding"] = df["sentence_text"].apply(lambda x: sent_emb_model.encode(x))
    logger.info("Generated sentence embeddings for %d rows.", len(df))

    data_df_path = os.path.join(output_folder, "data_df.pkl")
    with open(data_df_path, "wb") as pkl:
        pickle.dump(df, pkl)
    logger.info("DataFrame with embeddings saved to: %s", data_df_path)

    logger.info("Calculating and plotting OCR quality...")
    ocr_scores = calculate_ocr_quality(texts)
    hist_path = plot_ocr_quality_histogram(ocr_scores, output_folder)
    logger.info("OCR quality histogram generated for %d documents.", len(ocr_scores))
    if hist_path:
        logger.info("OCR histogram available at: %s", hist_path)

    logger.info("Running classification model...")
    df_model_results = run_classification_model(df, model_folder, threshold=threshold)
    logger.info(
        "Classification model run successfully. %d result rows returned; %d are below threshold.",
        len(df_model_results),
        int(df_model_results["below_threshold"].sum()) if "below_threshold" in df_model_results else 0,
    )

    logger.info("Parsing day counts from top document context...")
    df_model_results["parsed_days"] = df_model_results["sentence_text"].apply(parse_days_from_context)
    parsed_count = int(df_model_results["parsed_days"].notna().sum())
    logger.info("Parsed day values for %d/%d documents.", parsed_count, len(df_model_results))

    output_path = _save_results(df_model_results, output_folder, output_format, output_name)
    run_summary = _build_run_summary(
        input_folder=input_folder,
        output_folder=output_folder,
        model_folder=model_folder,
        output_format=output_format,
        threshold=threshold,
        pdf_files=pdf_files,
        html_files=html_files,
        skipped_files=skipped_files,
        texts=texts,
        filenames=filenames,
        df_sentences=df,
        df_model_results=df_model_results,
        output_path=output_path,
    )
    save_run_summary(run_summary, output_folder)

    logger.info("File processing and classification pipeline finished. Final output: %s", output_path)
    return df_model_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PDF/HTML files, calculate OCR quality, run a classification model, and save results."
    )
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing PDF and HTML files.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder where outputs will be saved.")
    parser.add_argument("--model_folder", type=str, required=True, help="Path to the directory containing the trained model.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold below which model predictions are masked.")
    parser.add_argument(
        "--output_format",
        type=str,
        default="excel",
        choices=sorted(SUPPORTED_OUTPUT_FORMATS),
        help="Output format: excel, pickle, or sqlite."
    )
    parser.add_argument("--output_name", type=str, default=None, help="Optional output filename. Example: model_results.xlsx")
    args = parser.parse_args()

    logger.info("--- Starting File Processing and Classification Pipeline ---")
    results_df = process_and_classify_files(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        model_folder=args.model_folder,
        sent_emb_model=sent_emb_model,
        threshold=args.threshold,
        output_format=args.output_format,
        output_name=args.output_name,
    )
    logger.info("--- Pipeline Execution Finished ---")
    logger.info("Results DataFrame head:\n%s", results_df.head())
