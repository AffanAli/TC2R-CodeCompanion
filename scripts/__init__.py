from .pipeline import process_and_classify_files
from .helper_functions import (
    parse_days_from_context,
    pdf_to_text_with_ocr,
    pull_text_from_html,
    read_text_files,
    calculate_ocr_quality,
    plot_ocr_quality_histogram,
    process_texts_to_dataframe,
    run_classification_model,
    save_results_to_pickle,
    save_results_to_sqlite,
)

__version__ = "0.2.0"
