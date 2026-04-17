import os
import pickle
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from helper_functions import (
    parse_days_from_context,
    pull_text_from_html,
    read_text_files,
    calculate_ocr_quality,
    plot_ocr_quality_histogram,
    process_texts_to_dataframe,
    run_classification_model,
    save_results_to_pickle,
    save_results_to_sqlite,
    pdf_to_text_with_ocr,
    save_results_to_excel,
    save_run_summary,
    prepare_results_for_export,
)
from pipeline import process_and_classify_files


class DummyClassifier:
    def predict_proba(self, X):
        rows = []
        for emb in X:
            score = float(emb[0])
            score = max(0.0, min(1.0, score))
            rows.append([1 - score, score])
        return np.array(rows)


class DummySentenceModel:
    def encode(self, sentence):
        text = sentence.lower()
        if "30 days" in text:
            return np.array([0.95, 0.05])
        if "ten days" in text or "10 days" in text:
            return np.array([0.85, 0.15])
        if "deadline" in text:
            return np.array([0.65, 0.35])
        return np.array([0.20, 0.80])


class BaseTempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tc2r_tests_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestParseDaysFromContext(unittest.TestCase):
    def test_numeric_days(self):
        self.assertEqual(parse_days_from_context("within 30 days"), 30)
        self.assertEqual(parse_days_from_context("a 14-day notice"), 14)
        self.assertEqual(parse_days_from_context("response due in 1 day"), 1)

    def test_word_days(self):
        self.assertEqual(parse_days_from_context("after ten days"), 10)
        self.assertEqual(parse_days_from_context("twenty-one days remaining"), 21)
        self.assertEqual(parse_days_from_context("submit after ninety days"), 90)

    def test_invalid_cases(self):
        self.assertIsNone(parse_days_from_context("no deadline mentioned"))
        self.assertIsNone(parse_days_from_context(""))
        self.assertIsNone(parse_days_from_context(None))


class TestPullTextFromHtml(BaseTempDirTest):
    def test_successful_execution(self):
        file1 = os.path.join(self.tmpdir, "test1.html")
        file2 = os.path.join(self.tmpdir, "test2.html")
        with open(file1, "w", encoding="utf-8") as f:
            f.write("<html><body><p>This is test content.</p></body></html>")
        with open(file2, "w", encoding="utf-8") as f:
            f.write("<html><body><h1>Another test</h1><p>And more content.</p></body></html>")

        extracted = pull_text_from_html([file1, file2])
        self.assertEqual(len(extracted), 2)
        self.assertIn("This is test content.", extracted)
        self.assertIn("Another test And more content.", extracted)

    def test_missing_or_empty_files(self):
        empty_file = os.path.join(self.tmpdir, "empty.html")
        missing_file = os.path.join(self.tmpdir, "missing.html")
        open(empty_file, "w", encoding="utf-8").close()

        extracted = pull_text_from_html([empty_file, missing_file])
        self.assertEqual(extracted, [None, None])


class TestReadTextFiles(BaseTempDirTest):
    def test_reads_only_text_files(self):
        with open(os.path.join(self.tmpdir, "file1.txt"), "w", encoding="utf-8") as f:
            f.write("alpha")
        with open(os.path.join(self.tmpdir, "file2.txt"), "w", encoding="utf-8") as f:
            f.write("beta")
        with open(os.path.join(self.tmpdir, "ignore.csv"), "w", encoding="utf-8") as f:
            f.write("x,y")

        texts, files = read_text_files(self.tmpdir)
        self.assertEqual(len(texts), 2)
        self.assertEqual(set(files), {"file1.txt", "file2.txt"})
        self.assertIn("alpha", texts)
        self.assertIn("beta", texts)

    def test_handles_missing_folder(self):
        texts, files = read_text_files(os.path.join(self.tmpdir, "missing"))
        self.assertEqual(texts, [])
        self.assertEqual(files, [])


class TestCalculateOcrQuality(unittest.TestCase):
    @patch("helper_functions.get_spacy_model")
    def test_returns_float_scores(self, mock_get_spacy_model):
        class Token:
            def __init__(self, text, is_alpha=True):
                self.text = text
                self.is_alpha = is_alpha

        class FakeDoc(list):
            pass

        class FakeNLP:
            vocab = {"this": True, "is": True, "clean": True, "text": True}

            def __call__(self, text):
                if text == "This is clean text":
                    return FakeDoc([Token("This"), Token("is"), Token("clean"), Token("text")])
                return FakeDoc([Token("xzyq"), Token("123", is_alpha=False)])

        mock_get_spacy_model.return_value = FakeNLP()
        scores = calculate_ocr_quality(["This is clean text", "xzyq 123"])
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(isinstance(x, float) for x in scores))
        self.assertGreater(scores[0], scores[1])


class TestPlotOcrQualityHistogram(BaseTempDirTest):
    def test_creates_histogram_file(self):
        output = plot_ocr_quality_histogram([0.1, 0.5, 0.9], self.tmpdir)
        self.assertTrue(os.path.exists(output))

    def test_empty_scores_returns_none(self):
        output = plot_ocr_quality_histogram([], self.tmpdir)
        self.assertIsNone(output)


class TestProcessTextsToDataFrame(unittest.TestCase):
    @patch("helper_functions.get_spacy_model")
    def test_dataframe_shape_and_columns(self, mock_get_spacy_model):
        class FakeSent:
            def __init__(self, text):
                self.text = text

        class FakeDoc:
            def __init__(self, text):
                self.sents = [FakeSent(x.strip()) for x in text.split(".") if x.strip()]

        class FakeNLP:
            pipe_names = []

            def add_pipe(self, *args, **kwargs):
                self.pipe_names.append("sentencizer")

            def __call__(self, text):
                return FakeDoc(text)

        mock_get_spacy_model.return_value = FakeNLP()
        df = process_texts_to_dataframe(["Sentence one. Sentence two."], ["doc1.txt"])
        self.assertEqual(list(df.columns), ["filename", "sentence_index", "sentence_text"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["filename"], "doc1.txt")

    def test_raises_on_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            process_texts_to_dataframe(["text1"], ["f1", "f2"])


class TestRunClassificationModel(BaseTempDirTest):
    def setUp(self):
        super().setUp()
        self.model_dir = os.path.join(self.tmpdir, "model")
        os.makedirs(self.model_dir, exist_ok=True)
        with open(os.path.join(self.model_dir, "ml_classifier_gbc.pkl"), "wb") as f:
            pickle.dump(DummyClassifier(), f)

    def test_returns_top_sentence_per_file(self):
        df = pd.DataFrame(
            {
                "filename": ["a.txt", "a.txt", "b.txt"],
                "sentence_index": [0, 1, 0],
                "sentence_text": ["low", "high", "medium"],
                "Embedding": [np.array([0.2]), np.array([0.9]), np.array([0.7])],
            }
        )
        results = run_classification_model(df, self.model_dir, threshold=0.5)
        self.assertEqual(len(results), 2)
        self.assertEqual(results.loc[results["filename"] == "a.txt", "sentence_text"].iloc[0], "high")

    def test_masks_low_probability_rows(self):
        df = pd.DataFrame(
            {
                "filename": ["a.txt"],
                "sentence_index": [0],
                "sentence_text": ["low"],
                "Embedding": [np.array([0.2])],
            }
        )
        results = run_classification_model(df, self.model_dir, threshold=0.5)
        self.assertTrue(pd.isna(results.iloc[0]["sentence_text"]))
        self.assertTrue(pd.isna(results.iloc[0]["Probability"]))

    def test_raises_when_required_columns_missing(self):
        df = pd.DataFrame({"filename": ["a.txt"]})
        with self.assertRaises(ValueError):
            run_classification_model(df, self.model_dir)


class TestOutputHelpers(BaseTempDirTest):
    def test_save_results_to_pickle(self):
        df = pd.DataFrame({"a": [1, 2]})
        output_path = save_results_to_pickle(df, self.tmpdir, "results.pkl")
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, "rb") as f:
            loaded = pickle.load(f)
        pd.testing.assert_frame_equal(df, loaded)

    def test_save_results_to_sqlite(self):
        df = pd.DataFrame({"filename": ["a.txt"], "parsed_days": [30]})
        db_path = os.path.join(self.tmpdir, "results.db")
        save_results_to_sqlite(df, db_path)
        self.assertTrue(os.path.exists(db_path))
        with sqlite3.connect(db_path) as conn:
            loaded = pd.read_sql_query("SELECT * FROM model_results", conn)
        self.assertEqual(int(loaded.loc[0, "parsed_days"]), 30)


class TestPipelineEndToEnd(BaseTempDirTest):
    def setUp(self):
        super().setUp()
        self.input_dir = os.path.join(self.tmpdir, "tests", "docs")
        self.output_dir = os.path.join(self.tmpdir, "tests", "output")
        self.model_dir = os.path.join(self.tmpdir, "tests", "model")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        with open(os.path.join(self.input_dir, "doc1.html"), "w", encoding="utf-8") as f:
            f.write("<html><body><p>Please respond within 30 days.</p><p>General text.</p></body></html>")
        with open(os.path.join(self.input_dir, "doc2.htm"), "w", encoding="utf-8") as f:
            f.write("<html><body><p>You have ten days to reply.</p></body></html>")
        with open(os.path.join(self.input_dir, "skip.csv"), "w", encoding="utf-8") as f:
            f.write("ignore me")

        with open(os.path.join(self.model_dir, "ml_classifier_gbc.pkl"), "wb") as f:
            pickle.dump(DummyClassifier(), f)

    @patch("pipeline.calculate_ocr_quality", return_value=[0.8, 0.9])
    def test_pipeline_excel_output(self, _mock_scores):
        results = process_and_classify_files(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            model_folder=self.model_dir,
            sent_emb_model=DummySentenceModel(),
            threshold=0.5,
            output_format="excel",
        )
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "model_results.xlsx")))
        self.assertIn("parsed_days", results.columns)
        parsed_map = dict(zip(results["filename"], results["parsed_days"]))
        self.assertEqual(parsed_map["doc1.txt"], 30)
        self.assertEqual(parsed_map["doc2.txt"], 10)

    @patch("pipeline.calculate_ocr_quality", return_value=[0.8, 0.9])
    def test_pipeline_pickle_output(self, _mock_scores):
        process_and_classify_files(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            model_folder=self.model_dir,
            sent_emb_model=DummySentenceModel(),
            threshold=0.5,
            output_format="pickle",
            output_name="custom_results.pkl",
        )
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "custom_results.pkl")))

    @patch("pipeline.calculate_ocr_quality", return_value=[0.8, 0.9])
    def test_pipeline_sqlite_output(self, _mock_scores):
        process_and_classify_files(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            model_folder=self.model_dir,
            sent_emb_model=DummySentenceModel(),
            threshold=0.5,
            output_format="sqlite",
            output_name="custom_results.db",
        )
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "custom_results.db")))

    def test_pipeline_raises_on_empty_input(self):
        empty_input = os.path.join(self.tmpdir, "empty_input")
        os.makedirs(empty_input, exist_ok=True)
        with self.assertRaises(ValueError):
            process_and_classify_files(
                input_folder=empty_input,
                output_folder=self.output_dir,
                model_folder=self.model_dir,
                sent_emb_model=DummySentenceModel(),
                threshold=0.5,
                output_format="excel",
            )

class TestPdfToTextWithOcr(BaseTempDirTest):
    def test_successful_execution(self):
        pdf_path = os.path.join(self.tmpdir, "sample.pdf")
        output_dir = os.path.join(self.tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Payment due within 30 days")
        doc.save(pdf_path)
        doc.close()

        pdf_to_text_with_ocr(pdf_path, output_dir)

        expected_txt = os.path.join(output_dir, "sample.pdf.txt")
        expected_xlsx = os.path.join(output_dir, "sample.pdf.xlsx")

        self.assertTrue(os.path.exists(expected_txt))
        self.assertTrue(os.path.exists(expected_xlsx))
        self.assertGreater(os.path.getsize(expected_txt), 0)
        self.assertGreater(os.path.getsize(expected_xlsx), 0)

    def test_missing_pdf_raises(self):
        output_dir = os.path.join(self.tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        with self.assertRaises(Exception):
            pdf_to_text_with_ocr(os.path.join(self.tmpdir, "missing.pdf"), output_dir)


class TestExportHelpersAdditional(BaseTempDirTest):
    def test_save_results_to_excel(self):
        df = pd.DataFrame(
            {
                "filename": ["a.txt"],
                "sentence_text": ["within 30 days"],
                "parsed_days": [30],
            }
        )
        output_path = save_results_to_excel(df, self.tmpdir, "results.xlsx")
        self.assertTrue(os.path.exists(output_path))

        loaded = pd.read_excel(output_path)
        self.assertEqual(loaded.loc[0, "filename"], "a.txt")
        self.assertEqual(int(loaded.loc[0, "parsed_days"]), 30)

    def test_save_run_summary(self):
        summary = {
            "input_folder": "tests/docs",
            "output_format": "excel",
            "document_result_count": 2,
        }
        output_path = save_run_summary(summary, self.tmpdir)
        self.assertTrue(os.path.exists(output_path))

        import json
        with open(output_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        self.assertEqual(loaded["output_format"], "excel")
        self.assertEqual(loaded["document_result_count"], 2)

    def test_prepare_results_for_export_handles_arrays_and_na(self):
        df = pd.DataFrame(
            {
                "filename": ["a.txt", "b.txt"],
                "Embedding": [np.array([0.1, 0.2]), pd.NA],
                "parsed_days": [30, pd.NA],
            }
        )
        export_df = prepare_results_for_export(df)

        self.assertIsInstance(export_df.loc[0, "Embedding"], str)
        self.assertTrue(pd.isna(export_df.loc[1, "Embedding"]))

        self.assertEqual(export_df.loc[0, "parsed_days"], 30)
        self.assertTrue(pd.isna(export_df.loc[1, "parsed_days"]))


class TestParseDaysAdditional(unittest.TestCase):
    def test_business_and_calendar_days(self):
        self.assertEqual(parse_days_from_context("payment due within 5 business days"), 5)
        self.assertEqual(parse_days_from_context("payment due in thirty calendar days"), 30)


class TestRunClassificationModelAdditional(BaseTempDirTest):
    def setUp(self):
        super().setUp()
        self.model_dir = os.path.join(self.tmpdir, "model")
        os.makedirs(self.model_dir, exist_ok=True)
        with open(os.path.join(self.model_dir, "ml_classifier_gbc.pkl"), "wb") as f:
            pickle.dump(DummyClassifier(), f)

    def test_below_threshold_column_exists_and_is_correct(self):
        df = pd.DataFrame(
            {
                "filename": ["a.txt", "b.txt"],
                "sentence_index": [0, 0],
                "sentence_text": ["low", "high"],
                "Embedding": [np.array([0.2]), np.array([0.9])],
            }
        )
        results = run_classification_model(df, self.model_dir, threshold=0.5)

        self.assertIn("below_threshold", results.columns)

        below_map = dict(zip(results["filename"], results["below_threshold"]))
        self.assertTrue(bool(below_map["a.txt"]))
        self.assertFalse(bool(below_map["b.txt"]))


class TestPipelineValidation(BaseTempDirTest):
    def setUp(self):
        super().setUp()
        self.input_dir = os.path.join(self.tmpdir, "input")
        self.output_dir = os.path.join(self.tmpdir, "output")
        self.model_dir = os.path.join(self.tmpdir, "model")

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        with open(os.path.join(self.model_dir, "ml_classifier_gbc.pkl"), "wb") as f:
            pickle.dump(DummyClassifier(), f)

    def test_pipeline_raises_when_input_folder_missing(self):
        with self.assertRaises(ValueError):
            process_and_classify_files(
                input_folder=os.path.join(self.tmpdir, "does_not_exist"),
                output_folder=self.output_dir,
                model_folder=self.model_dir,
                sent_emb_model=DummySentenceModel(),
                threshold=0.5,
                output_format="excel",
            )

    def test_pipeline_raises_when_model_folder_missing(self):
        with self.assertRaises(ValueError):
            process_and_classify_files(
                input_folder=self.input_dir,
                output_folder=self.output_dir,
                model_folder=os.path.join(self.tmpdir, "missing_model_dir"),
                sent_emb_model=DummySentenceModel(),
                threshold=0.5,
                output_format="excel",
            )

    def test_pipeline_raises_on_invalid_output_format(self):
        with self.assertRaises(ValueError):
            process_and_classify_files(
                input_folder=self.input_dir,
                output_folder=self.output_dir,
                model_folder=self.model_dir,
                sent_emb_model=DummySentenceModel(),
                threshold=0.5,
                output_format="csv",
            )

    def test_pipeline_raises_when_sentence_model_missing(self):
        with self.assertRaises(RuntimeError):
            process_and_classify_files(
                input_folder=self.input_dir,
                output_folder=self.output_dir,
                model_folder=self.model_dir,
                sent_emb_model=None,
                threshold=0.5,
                output_format="excel",
            )


class TestPipelineArtifacts(BaseTempDirTest):
    def setUp(self):
        super().setUp()
        self.input_dir = os.path.join(self.tmpdir, "tests", "docs")
        self.output_dir = os.path.join(self.tmpdir, "tests", "output")
        self.model_dir = os.path.join(self.tmpdir, "tests", "model")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        with open(os.path.join(self.input_dir, "doc1.html"), "w", encoding="utf-8") as f:
            f.write("<html><body><p>Please respond within 30 days.</p></body></html>")

        with open(os.path.join(self.model_dir, "ml_classifier_gbc.pkl"), "wb") as f:
            pickle.dump(DummyClassifier(), f)

    @patch("pipeline.calculate_ocr_quality", return_value=[0.8])
    def test_pipeline_creates_run_summary(self, _mock_scores):
        process_and_classify_files(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            model_folder=self.model_dir,
            sent_emb_model=DummySentenceModel(),
            threshold=0.5,
            output_format="excel",
        )

        summary_path = os.path.join(self.output_dir, "run_summary.json")
        self.assertTrue(os.path.exists(summary_path))

        import json
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertEqual(summary["discovered_html_count"], 1)
        self.assertEqual(summary["processed_text_file_count"], 1)
        self.assertEqual(summary["parsed_days_count"], 1)

if __name__ == "__main__":
    unittest.main()
