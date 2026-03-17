# Data Ingestion & Pre-processing – Summary of Findings

This document summarizes the results and findings from **Notebook 2 – Data Ingestion** and **Notebook 3 – Pre-processing and EDA**, using the exact statistics and outputs produced when the notebooks were run. All exercises in both notebooks have been completed.

---

# Part A: Notebook 2 – Data Ingestion

## 1. Purpose and Pipeline

The notebook:

- Ingests PDFs from Cook County (and HTMLs from SEC Edgar) into usable text.
- Runs **OCR** (Tesseract) on PDFs and parses HTML for SEC Edgar files.
- Saves extracted text to `data/data_text` (`.pdf.txt`, `.html.txt`) and OCR position data to `.pdf.xlsx`.
- Performs **quality assessment** (OCR quality = dictionary-word rate; OCR confidence = Tesseract confidence).
- Builds **TFIDF** features, runs **K-Means** clustering, and visualizes with **PCA**.
- Explores relationships between document metadata (length, approval date) and OCR quality/confidence.

Paths used in the run:

- **HOME_DIRECTORY:** `/Users/affanali/Downloads/Middlesex/CST4012-MachineLearning/TC2R-CodeCompanion`
- **DATA_RAW:** `.../data/data_raw`
- **DATA_TEXT:** `.../data/data_text`

---

## 2. Document Counts and Corpus

- **Total documents in clustering:** **1,059** (sum of documents across all clusters).
- **Number of clusters:** **19** (`num_clusters=19` from silhouette-based choice).
- **Documents per cluster (exact counts from notebook output):**

| Cluster | Count | Cluster | Count | Cluster | Count |
|---------|-------|---------|-------|---------|-------|
| 0       | 17    | 7       | 115   | 14      | 53    |
| 1       | 36    | 8       | 53    | 15      | 59    |
| 2       | 32    | 9       | 83    | 16      | 100   |
| 3       | 80    | 10      | 26    | 17      | 39    |
| 4       | 102   | 11      | 17    | 18      | 33    |
| 5       | 27    | 12      | 35    |         |       |
| 6       | 16    | 13      | 58    |         |       |

Largest clusters: **7** (115), **4** (102), **16** (100), **9** (83), **3** (80). Smallest: **6** (16), **0** and **11** (17 each).

---

## 3. OCR Quality and Confidence

### 3.1 Quality dataframe (`df_qual`)

- Each document has **OCR Qual** (fraction of tokens that are dictionary words) and **OCR Conf** (Tesseract confidence, 0–100).
- **Low OCR quality examples** (OCR Qual &lt; 0.45) include:
  - `00010J1Z.pdf.xlsx`: OCR Qual **0.417**, OCR Conf **69.87**
  - `000030LG.pdf.xlsx`: OCR Qual **0.32**, OCR Conf **71.63**
  - `{3FA268E9-1763-4F99-A55F-CC66ED01491A}.pdf.xlsx`: OCR Qual **0.448**, OCR Conf **53.86**
  - `00003C1K.pdf.xlsx`: OCR Qual **0.449**, OCR Conf **69.84**
  - `{B6B03456-BA73-42AE-9106-B5BDCB9C78D7}.pdf.xlsx`: OCR Qual **0.434**, OCR Conf **67.83**

### 3.2 High-confidence documents (Exercise 2.2)

- **Documents with OCR Conf ≥ 80:** **48**.
- Examples of very high confidence: e.g. **86.86** (`00002JMP.pdf.xlsx`), **86.72** (`{8A1E379E-343A-4970-B9F9-4E38D5D3738A}.pdf.xlsx`).

---

## 4. Correlations: Document Length and Approval Year (Exercise 2.3)

**Exact correlation values from the notebook run:**

| Pair                         | Pearson ρ  |
|-----------------------------|------------|
| doc_length vs OCR Qual      | **−0.021** |
| doc_length vs OCR Conf      | **0.046**  |
| approval_year vs OCR Qual   | **0.005**  |
| approval_year vs OCR Conf   | **−0.176** |

**Findings:**

- **Document length:** Effectively **no** linear relationship with OCR Qual or OCR Conf (ρ ≈ −0.02 and 0.05). Length alone is not a useful proxy for quality; direct quality metrics are needed.
- **Approval year:** **No** correlation with OCR Qual (ρ = 0.005). **Weak negative** correlation with OCR Conf (ρ = −0.176): older contracts tend to have slightly lower Tesseract confidence (e.g. older scans, faxes, or photocopies).

---

## 5. Clustering and Cluster Themes (Exercise 2.4)

**Summary of cluster content from the notebook’s exploration:**

- **Cluster 0** (17 docs): Modification Summary Reports—roofing RFQ, general services.
- **Cluster 1** (36 docs): Modification Summary Reports—aviation, QAMT, reassignments.
- **Cluster 2** (32 docs): Delegate Agency Grant Agreements—Head Start, Early Head Start, child care.
- **Cluster 3** (80 docs): Contract Summary Sheets—CDBG, human services, youth development.
- **Cluster 4** (102 docs): Contract modifications and delegate agency grants (from main narrative).
- **Cluster 5** (27 docs): Modification Summary Reports—transportation, construction.

**Interpretation:**

- Clustering is **interpretable** and aligns with **document form/template**: Modification Summary vs Delegate Agency Grant vs Contract Summary.
- Clusters **0, 1, 5** are all modification-style but differ by department (general services, aviation, transportation).
- Clusters **2** and **3** separate clearly by agreement type (grant vs summary).
- Most clusters have a **single dominant category** and are cohesive; the split of modification documents across 0, 1, 5 is driven by departmental vocabulary rather than different document types.

---

## 6. Methodology Recap (Notebook 2)

- **OCR:** Tesseract on PDFs; text positions saved to `.pdf.xlsx`.
- **HTML:** Parsed (BeautifulSoup) and saved as `.html.txt`.
- **OCR quality:** Dictionary-word rate (OCR Qual) and Tesseract confidence (OCR Conf).
- **Clustering:** TFIDF → K-Means with **k = 19** (from silhouette scores) → PCA for 2D visualization.
- **Extended analysis:** Document length (character count from `.txt`), approval date/year from `Contracts_20231227.csv` joined by URL ID where available.

---

# Part B: Notebook 3 – Pre-processing and EDA

## 1. Purpose and Pipeline

The notebook:

- Ingests OCR-extracted text from `data/data_text` and segments it into **sentences** using **SpaCy** (`en_core_web_sm`).
- Builds a **sentence-level DataFrame** (`sentences_df`: `filename`, `sentence_index`, `sentence_text`) and optional **document-level** stats (`corpus_lengths`: `filename`, `document_length_chars`, `sentence_count`).
- Explores **document length** (characters and sentence count), **special character rate**, and **sentence length** distributions.
- Correlates **OCR quality metrics** (OCR Qual, OCR Conf) with: special character rate (doc and sentence level), average sentence length, and document date.
- Generates **word clouds** for the full corpus and for subsets by **Procurement Type** (Master agreements, RFPs, Bids) using the contracts CSV.

Paths:

- **HOME_DIRECTORY**, **data_text**, **data_pickle** (e.g. `corpus_sentences.pkl` for cached `sentences_df`).

---

## 2. Data and Setup

- **Corpus size:** **981 documents** (rows in `corpus_lengths`; `.pdf.txt` files in `data_text` used for EDA). This matches the subset of Cook County PDFs with BID/RFP/RFQ procurement types that were downloaded and ingested.
- **Sentence segmentation:** SpaCy pipeline; result stored in `sentences_df` (can be saved/loaded from `data_pickles/corpus_sentences.pkl`).
- **Document length exploration:** `plot_document_length_histograms(texts, sentences_df)` produces histograms of per-document character length and sentence count and returns `corpus_lengths`.
- **Special character rate:** `calculate_special_character_percentage(texts)` returns a list of per-document special-character percentages (excluding spaces and newlines); distribution plotted with `plot_special_character_distribution`.
- **OCR metrics in Notebook 3:** For exercises, OCR Qual is computed from texts (dictionary-word rate via SpaCy vocab); OCR Conf is read from `.xlsx` files in `data_text`. Merged per-document frame: `df_doc` (filename, special_pct, ocr_qual, ocr_conf).

---

## 3. Exercise 3.1 – Documents with Very Few or Many Sentences

**Objective:** Repeat exploration for documents with very few or many sentences to flag potential data-quality issues.

**Implementation:**

- **N_EXTREME = 5:** Top 5 and bottom 5 documents by sentence count from `corpus_lengths`.
- **Outputs:** Tables of documents with the **most** and **fewest** sentences; text previews (e.g. 1,500 characters) for a sample of each extreme.

**Findings:**

- Documents with the most sentences (e.g. very high `sentence_count` and `document_length_chars`) and those with the fewest are listed and inspected. Outliers can indicate OCR or segmentation issues; in the notebook run, some extremes were still valid (e.g. long contract text, short cover sheets).

**Status:** Solved.

---

## 4. Exercise 3.2 – OCR Quality vs Special Character Rate (Document and Sentence Level)

**Objective:** Explore correlation between OCR quality metrics and special character rate at document level and at sentence level.

**Implementation:**

- **Document level:** `df_doc` with `special_pct` (from `special_char_percentages`), `ocr_qual`, `ocr_conf`. Scatter plots and Pearson *r*.
- **Sentence level:** Per-sentence special character percentage; mean per document (`mean_sent_special_pct`); merge with `df_doc`; scatter plots and Pearson *r*.

**Exact results (from notebook output):**

| Level      | Pair                          | Pearson r  |
|-----------|--------------------------------|------------|
| Document  | special_pct vs OCR Qual        | **−0.857** |
| Document  | special_pct vs OCR Conf        | **−0.171** |
| Sentence  | mean_sent_special_pct vs OCR Qual | **0.002** |
| Sentence  | mean_sent_special_pct vs OCR Conf | **−0.010** |

**Findings:**

- **Document level:** **Strong negative** correlation between special character % and OCR Qual (r = −0.857): more special characters go with lower dictionary-word rate. **Weak negative** with OCR Conf (r = −0.171).
- **Sentence level:** Effectively **no** linear relationship (r ≈ 0.002 and −0.010); mean sentence-level special % does not add much beyond document-level for these metrics.

**Status:** Solved.

---

## 5. Exercise 3.3 – OCR Quality vs Average Sentence Length

**Objective:** Explore correlation between OCR quality metrics and average sentence length (words) across documents.

**Implementation:**

- Per-sentence word count from `sentence_text`; mean per document (`avg_sentence_length_words`); merge with `df_doc`; scatter plots and Pearson *r*.

**Exact results (from notebook output):**

| Pair                              | Pearson r  |
|-----------------------------------|------------|
| avg_sentence_length vs OCR Qual   | **0.773**  |
| avg_sentence_length vs OCR Conf   | **0.283**  |

**Findings:**

- **Strong positive** correlation between average sentence length (words) and OCR Qual (r = 0.773): documents with longer average sentences tend to have higher dictionary-word rate (e.g. more fluent, less fragmented or garbled text).
- **Moderate positive** with OCR Conf (r = 0.283).

**Status:** Solved.

---

## 6. Exercise 3.4 – OCR Quality vs Document Date

**Objective:** Explore correlation between OCR quality metrics and document date (approval year from contracts CSV).

**Implementation:**

- Merge `df_doc` with `Contracts_20231227.csv` via URL ID (from filename: strip `.pdf.txt`, match to Contract PDF–derived URL ID). Approval year from Approval Date. Scatter plots and Pearson *r* for documents with non-null approval year.

**Exact results (from notebook output):**

| Pair                        | Pearson r  |
|-----------------------------|------------|
| approval_year vs OCR Qual   | **−0.019** |
| approval_year vs OCR Conf   | **−0.176** |

- **Documents with date:** **981 of 981** (all corpus documents in this run had an approval year after merge).

**Findings:**

- **No** linear relationship between approval year and OCR Qual (r = −0.019).
- **Weak negative** between approval year and OCR Conf (r = −0.176), consistent with Notebook 2: older documents tend to have slightly lower Tesseract confidence.

**Status:** Solved.

---

## 7. Exercise 3.5 – Word Clouds by Subset (Master Agreements, RFPs, Bids)

**Objective:** Generate word clouds for (1) Master agreements, (2) RFPs, (3) Bids, using the original contracts CSV to subset by filename (URL ID).

**Implementation:**

- Load `Contracts_20231227.csv`; build URL ID from Contract PDF; subset by **Procurement Type**: `MASTER AGREEMENT`, `RFP`, `BID`. Map URL IDs to corpus filenames (`{URL_ID}.pdf.txt`). For each subset, filter `sentences_df` by those filenames and call `generate_word_cloud(df_sub, "sentence_text", title_string=...)`.

**Results (from notebook output):**

| Subset            | Status   | Sentences   | Documents |
|-------------------|----------|-------------|-----------|
| Master agreements | Skipped | —           | 0         |
| RFPs              | Generated| 256,817     | 296       |
| Bids              | Generated| 270,534     | 487       |

**Findings:**

- **Master agreements:** **No matching documents in corpus.** The current corpus was built from BID, RFP, and RFQ samples only (Notebook 1 download/ingestion); no Master agreement PDFs are present. The notebook prints an explanatory message and skips the word cloud.
- **RFPs and Bids:** Word clouds were generated; counts above are from the notebook output. Common themes (e.g. legal/administrative terms, contract-specific wording) can be compared across subsets for patterns and commonalities.

**Status:** Solved (with expected limitation for Master agreements given the corpus composition).

---

## 8. Additional Elements in Notebook 3

- **Word cloud for entire corpus:** `generate_word_cloud(sentences_df, 'sentence_text')` (single full-corpus word cloud).
- **Sentence length histograms:** `plot_sentence_length_histograms(sentences_df, 'sentence_text')` for distribution of sentence length (e.g. in words or characters, as implemented).
- **Outlier inspection:** Very long/short documents by character length are explored before Exercise 3.1; text previews support data-quality checks.

---

## 9. Methodology Recap (Notebook 3)

- **Sentence segmentation:** SpaCy `en_core_web_sm` (sentence boundary detection).
- **Document length:** Character count and sentence count per document from `sentences_df`; aggregated in `corpus_lengths`.
- **Special character rate:** Percentage of non-alphanumeric characters (excluding space and newline) per document and per sentence.
- **OCR metrics:** OCR Qual = dictionary-word rate (SpaCy vocab); OCR Conf = mean Tesseract confidence from `.pdf.xlsx` in `data_text`.
- **Date:** Approval year from `Contracts_20231227.csv` joined by URL ID (filename → URL ID via `.pdf.txt` / Contract PDF).
- **Word clouds:** `WordCloud` on concatenated `sentence_text` per (sub)set; subsets defined by Procurement Type and URL ID.

---

## 10. Quick Reference – Notebook 3

| Metric / result                                      | Value / finding        |
|------------------------------------------------------|-------------------------|
| Documents in EDA corpus                              | 981                     |
| special_pct vs OCR Qual (document)                   | r = −0.857              |
| special_pct vs OCR Conf (document)                   | r = −0.171              |
| mean_sent_special_pct vs OCR Qual (sentence-level)   | r = 0.002               |
| mean_sent_special_pct vs OCR Conf (sentence-level)   | r = −0.010              |
| avg_sentence_length vs OCR Qual                     | r = 0.773               |
| avg_sentence_length vs OCR Conf                     | r = 0.283               |
| approval_year vs OCR Qual                           | r = −0.019              |
| approval_year vs OCR Conf                           | r = −0.176              |
| Documents with approval year                         | 981 of 981              |
| RFP subset (word cloud)                              | 256,817 sentences, 296 docs |
| Bids subset (word cloud)                            | 270,534 sentences, 487 docs |
| Master agreements subset                             | 0 docs in corpus        |
| Exercise status                                      | 3.1–3.5 all solved      |

---

# Combined Quick Reference (Notebooks 2 & 3)

| Source    | Metric / result                    | Value / finding   |
|-----------|------------------------------------|-------------------|
| NB2       | Total documents (clustered)        | 1,059             |
| NB2       | Clusters                           | 19                |
| NB2       | doc_length vs OCR Qual (ρ)         | −0.021            |
| NB2       | doc_length vs OCR Conf (ρ)         | 0.046             |
| NB2       | approval_year vs OCR Qual (ρ)      | 0.005             |
| NB2       | approval_year vs OCR Conf (ρ)      | −0.176            |
| NB3       | Documents in EDA corpus            | 981               |
| NB3       | special_pct vs OCR Qual (r)        | −0.857            |
| NB3       | avg_sentence_length vs OCR Qual (r)| 0.773             |
| NB3       | approval_year vs OCR Qual (r)       | −0.019            |
| NB3       | approval_year vs OCR Conf (r)      | −0.176            |

All statistics above are taken directly from the executed outputs of **Notebook 2 – Data Ingestion** and **Notebook 3 – Pre-processing and EDA**.
