# CLCPP

This repository contains an anonymized export of a multi-lingual chemical Question-Answer-Context (QAC) pipeline, together with the dataset snapshots and annotation workbook used for evaluation.

The code supports building patent-centered QAC retrieval data, preparing benchmark-style corpus/query/qrels files, and running retrieval evaluation workflows.

## Layout

- `code/` contains the Python implementation and prompt templates.
- `data/multi-lingual-qac-chem-patents/` contains the first dataset snapshot in parquet format.
- `data/multi-lingual-qac-epo/` contains the second dataset snapshot in parquet format.
- `data/annotations/evaluated_annotations.xlsx` contains the annotation workbook.

## Setup

```bash
cd code
uv sync
```

## Quick Check

```bash
uv run main.py --help
```

This command verifies that the Python environment and CLI entrypoint load correctly. It prints the available pipeline and evaluation options.

## Data

The included data snapshots are already present under `data/`, so they can be inspected or loaded directly without downloading from external sources.

Some pipeline modes create new data or run evaluations and may require additional local credentials, API keys, or model downloads depending on the selected command-line options.

