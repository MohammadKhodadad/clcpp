from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.multi_lingual_qac.config import PipelineConfig, PipelinePaths
from src.multi_lingual_qac.pipeline import run_pipeline


def _normalize_hf_dataset_repo(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    marker = "huggingface.co/datasets/"
    if marker in raw:
        raw = raw.split(marker, 1)[1]
    return raw.strip().strip("/")


def _normalize_mteb_variant(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in {"multilingual", "cross_language"}:
        raise argparse.ArgumentTypeError(
            "Unsupported MTEB variant. Use `multilingual` or `cross_language`."
        )
    return normalized


def _resolve_eval_models(values: list[str] | None) -> tuple[str, ...]:
    """Resolve `--evaluate-mteb MODEL...`: flag absent -> (); no models or `all`
    -> the curated ALIAS_GRAPH_MODELS set; else the given ids with any `all` token
    expanded, order-preserving de-dupe. (Lazy import keeps non-eval commands light.)"""
    if values is None:
        return ()
    from src.multi_lingual_qac.mteb.evaluation import ALIAS_GRAPH_MODELS

    if not values or [v.lower() for v in values] == ["all"]:
        return tuple(ALIAS_GRAPH_MODELS)
    out: list[str] = []
    for v in values:
        out.extend(ALIAS_GRAPH_MODELS if v.lower() == "all" else [v])
    return tuple(dict.fromkeys(out))


def _resolve_results_dir(project_root: Path, mteb_results_dir: str | None) -> Path:
    """Where to read existing results from: explicit dir, else the latest run, else legacy."""
    if mteb_results_dir:
        return Path(mteb_results_dir)
    latest = project_root / "reports" / "runs" / "latest"
    if latest.exists():
        return latest
    return project_root / "reports" / "mteb"


def _dataset_from_run_metadata(results_dir: Path) -> tuple[str | None, str | None]:
    """Read (dataset_repo, dataset_variant) from a run's run_metadata.json, if present.

    Lets the analysis step target whatever dataset the run was evaluated against,
    so the user does not have to re-specify --mteb-dataset-repo.
    """
    import json

    meta_path = Path(results_dir) / "run_metadata.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    return meta.get("dataset_repo"), meta.get("dataset_variant")


def parse_args() -> PipelineConfig:
    default_mteb_dataset_repo = "anonymous/multi-lingual-qac-chem-patents"
    parser = argparse.ArgumentParser(
        description="Multi-Lingual Chemical QAC: extract patents, preprocess to CSV."
    )
    parser.add_argument("--yes", "-y", action="store_true", help="No prompts; redo all")
    parser.add_argument("--no-extraction", action="store_true", help="Skip extraction; only preprocess")
    parser.add_argument("--limit", type=int, default=None, help="Max patents per language (if omitted in interactive mode, you will be prompted)")
    parser.add_argument("--qa-sample", type=int, default=None, help="Sample size for Q&A generation (if omitted in interactive mode, you will be prompted; 0 = skip Q&A)")
    parser.add_argument("--qa-batch", action="store_true", help="Batch QA generation using worker threads based on available CPUs")
    parser.add_argument("--qa-no-batch", action="store_true", help="Disable batch QA generation")
    parser.add_argument("--push-hf", action="store_true", help="Push corpus + QAC to Hugging Face Hub")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face repo ID (e.g. username/multi-lingual-chemical-qac); required if --push-hf")
    parser.add_argument(
        "--evaluate-mteb",
        nargs="*",
        metavar="MODEL",
        help=(
            "Evaluate embedding models against the HF retrieval dataset via MTEB. "
            "Pass specific HF model ids, or `all` (or no models) to run the curated "
            "10-model set (ALIAS_GRAPH_MODELS: Qwen3-Embedding-0.6B, e5-large-instruct, "
            "bge-m3, jina-v3, gte-multilingual-base, embeddinggemma-300m, granite-278m, "
            "jina-colbert-v2, SapBERT-XLMR, LaBSE)."
        ),
    )
    parser.add_argument(
        "--mteb-dataset-repo",
        type=str,
        default=default_mteb_dataset_repo,
        help=(
            "Hugging Face dataset repo to evaluate with MTEB "
            f"(default: {default_mteb_dataset_repo})"
        ),
    )
    parser.add_argument(
        "--mteb-corpus-repo",
        type=str,
        default="owner/multilingual-corpus",
        help=(
            "Shared corpus repo used as the retrieval haystack for every evaluation "
            "(default: owner/multilingual-corpus). Queries/qrels still come from "
            "--mteb-dataset-repo. Pass an empty string to use the dataset's own corpus config."
        ),
    )
    parser.add_argument(
        "--mteb-variant",
        type=_normalize_mteb_variant,
        default="multilingual",
        metavar="{multilingual,cross_language}",
        help=(
            "Which retrieval subset to evaluate. `multilingual` uses the `qrels` config "
            "(every doc sharing a publication_number is a positive, including the source-language "
            "doc). `cross_language` uses `cross_language-qrels` (only foreign-language docs are positives). "
            "Default: multilingual."
        ),
    )
    parser.add_argument(
        "--mteb-output-dir",
        type=str,
        default=None,
        help="Directory for MTEB results and summary reports (default: reports/mteb)",
    )
    parser.add_argument(
        "--mteb-batch-size",
        type=int,
        default=32,
        help="Batch size passed to sentence-transformers encoding during MTEB evaluation",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="During --evaluate-mteb, save per-query rankings to <output-dir>/predictions/<model> "
        "(needed for --analyze-questions)",
    )
    parser.add_argument(
        "--analyze-questions",
        action="store_true",
        help="Produce a question-level analysis (Recall@10/MRR by query language, query origin, "
        "same/cross-language mode, language-pair matrix) from saved per-query predictions. "
        "Implies --save-predictions when combined with --evaluate-mteb; otherwise reads existing "
        "predictions under --mteb-results-dir/predictions.",
    )
    parser.add_argument(
        "--mteb-analysis-dir",
        type=str,
        default=None,
        help="Output directory for the question-level analysis (default: <output-dir>/question_analysis)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable the PNG plots that --analyze-questions writes to <run>/question_analysis/plots/",
    )
    parser.add_argument(
        "--query-metadata",
        type=str,
        default=None,
        metavar="CSV",
        help="CSV holding per-question `mode`/`strategy` (joined to queries by question text). "
        "Only needed when the dataset's queries config doesn't already carry those columns.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        metavar="LABEL",
        help="Optional label appended to the auto UTC timestamp run id, e.g. --run-id add-zh "
        "produces reports/runs/20260601-143052_add-zh/. Without it the run id is just the timestamp.",
    )
    parser.add_argument(
        "--generate-mteb-tables",
        action="store_true",
        help="Generate model comparison tables from saved MTEB results without rerunning evaluation",
    )
    parser.add_argument(
        "--mteb-results-dir",
        type=str,
        default=None,
        help="Directory containing saved MTEB results and summary.json (default: reports/mteb)",
    )
    parser.add_argument(
        "--mteb-tables-dir",
        type=str,
        default=None,
        help="Directory for generated comparison tables (default: reports/mteb_tables)",
    )
    parser.add_argument(
        "--upload-mteb-results",
        action="store_true",
        help="Upload generated MTEB comparison tables to a Hugging Face dataset repo",
    )
    parser.add_argument(
        "--mteb-upload-repo",
        type=str,
        default=None,
        help="Hugging Face dataset repo ID or URL for uploaded MTEB comparison tables",
    )
    parser.add_argument(
        "--epo-ingest",
        action="store_true",
        help="Stream the next BDDS item(s) of EP full-text data, filter to chemistry multilingual rows, append to data/EPO/multilingual_corpus.csv",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="With --epo-ingest, how many BDDS items to process in sequence (default: 1)",
    )
    parser.add_argument(
        "--chemistry-strict",
        action="store_true",
        help="With --epo-ingest, keep only docs whose chemistry signal comes from CPC/IPC classification (drop title-keyword-only matches)",
    )
    parser.add_argument(
        "--build-alias-graph",
        action="store_true",
        help="Build the Alias-Graph Retrieval benchmark: pick ChEBI concepts, find gold docs + "
        "taxonomic-neighbor hard negatives in a corpus CSV, write one CSV per concept + a manifest.",
    )
    parser.add_argument(
        "--alias-corpus",
        type=str,
        default="data/google_patents/multilingual_corpus.csv",
        metavar="CSV",
        help="Corpus CSV to search for gold/hard-negative documents "
        "(default: data/google_patents/multilingual_corpus.csv)",
    )
    parser.add_argument(
        "--alias-output-dir",
        type=str,
        default=None,
        help="Output directory for per-concept CSVs and manifest (default: data/alias_graph)",
    )
    parser.add_argument(
        "--chebi-variant",
        type=str,
        default="full",
        choices=["full", "core", "lite"],
        help="ChEBI OBO release to use. Only `full` carries synonyms (needed for balanced "
        "matching); `core`/`lite` are lighter, primary-name-only. Default: full.",
    )
    parser.add_argument(
        "--alias-langs",
        nargs="+",
        default=["zh", "en", "de", "fr", "es"],
        help="Target languages for Wikipedia names (default: zh en de fr es)",
    )
    parser.add_argument(
        "--alias-no-wikipedia",
        action="store_true",
        help="Skip the ChEBI->Wikidata->Wikipedia name bridge (KG names only).",
    )
    parser.add_argument(
        "--alias-min-gold",
        type=int,
        default=2,
        help="Minimum gold documents a concept must have to qualify (default: 2)",
    )
    parser.add_argument(
        "--alias-min-neg",
        type=int,
        default=3,
        help="Minimum hard-negative documents a concept must have to qualify (default: 3)",
    )
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=None,
        help="With --build-alias-graph, cap the number of concept files written "
        "(most-attested concepts first). Default: no cap.",
    )
    parser.add_argument(
        "--alias-max-df",
        type=float,
        default=0.02,
        help="Drop any concept name appearing in more than this fraction of documents "
        "as a corpus stopword (default: 0.02).",
    )
    parser.add_argument(
        "--alias-include-non-molecular",
        action="store_true",
        help="Do not restrict main concepts to the ChEBI molecular-entity subtree "
        "(by default role/group/atom classes are excluded).",
    )
    parser.add_argument(
        "--alias-include-classes",
        action="store_true",
        help="Allow broad class concepts (those with is_a children) as main concepts. "
        "By default only specific leaf concepts (e.g. named compounds) are used.",
    )
    parser.add_argument(
        "--check-wiki-names",
        action="store_true",
        help="Validate the Wikipedia-derived names: for a concept mentioned in an English "
        "doc, check whether its Wikipedia title for language L appears in the parallel "
        "L-language translation of the same patent. Writes reports/wiki_name_quality/.",
    )
    parser.add_argument(
        "--export-concept",
        type=str,
        default=None,
        metavar="CHEBI_ID",
        help="Materialize one concept's gold + hard-negative documents (joined from the "
        "corpus, all languages) to a CSV for inspection, e.g. --export-concept CHEBI:2942. "
        "Reads data/alias_graph/alias_graph.json.",
    )
    parser.add_argument(
        "--alias-generate-qa",
        action="store_true",
        help="Generate one concept-centric technical query per selected document from "
        "alias_graph.json: the query describes the concept without naming it and the answer "
        "is the concept. All language variants of the document feed the prompt. Reuses the "
        "faithfulness + technical-quality verifiers. Writes <alias-output-dir>/qac/concept_qa.csv.",
    )
    parser.add_argument(
        "--alias-qa-strategy",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Query-language strategy: 1=random any, 2=random missing, 3=random existing, "
        "4=all languages (default: 1).",
    )
    parser.add_argument(
        "--alias-qa-model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model for concept-query generation and grading (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--alias-qa-seed",
        type=int,
        default=42,
        help="Random seed for concept-query generation (default: 42)",
    )
    parser.add_argument(
        "--alias-qa-limit",
        type=int,
        default=None,
        help="Select documents balanced by language: about N/5 documents existing in each of "
        "en/de/fr/es/zh (soft cap; a language with fewer eligible docs is warned and taken "
        "as-is). One query per document; query language is chosen by --alias-qa-strategy. The "
        "total may be below N since a multilingual document counts for every language it "
        "contains. Omit to use every eligible document.",
    )
    parser.add_argument(
        "--alias-qa-workers",
        type=int,
        default=1,
        help="Worker threads for concept-query generation (default: 1)",
    )
    parser.add_argument(
        "--build-code-switched",
        action="store_true",
        help="Idea 2: build code-switched document variants (A-F) from alias_graph.json + corpus, "
        "with change-tracking columns. Writes data/code_switched/code_switched_corpus.csv.",
    )
    parser.add_argument(
        "--cs-variants",
        type=str,
        default="A,B,C,D,F",
        help="Comma-separated variants to emit: A(baseline) B(in-set) C(out-of-set) "
        "D(noisy) E(non-chem, LLM) F(chebi). Default: A,B,C,D,F (E excluded).",
    )
    parser.add_argument(
        "--cs-limit",
        type=int,
        default=None,
        help="With --build-code-switched, process only the first N concepts.",
    )
    parser.add_argument(
        "--cs-model",
        type=str,
        default="gpt-5-mini",
        help="LLM for variant E (default: gpt-5-mini).",
    )
    parser.add_argument(
        "--cs-seed",
        type=int,
        default=42,
        help="Random seed for code-switching (default: 42).",
    )
    parser.add_argument(
        "--cs-output-dir",
        type=str,
        default=None,
        help="Output directory for the variant corpus (default: data/code_switched).",
    )
    parser.add_argument(
        "--cs-generate-qa",
        action="store_true",
        help="Generate questions for the code-switched variants: B/C/D/F use the ORIGINAL "
        "term verbatim (gold = the variant doc); E is a normal doc QA. Reads "
        "<cs-output-dir>/code_switched_corpus.csv, writes code_switched_qac.csv.",
    )
    parser.add_argument("--cs-qa-model", type=str, default="gpt-5-mini",
                        help="LLM for variant QA generation/grading (default: gpt-5-mini).")
    parser.add_argument("--cs-qa-seed", type=int, default=42, help="Seed for variant QA (default: 42).")
    parser.add_argument("--cs-qa-limit", type=int, default=None,
                        help="Process only the first N B/C/D/F groups (and N E docs).")
    parser.add_argument("--cs-qa-workers", type=int, default=1,
                        help="Worker threads for variant QA (default: 1).")
    # ---- Progressive code-switching: TWO commands ---------------------------- #
    # (1) DATA CREATION: build the ladder corpus, generate the fixed queries, and push
    #     the dataset to HF in MTEB retrieval format (corpus/queries/qrels).
    parser.add_argument(
        "--create-progressive-data",
        action="store_true",
        help="DATA COMMAND. Progressive code-switching end-to-end: build the cumulative-ladder "
        "corpus (clean -> 1 -> ... -> N swaps, random B/C/D/F per step), generate one fixed query "
        "per base doc (about the step-1 term), and push the dataset to HF (corpus/queries/qrels). "
        "Use --hf-dry-run to write parquet locally instead of uploading.",
    )
    # (2) EVALUATION: run embedding models against the published dataset + shared haystack.
    parser.add_argument(
        "--eval-progressive-cs",
        action="store_true",
        help="EVAL COMMAND. Read the published progressive dataset (--pcs-hf-repo) + the shared "
        "corpus haystack (--mteb-corpus-repo), encode each ladder variant, and measure how retrieval "
        "rank/score decays with depth. Writes the decay curve + plot under reports/runs/progressive_cs.",
    )
    parser.add_argument("--pcs-hf-repo", type=str, default="owner/progressive-code-switch",
                        help="HF dataset repo (or local dry-run dir) for the progressive benchmark.")
    parser.add_argument("--pcs-steps", type=int, default=5,
                        help="Number of cumulative replacement steps / ladder depth (default: 5).")
    parser.add_argument("--pcs-modes", type=str, default="B,C,D,F",
                        help="Comma-separated swap modes to draw from (default: B,C,D,F; E excluded).")
    parser.add_argument("--pcs-seed", type=int, default=42,
                        help="Random seed for progressive code-switching (default: 42).")
    parser.add_argument("--pcs-limit", type=int, default=None,
                        help="Process only the first N base docs when building the corpus.")
    parser.add_argument("--pcs-output-dir", type=str, default=None,
                        help="Working dir for the progressive CSVs (default: data/progressive_cs).")
    parser.add_argument("--pcs-qa-model", type=str, default="gpt-5-mini",
                        help="LLM for progressive query GENERATION (default: gpt-5-mini, OpenAI).")
    parser.add_argument("--pcs-grader-model", type=str, default="anthropic/claude-sonnet-4.5",
                        help="LLM for the two FEEDBACK verifiers, via OpenRouter "
                        "(default: anthropic/claude-sonnet-4.5).")
    parser.add_argument("--pcs-qa-strategy", type=int, default=4, choices=[1, 2, 3, 4],
                        help="Query-language strategy (alias-graph): 1=random_any, 2=random_missing, "
                        "3=random_existing, 4=all (one query per language). Default: 4.")
    parser.add_argument("--pcs-qa-seed", type=int, default=42, help="Seed for progressive QA (default: 42).")
    parser.add_argument("--pcs-qa-limit", type=int, default=None,
                        help="Generate queries for only the first N base docs.")
    parser.add_argument("--pcs-qa-workers", type=int, default=1,
                        help="Worker threads for progressive query generation (default: 1).")
    parser.add_argument("--pcs-eval-models", type=str, nargs="*", default=None,
                        help="Embedding models for the progressive eval (default: ALIAS_GRAPH_MODELS).")
    parser.add_argument("--pcs-eval-limit", type=int, default=None,
                        help="Evaluate only the first N base docs (haystack still full).")
    parser.add_argument("--pcs-eval-batch-size", type=int, default=32,
                        help="Encoding batch size for the progressive eval (default: 32).")
    # Granular sub-steps (optional; --create-progressive-data runs all three in order):
    parser.add_argument("--build-progressive-cs", action="store_true",
                        help="Sub-step: only build the ladder corpus CSV (no QA, no push).")
    parser.add_argument("--progressive-cs-qa", action="store_true",
                        help="Sub-step: only generate the fixed queries CSV from an existing corpus.")
    parser.add_argument("--push-progressive-cs", action="store_true",
                        help="Sub-step: only push the existing corpus+queries CSVs to HF.")
    parser.add_argument(
        "--push-alias-graph-hf",
        action="store_true",
        help="Publish the Alias-Graph Retrieval benchmark (alias_graph.json + concept_qa.csv + corpus) "
        "to Hugging Face as corpus/queries/qrels/hard_negatives/qac/concepts configs.",
    )
    parser.add_argument("--alias-hf-repo", type=str,
                        default="owner/multi-lingual-qac-alias-graph",
                        help="Target HF dataset repo for --push-alias-graph-hf.")
    parser.add_argument("--hf-dry-run", action="store_true",
                        help="With --push-alias-graph-hf, write parquet locally instead of uploading.")
    parser.add_argument("--hf-private", action="store_true", help="Create the HF dataset repo as private.")
    parser.add_argument(
        "--push-corpus-hf",
        action="store_true",
        help="Publish the full patent corpus (data/google_patents/multilingual_corpus.csv) as a "
        "single-`corpus`-config HF dataset (the shared retrieval haystack). Honors --hf-dry-run/--hf-private.",
    )
    parser.add_argument("--corpus-hf-repo", type=str, default="owner/multilingual-corpus",
                        help="Target HF dataset repo for --push-corpus-hf (default: owner/multilingual-corpus).")
    parser.add_argument(
        "--analyze-confusion",
        action="store_true",
        help="With --evaluate-mteb (or --analyze-questions on an existing run): also compute, per "
        "query language, how often a confusable wrong compound (hard negative) outranks the right "
        "one (gold), from the saved per-query predictions. Writes <run>/confusion/ (+ plot). "
        "Implies prediction saving.",
    )
    args = parser.parse_args()
    qa_batch = None
    if args.qa_batch and args.qa_no_batch:
        parser.error("Use only one of --qa-batch or --qa-no-batch")
    if args.qa_batch:
        qa_batch = True
    elif args.qa_no_batch:
        qa_batch = False
    return PipelineConfig(
        yes=args.yes,
        no_extraction=args.no_extraction,
        limit=args.limit,
        qa_sample=args.qa_sample,
        qa_batch=qa_batch,
        push_hf=args.push_hf,
        hf_repo=args.hf_repo,
        evaluate_mteb_models=_resolve_eval_models(args.evaluate_mteb),
        mteb_dataset_repo=args.mteb_dataset_repo,
        mteb_corpus_repo=args.mteb_corpus_repo,
        mteb_dataset_variant=args.mteb_variant,
        mteb_output_dir=args.mteb_output_dir,
        mteb_batch_size=max(1, args.mteb_batch_size),
        mteb_save_predictions=args.save_predictions,
        analyze_questions=args.analyze_questions,
        mteb_analysis_dir=args.mteb_analysis_dir,
        mteb_no_plots=args.no_plots,
        mteb_query_metadata=args.query_metadata,
        run_id_label=args.run_id,
        generate_mteb_tables=args.generate_mteb_tables,
        mteb_results_dir=args.mteb_results_dir,
        mteb_tables_dir=args.mteb_tables_dir,
        upload_mteb_results=args.upload_mteb_results,
        mteb_upload_repo=args.mteb_upload_repo,
        epo_ingest=args.epo_ingest,
        epo_num_batches=max(1, args.num_batches),
        epo_chemistry_strict=args.chemistry_strict,
        build_alias_graph=args.build_alias_graph,
        alias_corpus=args.alias_corpus,
        alias_output_dir=args.alias_output_dir,
        chebi_variant=args.chebi_variant,
        alias_langs=tuple(args.alias_langs),
        alias_use_wikipedia=not args.alias_no_wikipedia,
        alias_min_gold=args.alias_min_gold,
        alias_min_neg=args.alias_min_neg,
        alias_max_concepts=args.max_concepts,
        alias_max_df=args.alias_max_df,
        alias_molecular_only=not args.alias_include_non_molecular,
        alias_leaf_only=not args.alias_include_classes,
        check_wiki_names=args.check_wiki_names,
        export_concept=args.export_concept,
        alias_generate_qa=args.alias_generate_qa,
        alias_qa_strategy=args.alias_qa_strategy,
        alias_qa_model=args.alias_qa_model,
        alias_qa_seed=args.alias_qa_seed,
        alias_qa_limit=args.alias_qa_limit,
        alias_qa_workers=args.alias_qa_workers,
        build_code_switched=args.build_code_switched,
        cs_variants=args.cs_variants,
        cs_limit=args.cs_limit,
        cs_model=args.cs_model,
        cs_seed=args.cs_seed,
        cs_output_dir=args.cs_output_dir,
        cs_generate_qa=args.cs_generate_qa,
        cs_qa_model=args.cs_qa_model,
        cs_qa_seed=args.cs_qa_seed,
        cs_qa_limit=args.cs_qa_limit,
        cs_qa_workers=args.cs_qa_workers,
        create_progressive_data=args.create_progressive_data,
        eval_progressive_cs=args.eval_progressive_cs,
        pcs_hf_repo=args.pcs_hf_repo,
        pcs_steps=args.pcs_steps,
        pcs_modes=args.pcs_modes,
        pcs_seed=args.pcs_seed,
        pcs_limit=args.pcs_limit,
        pcs_output_dir=args.pcs_output_dir,
        pcs_qa_model=args.pcs_qa_model,
        pcs_grader_model=args.pcs_grader_model,
        pcs_qa_strategy=args.pcs_qa_strategy,
        pcs_qa_seed=args.pcs_qa_seed,
        pcs_qa_limit=args.pcs_qa_limit,
        pcs_qa_workers=args.pcs_qa_workers,
        pcs_eval_models=tuple(args.pcs_eval_models) if args.pcs_eval_models else (),
        pcs_eval_limit=args.pcs_eval_limit,
        pcs_eval_batch_size=args.pcs_eval_batch_size,
        build_progressive_cs=args.build_progressive_cs,
        progressive_cs_qa=args.progressive_cs_qa,
        push_progressive_cs=args.push_progressive_cs,
        push_alias_graph_hf=args.push_alias_graph_hf,
        alias_hf_repo=args.alias_hf_repo,
        hf_dry_run=args.hf_dry_run,
        hf_private=args.hf_private,
        analyze_confusion=args.analyze_confusion,
        push_corpus_hf=args.push_corpus_hf,
        corpus_hf_repo=args.corpus_hf_repo,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    config = parse_args()

    if config.push_corpus_hf:
        from src.multi_lingual_qac.export.hf_upload import push_corpus_to_hub

        paths = PipelinePaths.from_project_root(project_root)
        push_corpus_to_hub(
            corpus_csv=paths.multilingual_corpus_csv,
            repo_id=config.corpus_hf_repo,
            private=config.hf_private,
            dry_run=config.hf_dry_run,
        )
        return

    if config.push_alias_graph_hf:
        from src.alias_graph.hf_export import push_alias_graph_to_hub

        paths = PipelinePaths.from_project_root(project_root)
        corpus_csv = (
            Path(config.alias_corpus)
            if config.alias_corpus
            else paths.multilingual_corpus_csv
        )
        alias_dir = (
            Path(config.alias_output_dir)
            if config.alias_output_dir
            else paths.alias_graph_dir
        )
        push_alias_graph_to_hub(
            alias_json=alias_dir / "alias_graph.json",
            qac_csv=alias_dir / "qac" / "concept_qa.csv",
            corpus_csv=corpus_csv,
            repo_id=config.alias_hf_repo,
            private=config.hf_private,
            dry_run=config.hf_dry_run,
            chebi_cache_dir=paths.chebi_dir,
        )
        return

    if config.build_code_switched:
        from src.alias_graph.code_switch import run_code_switch

        paths = PipelinePaths.from_project_root(project_root)
        corpus_csv = (
            Path(config.alias_corpus)
            if config.alias_corpus
            else paths.multilingual_corpus_csv
        )
        output_dir = (
            Path(config.cs_output_dir)
            if config.cs_output_dir
            else paths.code_switched_dir
        )
        run_code_switch(
            alias_json=paths.alias_graph_dir / "alias_graph.json",
            corpus_path=corpus_csv,
            output_path=output_dir / "code_switched_corpus.csv",
            variants=[v.strip() for v in config.cs_variants.split(",") if v.strip()],
            limit=config.cs_limit,
            model=config.cs_model,
            seed=config.cs_seed,
        )
        return

    if config.cs_generate_qa:
        from src.alias_graph.qac_generation import run_variant_qa

        paths = PipelinePaths.from_project_root(project_root)
        corpus_csv = (
            Path(config.alias_corpus)
            if config.alias_corpus
            else paths.multilingual_corpus_csv
        )
        output_dir = (
            Path(config.cs_output_dir)
            if config.cs_output_dir
            else paths.code_switched_dir
        )
        run_variant_qa(
            corpus_csv=output_dir / "code_switched_corpus.csv",
            source_corpus=corpus_csv,
            alias_json=paths.alias_graph_dir / "alias_graph.json",
            output_path=output_dir / "code_switched_qac.csv",
            model=config.cs_qa_model,
            seed=config.cs_qa_seed,
            limit=config.cs_qa_limit,
            workers=config.cs_qa_workers,
        )
        return

    # ---- Progressive code-switching: data-creation + evaluation -------------- #
    pcs_data = config.create_progressive_data
    if pcs_data or config.build_progressive_cs or config.progressive_cs_qa or config.push_progressive_cs:
        paths = PipelinePaths.from_project_root(project_root)
        source_corpus = (
            Path(config.alias_corpus) if config.alias_corpus else paths.multilingual_corpus_csv
        )
        output_dir = (
            Path(config.pcs_output_dir) if config.pcs_output_dir else paths.progressive_cs_dir
        )
        corpus_out = output_dir / "progressive_corpus.csv"
        qac_out = output_dir / "progressive_qac.csv"

        if pcs_data or config.build_progressive_cs:
            from src.alias_graph.progressive_code_switch import run_progressive_code_switch
            run_progressive_code_switch(
                alias_json=paths.alias_graph_dir / "alias_graph.json",
                corpus_path=source_corpus,
                output_path=corpus_out,
                n_steps=config.pcs_steps,
                modes=[m.strip() for m in config.pcs_modes.split(",") if m.strip()],
                limit=config.pcs_limit,
                seed=config.pcs_seed,
            )

        if pcs_data or config.progressive_cs_qa:
            from src.alias_graph.qac_generation import run_progressive_qa
            run_progressive_qa(
                corpus_csv=corpus_out,
                source_corpus=source_corpus,
                alias_json=paths.alias_graph_dir / "alias_graph.json",
                output_path=qac_out,
                model=config.pcs_qa_model,
                grader_model=config.pcs_grader_model,
                strategy=config.pcs_qa_strategy,
                seed=config.pcs_qa_seed,
                limit=config.pcs_qa_limit,
                workers=config.pcs_qa_workers,
            )

        if pcs_data or config.push_progressive_cs:
            from src.alias_graph.progressive_hf import push_progressive_to_hub
            push_progressive_to_hub(
                corpus_csv=corpus_out,
                qac_csv=qac_out,
                repo_id=config.pcs_hf_repo,
                private=config.hf_private,
                dry_run=config.hf_dry_run,
            )
        return

    if config.eval_progressive_cs:
        from src.multi_lingual_qac.progressive.eval import run_progressive_eval

        paths = PipelinePaths.from_project_root(project_root)
        output_dir = (
            Path(config.pcs_output_dir) if config.pcs_output_dir else paths.progressive_cs_dir
        )
        run_progressive_eval(
            dataset_repo=config.pcs_hf_repo,
            haystack_repo=config.mteb_corpus_repo,
            output_dir=project_root / "reports" / "runs" / "progressive_cs",
            models=config.pcs_eval_models or None,
            limit=config.pcs_eval_limit,
            batch_size=config.pcs_eval_batch_size,
            emb_cache_dir=output_dir / "emb",
        )
        return

    if config.alias_generate_qa:
        from src.alias_graph.qac_generation import run_concept_qa

        paths = PipelinePaths.from_project_root(project_root)
        corpus_csv = (
            Path(config.alias_corpus)
            if config.alias_corpus
            else paths.multilingual_corpus_csv
        )
        output_dir = (
            Path(config.alias_output_dir)
            if config.alias_output_dir
            else paths.alias_graph_dir
        )
        run_concept_qa(
            alias_json=output_dir / "alias_graph.json",
            corpus_path=corpus_csv,
            output_path=output_dir / "qac" / "concept_qa.csv",
            strategy=config.alias_qa_strategy,
            model=config.alias_qa_model,
            seed=config.alias_qa_seed,
            limit=config.alias_qa_limit,
            workers=config.alias_qa_workers,
        )
        return

    if config.export_concept:
        from src.alias_graph import export_concept

        paths = PipelinePaths.from_project_root(project_root)
        corpus_csv = (
            Path(config.alias_corpus)
            if config.alias_corpus
            else paths.multilingual_corpus_csv
        )
        output_dir = (
            Path(config.alias_output_dir)
            if config.alias_output_dir
            else paths.alias_graph_dir
        )
        try:
            export_concept(
                json_path=output_dir / "alias_graph.json",
                corpus_csv=corpus_csv,
                chebi_id=config.export_concept,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(
                f"Cannot export {config.export_concept}: {exc}\n"
                "(It may have been filtered out -- see data/alias_graph/manifest.csv for available concepts.)"
            )
        return

    if config.build_alias_graph:
        from src.alias_graph import build_alias_graph

        paths = PipelinePaths.from_project_root(project_root)
        corpus_csv = (
            Path(config.alias_corpus)
            if config.alias_corpus
            else paths.multilingual_corpus_csv
        )
        output_dir = (
            Path(config.alias_output_dir)
            if config.alias_output_dir
            else paths.alias_graph_dir
        )
        build_alias_graph(
            corpus_csv=corpus_csv,
            output_dir=output_dir,
            chebi_cache_dir=paths.chebi_dir,
            variant=config.chebi_variant,
            langs=config.alias_langs,
            use_wikipedia=config.alias_use_wikipedia,
            min_gold=config.alias_min_gold,
            min_neg=config.alias_min_neg,
            max_concepts=config.alias_max_concepts,
            max_df_ratio=config.alias_max_df,
            molecular_only=config.alias_molecular_only,
            leaf_only=config.alias_leaf_only,
        )
        return

    if config.check_wiki_names:
        from src.alias_graph import check_wiki_name_quality

        paths = PipelinePaths.from_project_root(project_root)
        corpus_csv = (
            Path(config.alias_corpus)
            if config.alias_corpus
            else paths.multilingual_corpus_csv
        )
        output_dir = (
            Path(config.alias_output_dir)
            if config.alias_output_dir
            else paths.wiki_quality_dir
        )
        check_wiki_name_quality(
            corpus_csv=corpus_csv,
            chebi_cache_dir=paths.chebi_dir,
            output_dir=output_dir,
            langs=config.alias_langs,
            variant=config.chebi_variant,
        )
        return

    if config.epo_ingest:
        from src.multi_lingual_qac.dataloaders.epo_bdds import ingest_n_batches

        paths = PipelinePaths.from_project_root(project_root)
        ingest_n_batches(
            config.epo_num_batches,
            manifest_path=paths.epo_manifest_path,
            corpus_path=paths.epo_corpus_path,
            chemistry_strict=config.epo_chemistry_strict,
        )
        return

    if config.evaluate_mteb_models:
        from datetime import datetime, timezone

        from src.multi_lingual_qac.mteb import (
            generate_mteb_comparison_tables,
            run_mteb_evaluation,
        )
        from src.multi_lingual_qac.mteb.runs import (
            append_index,
            dataset_sizes,
            git_info,
            make_run_id,
            update_latest_pointer,
            write_run_metadata,
        )

        now = datetime.now(timezone.utc)
        run_id = make_run_id(config.run_id_label, now=now)
        runs_root = project_root / "reports" / "runs"
        default_layout = not config.mteb_output_dir
        run_dir = Path(config.mteb_output_dir) if config.mteb_output_dir else (runs_root / run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Evaluation always saves everything later analysis needs: per-query
        # predictions + a tidy scored-rankings table (reporting is a separate step).
        prediction_dir = run_dir / "predictions"
        summaries = run_mteb_evaluation(
            list(config.evaluate_mteb_models),
            dataset_repo=config.mteb_dataset_repo,
            dataset_variant=config.mteb_dataset_variant,
            output_dir=run_dir,
            batch_size=config.mteb_batch_size,
            prediction_dir=prediction_dir,
            corpus_repo=config.mteb_corpus_repo,
        )

        # Comparison tables (cheap, no network/GPU) inside the run folder.
        generate_mteb_comparison_tables(results_dir=run_dir, output_dir=run_dir / "mteb_tables")

        # Persist the embedding model's rankings as a tidy table (for future metrics, e.g. CLIR@k).
        from src.alias_graph.retrieval_results import save_retrieval_results

        save_retrieval_results(
            prediction_dir,
            run_dir / "retrieval_results",
            dataset_repo=config.mteb_dataset_repo,
            dataset_variant=config.mteb_dataset_variant,
            model_names=list(config.evaluate_mteb_models),
        )

        # Run identity: metadata + rolling trend index + latest pointer.
        sizes = dataset_sizes(config.mteb_dataset_repo, "main", config.mteb_dataset_variant)
        commit, dirty = git_info(project_root)
        write_run_metadata(
            run_dir,
            run_id=run_id,
            created_at=now.isoformat(),
            dataset_repo=config.mteb_dataset_repo,
            dataset_variant=config.mteb_dataset_variant,
            dataset_revision="main",
            corpus_repo=config.mteb_corpus_repo,
            models=config.evaluate_mteb_models,
            batch_size=config.mteb_batch_size,
            summaries=summaries,
            sizes=sizes,
            git_commit=commit,
            git_dirty=dirty,
        )
        # Only the default reports/runs/<id> layout joins the global trend log + latest
        # pointer; an explicit --mteb-output-dir is treated as an unmanaged one-off.
        if default_layout:
            append_index(
                runs_root / "index.csv",
                run_id=run_id,
                created_at=now.isoformat(),
                dataset_repo=config.mteb_dataset_repo,
                dataset_variant=config.mteb_dataset_variant,
                git_commit=commit,
                sizes=sizes,
                summaries=summaries,
            )
            update_latest_pointer(runs_root, run_id)

        print("MTEB evaluation finished.")
        print(f"  Run id:  {run_id}")
        print(
            f"  Dataset: {config.mteb_dataset_repo} ({config.mteb_dataset_variant})"
            f"  queries={sizes.get('queries')} corpus={sizes.get('corpus')}"
        )
        print(f"  Output:  {run_dir}")
        for item in summaries:
            print(f"  {item.model_name}: {item.main_score:.4f}")
        if default_layout:
            print(f"  Trend index: {runs_root / 'index.csv'}")
        if config.analyze_questions or config.analyze_confusion:
            print("  Note: analysis is a separate step and was NOT run during evaluation.")
        print("  Next (analysis, re-runnable without re-evaluating):")
        print(
            f"    qac --analyze-questions --analyze-confusion --mteb-results-dir {run_dir}"
        )
        return

    if config.analyze_questions or config.analyze_confusion:
        results_dir = _resolve_results_dir(project_root, config.mteb_results_dir)
        prediction_dir = results_dir / "predictions"
        # Read the dataset the run was evaluated against (so it need not be re-specified).
        meta_repo, meta_variant = _dataset_from_run_metadata(results_dir)
        dataset_repo = meta_repo or config.mteb_dataset_repo
        dataset_variant = meta_variant or config.mteb_dataset_variant
        print(f"Analyzing run: {results_dir}  (dataset: {dataset_repo} / {dataset_variant})")
        if config.analyze_questions:
            from src.multi_lingual_qac.mteb import run_question_analysis

            analysis_dir = (
                Path(config.mteb_analysis_dir)
                if config.mteb_analysis_dir
                else (results_dir / "question_analysis")
            )
            report = run_question_analysis(
                prediction_dir,
                output_dir=analysis_dir,
                dataset_repo=dataset_repo,
                dataset_variant=dataset_variant,
                make_plots=not config.mteb_no_plots,
                query_metadata_csv=config.mteb_query_metadata,
            )
            print(f"  Question-level analysis: {report}")
        if config.analyze_confusion:
            from src.alias_graph.confusion_analysis import run_confusion_from_predictions

            run_confusion_from_predictions(
                prediction_dir,
                results_dir / "confusion",
                dataset_repo=dataset_repo,
                dataset_variant=dataset_variant,
                make_plots=not config.mteb_no_plots,
            )
        return

    if config.generate_mteb_tables:
        from src.multi_lingual_qac.export.hf_upload import upload_benchmark_outputs
        from src.multi_lingual_qac.mteb import generate_mteb_comparison_tables

        results_dir = _resolve_results_dir(project_root, config.mteb_results_dir)
        tables_dir = (
            Path(config.mteb_tables_dir)
            if config.mteb_tables_dir
            else (results_dir / "mteb_tables")
        )
        generated_dir = generate_mteb_comparison_tables(
            results_dir=results_dir,
            output_dir=tables_dir,
        )
        print("MTEB comparison tables generated.")
        print(f"  Source results: {results_dir}")
        print(f"  Output: {generated_dir}")

        if config.upload_mteb_results:
            repo_id = _normalize_hf_dataset_repo(
                config.mteb_upload_repo or config.mteb_dataset_repo
            )
            repo_tree_url = upload_benchmark_outputs(
                generated_dir,
                repo_id,
                path_in_repo="benchmark_outputs/mteb_tables",
            )
            print(f"  Hugging Face benchmark outputs: {repo_tree_url}")
        return

    paths = PipelinePaths.from_project_root(project_root)
    run_pipeline(config, paths)
