from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.multi_lingual_qac.dataloaders.google_patents import DEFAULT_LANGS


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path
    raw_ndjson: Path
    preprocessed_dir: Path
    corpus_csv: Path
    qac_dir: Path
    epo_data_dir: Path
    epo_manifest_path: Path
    epo_corpus_path: Path
    multilingual_corpus_csv: Path
    chebi_dir: Path
    alias_graph_dir: Path
    wiki_quality_dir: Path
    code_switched_dir: Path
    progressive_cs_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "PipelinePaths":
        data_dir = project_root / "data" / "google_patents"
        epo_dir = project_root / "data" / "EPO"
        return cls(
            project_root=project_root,
            raw_ndjson=data_dir / "chemistry_patents.ndjson",
            preprocessed_dir=data_dir / "preprocessed",
            corpus_csv=data_dir / "corpus.csv",
            qac_dir=data_dir / "qac",
            epo_data_dir=epo_dir,
            epo_manifest_path=epo_dir / "manifest.json",
            epo_corpus_path=epo_dir / "multilingual_corpus.csv",
            multilingual_corpus_csv=data_dir / "multilingual_corpus.csv",
            chebi_dir=project_root / "data" / "chebi",
            alias_graph_dir=project_root / "data" / "alias_graph",
            wiki_quality_dir=project_root / "reports" / "wiki_name_quality",
            code_switched_dir=project_root / "data" / "code_switched",
            progressive_cs_dir=project_root / "data" / "progressive_cs",
        )


@dataclass(frozen=True)
class PipelineConfig:
    yes: bool = False
    no_extraction: bool = False
    limit: Optional[int] = None
    qa_sample: Optional[int] = None
    qa_batch: Optional[bool] = None
    push_hf: bool = False
    hf_repo: Optional[str] = None
    evaluate_mteb_models: tuple[str, ...] = ()
    mteb_dataset_repo: str = ""
    mteb_corpus_repo: str = "owner/multilingual-corpus"
    mteb_dataset_variant: str = "multilingual"
    mteb_output_dir: Optional[str] = None
    mteb_batch_size: int = 32
    mteb_save_predictions: bool = False
    analyze_questions: bool = False
    mteb_analysis_dir: Optional[str] = None
    mteb_no_plots: bool = False
    mteb_query_metadata: Optional[str] = None
    run_id_label: Optional[str] = None
    generate_mteb_tables: bool = False
    mteb_results_dir: Optional[str] = None
    mteb_tables_dir: Optional[str] = None
    upload_mteb_results: bool = False
    mteb_upload_repo: Optional[str] = None
    languages: tuple[str, ...] = tuple(DEFAULT_LANGS)
    epo_ingest: bool = False
    epo_num_batches: int = 1
    epo_chemistry_strict: bool = False
    build_alias_graph: bool = False
    alias_corpus: Optional[str] = None
    alias_output_dir: Optional[str] = None
    chebi_variant: str = "full"
    alias_langs: tuple[str, ...] = ("zh", "en", "de", "fr", "es")
    alias_use_wikipedia: bool = True
    alias_min_gold: int = 2
    alias_min_neg: int = 3
    alias_max_concepts: Optional[int] = None
    alias_max_df: float = 0.02
    alias_molecular_only: bool = True
    alias_leaf_only: bool = True
    check_wiki_names: bool = False
    export_concept: Optional[str] = None
    alias_generate_qa: bool = False
    alias_qa_strategy: int = 1
    alias_qa_model: str = "gpt-5-mini"
    alias_qa_seed: int = 42
    alias_qa_limit: Optional[int] = None
    alias_qa_workers: int = 1
    build_code_switched: bool = False
    cs_variants: str = "A,B,C,D,F"
    cs_limit: Optional[int] = None
    cs_model: str = "gpt-5-mini"
    cs_seed: int = 42
    cs_output_dir: Optional[str] = None
    cs_generate_qa: bool = False
    cs_qa_model: str = "gpt-5-mini"
    cs_qa_seed: int = 42
    cs_qa_limit: Optional[int] = None
    cs_qa_workers: int = 1
    # Data-creation command (build corpus + queries + push to HF):
    create_progressive_data: bool = False
    # Evaluation command (run embedding models against the published dataset):
    eval_progressive_cs: bool = False
    pcs_hf_repo: str = "owner/progressive-code-switch"
    pcs_steps: int = 5
    pcs_modes: str = "B,C,D,F"
    pcs_seed: int = 42
    pcs_limit: Optional[int] = None
    pcs_output_dir: Optional[str] = None
    pcs_qa_model: str = "gpt-5-mini"
    pcs_grader_model: str = "anthropic/claude-sonnet-4.5"
    pcs_qa_strategy: int = 4  # query-language strategy; 4 = "all" (one query per language)
    pcs_qa_seed: int = 42
    pcs_qa_limit: Optional[int] = None
    pcs_qa_workers: int = 1
    pcs_eval_models: tuple[str, ...] = ()
    pcs_eval_limit: Optional[int] = None
    pcs_eval_batch_size: int = 32
    # Granular sub-steps (optional; --create-progressive-data runs all three):
    build_progressive_cs: bool = False
    progressive_cs_qa: bool = False
    push_progressive_cs: bool = False
    push_alias_graph_hf: bool = False
    alias_hf_repo: str = "owner/multi-lingual-qac-alias-graph"
    hf_dry_run: bool = False
    hf_private: bool = False
    analyze_confusion: bool = False
    push_corpus_hf: bool = False
    corpus_hf_repo: str = "owner/multilingual-corpus"
