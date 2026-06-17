from __future__ import annotations

import csv
import json
import math
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import get_dataset_config_names, load_dataset
import mteb
from mteb import MTEB
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.results import TaskResult
import pytrec_eval
from sentence_transformers import SentenceTransformer

DEFAULT_MTEB_DATASET_REPO = "anonymous/multi-lingual-qac-chem-patents"
DEFAULT_MTEB_VARIANT = "multilingual"
DEFAULT_MTEB_OUTPUT_DIR = "reports/mteb"
DEFAULT_MTEB_TABLES_DIR = "reports/mteb_tables"
DEFAULT_MTEB_CACHE_DIR = ".cache/huggingface"
DEFAULT_MTEB_MAIN_SCORE = "recall_at_10"
# Instruction handed to instruction-based models (Qwen3-Embedding, e5-instruct) for our
# CUSTOM Hub task. Set on TaskMetadata.prompt so MTEB's get_instruction() returns it
# directly instead of falling back to get_task(name) -- which KeyErrors because our task
# isn't in MTEB's static registry. Value mirrors AbsTaskRetrieval.abstask_prompt, so the
# behavior matches a built-in retrieval task. Instruct wrappers apply it to queries only.
DEFAULT_MTEB_RETRIEVAL_PROMPT = "Retrieve text based on user query."
RETRIEVAL_CUTOFFS = (10, 20, 50, 100)
SAME_LANGUAGE_DIAGNOSTIC_LANGS = ("de", "en", "es", "fr", "pt", "zh")
DEFAULT_MTEB_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
]

# Curated multilingual model set for the alias-graph benchmark (and the default for
# `--evaluate-mteb`). Loaded via `mteb.get_model` so instruction/query prompts,
# trust_remote_code and ColBERT late-interaction are handled per the MTEB registry;
# SapBERT (not registered) falls back to a plain SentenceTransformer.
ALIAS_GRAPH_MODELS = [
    "Qwen/Qwen3-Embedding-0.6B",
    "intfloat/multilingual-e5-large-instruct",
    "BAAI/bge-m3",
    "jinaai/jina-embeddings-v3",
    "Alibaba-NLP/gte-multilingual-base",                 # mGTE
    "google/embeddinggemma-300m",                        # gated on HF
    "ibm-granite/granite-embedding-278m-multilingual",
    "jinaai/jina-colbert-v2",                            # ColBERT (needs pylate)
    "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",  # ST fallback (not in MTEB registry)
    "sentence-transformers/LaBSE",
    "nomic-ai/nomic-embed-text-v2-moe",                  # MoE; mteb registry loader applies search_query/document prompts
]

# Some custom-code models register a NON-persistent `position_ids = arange(max_pos)` buffer in
# __init__; transformers 5.x can leave that buffer UNINITIALISED (garbage memory) after loading,
# so the model then indexes its RoPE table with garbage position ids and crashes on the first
# encode -- as an out-of-bounds IndexError on CPU, or an opaque "CUDA error: device-side assert
# triggered" on GPU. gte-multilingual-base hits exactly this. Fix: re-fill the buffer post-load.
MODELS_NEEDING_POSITION_IDS_REPAIR = {
    "Alibaba-NLP/gte-multilingual-base",
}


def _repair_position_ids_buffers(model: Any) -> None:
    """Re-initialise any 1-D integer `position_ids` buffer that the loader left as garbage to a
    correct ``arange``. Called ONLY for MODELS_NEEDING_POSITION_IDS_REPAIR, so it cannot affect
    models whose position_ids legitimately differ. Handles both the MTEB wrapper (``.model``) and
    a plain SentenceTransformer (an nn.Module itself)."""
    import torch

    module = model if isinstance(model, torch.nn.Module) else getattr(model, "model", None)
    if not isinstance(module, torch.nn.Module):
        return
    fixed = 0
    for sub in module.modules():
        buf = sub._buffers.get("position_ids")
        if isinstance(buf, torch.Tensor) and buf.dim() == 1 and not torch.is_floating_point(buf):
            sub.register_buffer(
                "position_ids",
                torch.arange(buf.numel(), dtype=buf.dtype, device=buf.device),
                persistent=False,
            )
            fixed += 1
    if fixed:
        print(f"[fixup] re-initialised {fixed} position_ids buffer(s) for `{model}` "
              "(transformers non-persistent-buffer bug)")

LANGUAGE_TO_MTEB = {
    "ar": "arb-Arab",
    "bg": "bul-Cyrl",
    "cs": "ces-Latn",
    "da": "dan-Latn",
    "de": "deu-Latn",
    "el": "ell-Grek",
    "en": "eng-Latn",
    "es": "spa-Latn",
    "et": "est-Latn",
    "fa": "pes-Arab",
    "fi": "fin-Latn",
    "fr": "fra-Latn",
    "hi": "hin-Deva",
    "hu": "hun-Latn",
    "it": "ita-Latn",
    "ja": "jpn-Jpan",
    "ko": "kor-Hang",
    "lt": "lit-Latn",
    "lv": "lav-Latn",
    "mt": "mlt-Latn",
    "nl": "nld-Latn",
    "pl": "pol-Latn",
    "pt": "por-Latn",
    "ro": "ron-Latn",
    "ru": "rus-Cyrl",
    "sk": "slk-Latn",
    "sl": "slv-Latn",
    "sv": "swe-Latn",
    "tr": "tur-Latn",
    "zh": "zho-Hans",
}

TABLE_METRICS = [
    "main_score",
    "recall_at_10",
    "recall_at_100",
    "map_at_10",
    "map_at_100",
    "map",
    "ndcg_at_10",
    "ndcg_at_100",
    "same_language_irrelevant_share_at_100",
]


@dataclass(frozen=True)
class ModelEvaluationSummary:
    model_name: str
    model_slug: str
    dataset_variant: str
    main_score: float
    metrics: dict[str, float]
    output_dir: str
    eval_languages: list[str]
    evaluation_time_seconds: float | None


class HubDatasetRetrievalTask(AbsTaskRetrieval):
    def __init__(
        self,
        metadata: TaskMetadata,
        *,
        dataset_repo: str,
        revision: str,
        dataset_variant: str,
        corpus_repo: str = "",
    ):
        self.metadata = metadata
        self.dataset_repo = dataset_repo
        # Shared retrieval corpus (haystack). When set and different from dataset_repo,
        # the corpus loaded from this repo replaces the dataset's own corpus config.
        self.corpus_repo = corpus_repo if corpus_repo and corpus_repo != dataset_repo else ""
        self.revision = revision
        self.dataset_variant = dataset_variant
        self._query_language_by_id: dict[str, str] | None = None
        self._corpus_language_by_id: dict[str, str] | None = None
        super().__init__()

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """Load retrieval data, working around a strict-offline-mode quirk.

        In offline mode (``HF_HUB_OFFLINE=1``), ``datasets.get_dataset_config_names``
        returns ``['default']`` instead of the dataset's real config list. That makes
        MTEB's ``RetrievalDatasetLoader`` believe ``default`` is a valid config and skip
        its ``default`` -> ``qrels`` fallback, so it tries to load a nonexistent
        ``default`` config and fails on compute nodes that have no internet. We
        temporarily substitute the real config names (from the hub when reachable, else
        from the local datasets cache) while the upstream loader runs, then restore the
        original function. Online behaviour is unchanged.
        """
        if self.data_loaded:
            return
        import mteb.abstasks.retrieval_dataset_loaders as _rdl

        real_configs = _resolve_loader_configs(self.dataset_repo, self.revision)
        original_get_config_names = _rdl.get_dataset_config_names
        if real_configs:
            def _patched_get_config_names(path, revision=None, *args, **kwargs):
                return list(real_configs)

            _rdl.get_dataset_config_names = _patched_get_config_names
        try:
            super().load_data(num_proc=num_proc, **kwargs)
        finally:
            _rdl.get_dataset_config_names = original_get_config_names
        if self.corpus_repo:
            self._swap_in_shared_corpus()

    def _swap_in_shared_corpus(self) -> None:
        """Replace each split's corpus with the shared corpus loaded from ``corpus_repo``.

        Any judged (qrels) document not present in the shared corpus is unioned in from
        the dataset's own corpus, so every gold / look-alike doc stays retrievable even
        if a benchmark's qrels reference ids outside the shared corpus.
        """
        from datasets import concatenate_datasets

        shared = _load_corpus_dataset(self.corpus_repo, self.revision)
        keep = [c for c in ("id", "title", "text") if c in shared.column_names]
        shared = shared.select_columns(keep)
        shared_ids = set(shared["id"])
        for subset, splits in self.dataset.items():
            for split, data in splits.items():
                judged: set[str] = set()
                for docs in data["relevant_docs"].values():
                    judged.update(str(d) for d in docs)
                missing = judged - shared_ids
                corpus = shared
                if missing:
                    own = data["corpus"]
                    extra = own.filter(lambda r: str(r["id"]) in missing).select_columns(
                        [c for c in keep if c in own.column_names]
                    )
                    corpus = concatenate_datasets([shared, extra])
                    print(f"[corpus] {subset}/{split}: unioned {len(extra)} judged doc(s) "
                          f"missing from {self.corpus_repo}")
                data["corpus"] = corpus
        print(f"[corpus] retrieval haystack = {self.corpus_repo} ({len(shared_ids)} docs)")

    def _get_query_language_by_id(self) -> dict[str, str]:
        if self._query_language_by_id is not None:
            return self._query_language_by_id
        query_config = _dataset_config_name(
            self.dataset_repo,
            self.revision,
            self.dataset_variant,
            "queries",
        )
        queries = load_dataset(
            self.dataset_repo,
            query_config,
            split="train",
            revision=self.revision,
        )
        lang_column = _query_language_column(list(queries.column_names))
        mapping: dict[str, str] = {}
        if lang_column is None:
            self._query_language_by_id = mapping
            return mapping
        id_column = "_id" if "_id" in queries.column_names else "query_id"
        for row in queries:
            query_id = str(row.get(id_column, "")).strip()
            if not query_id:
                continue
            language = str(row.get(lang_column, "")).strip().lower()
            if not language:
                language = _infer_language(query_id)
            mapping[query_id] = language
        self._query_language_by_id = mapping
        return mapping

    def _get_corpus_language_by_id(self) -> dict[str, str]:
        if self._corpus_language_by_id is not None:
            return self._corpus_language_by_id
        corpus_config = _dataset_config_name(
            self.dataset_repo,
            self.revision,
            self.dataset_variant,
            "corpus",
        )
        corpus = load_dataset(
            self.dataset_repo,
            corpus_config,
            split="train",
            revision=self.revision,
        )
        mapping: dict[str, str] = {}
        id_column = "_id" if "_id" in corpus.column_names else "corpus_id"
        lang_column = (
            "corpus_language"
            if "corpus_language" in corpus.column_names
            else "language"
            if "language" in corpus.column_names
            else None
        )
        for row in corpus:
            corpus_id = str(row.get(id_column, "")).strip()
            if not corpus_id:
                continue
            language = str(row.get(lang_column, "")).strip().lower() if lang_column else ""
            if not language:
                language = _infer_language(corpus_id)
            mapping[corpus_id] = language
        self._corpus_language_by_id = mapping
        return mapping

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int | float]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        del scores, hf_split, hf_subset
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map"})
        per_query_scores = evaluator.evaluate(results)
        query_language_by_id = self._get_query_language_by_id()
        corpus_language_by_id = self._get_corpus_language_by_id()

        metric_scores: dict[str, float] = {}
        if per_query_scores:
            full_map = sum(float(item.get("map", 0.0)) for item in per_query_scores.values()) / len(
                per_query_scores
            )
            metric_scores["map"] = round(full_map, 5)

        recall_scores: dict[int, list[float]] = {cutoff: [] for cutoff in RETRIEVAL_CUTOFFS}
        map_scores: dict[int, list[float]] = {cutoff: [] for cutoff in RETRIEVAL_CUTOFFS}
        ndcg_scores: dict[int, list[float]] = {cutoff: [] for cutoff in RETRIEVAL_CUTOFFS}
        same_language_irrelevant_shares: dict[int, list[float]] = {
            cutoff: [] for cutoff in RETRIEVAL_CUTOFFS
        }
        same_language_irrelevant_shares_at_100_by_query_lang: dict[str, list[float]] = {
            lang: [] for lang in SAME_LANGUAGE_DIAGNOSTIC_LANGS
        }
        for query_id, doc_scores in results.items():
            query_language = query_language_by_id.get(query_id, _infer_language(query_id))
            if not query_language:
                continue
            ranked_doc_ids = [
                doc_id
                for doc_id, _score in sorted(
                    doc_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
            relevant_doc_ids = {
                doc_id
                for doc_id, relevance in qrels.get(query_id, {}).items()
                if float(relevance) > 0.0
            }
            if not relevant_doc_ids:
                continue

            for cutoff in RETRIEVAL_CUTOFFS:
                top_doc_ids = ranked_doc_ids[:cutoff]
                if not top_doc_ids:
                    continue

                relevant_seen = 0
                precision_sum = 0.0
                dcg = 0.0
                for rank_idx, doc_id in enumerate(top_doc_ids, start=1):
                    if doc_id in relevant_doc_ids:
                        relevant_seen += 1
                        precision_sum += relevant_seen / rank_idx
                        dcg += 1.0 / math.log2(rank_idx + 1)
                recall_scores[cutoff].append(relevant_seen / len(relevant_doc_ids))
                map_scores[cutoff].append(
                    precision_sum / min(len(relevant_doc_ids), cutoff)
                    if relevant_doc_ids
                    else 0.0
                )
                ideal_relevant = min(len(relevant_doc_ids), cutoff)
                ideal_dcg = sum(
                    1.0 / math.log2(rank_idx + 1)
                    for rank_idx in range(1, ideal_relevant + 1)
                )
                ndcg_scores[cutoff].append(dcg / ideal_dcg if ideal_dcg else 0.0)

                unrelated_doc_ids = [
                    doc_id for doc_id in top_doc_ids if doc_id not in relevant_doc_ids
                ]
                if not unrelated_doc_ids:
                    same_language_irrelevant_share = 0.0
                else:
                    same_language_unrelated = sum(
                        1
                        for doc_id in unrelated_doc_ids
                        if corpus_language_by_id.get(doc_id, _infer_language(doc_id))
                        == query_language
                    )
                    same_language_irrelevant_share = same_language_unrelated / len(
                        unrelated_doc_ids
                    )
                same_language_irrelevant_shares[cutoff].append(
                    same_language_irrelevant_share
                )
                if cutoff == 100 and query_language in SAME_LANGUAGE_DIAGNOSTIC_LANGS:
                    same_language_irrelevant_shares_at_100_by_query_lang[
                        query_language
                    ].append(same_language_irrelevant_share)

        for cutoff in RETRIEVAL_CUTOFFS:
            if recall_scores[cutoff]:
                metric_scores[f"recall_at_{cutoff}"] = round(
                    sum(recall_scores[cutoff]) / len(recall_scores[cutoff]),
                    5,
                )
            if map_scores[cutoff]:
                metric_scores[f"map_at_{cutoff}"] = round(
                    sum(map_scores[cutoff]) / len(map_scores[cutoff]),
                    5,
                )
            if ndcg_scores[cutoff]:
                metric_scores[f"ndcg_at_{cutoff}"] = round(
                    sum(ndcg_scores[cutoff]) / len(ndcg_scores[cutoff]),
                    5,
                )
            if same_language_irrelevant_shares[cutoff]:
                metric_scores[f"same_language_irrelevant_share_at_{cutoff}"] = round(
                    sum(same_language_irrelevant_shares[cutoff])
                    / len(same_language_irrelevant_shares[cutoff]),
                    5,
                )

        for query_language in SAME_LANGUAGE_DIAGNOSTIC_LANGS:
            values = same_language_irrelevant_shares_at_100_by_query_lang[query_language]
            if values:
                metric_scores[
                    f"same_language_irrelevant_share_at_100_lang_{query_language}"
                ] = round(sum(values) / len(values), 5)
        return metric_scores


COMPARISON_METRICS = [
    "main_score",
    "recall_at_10",
    "recall_at_20",
    "recall_at_50",
    "recall_at_100",
    "map_at_10",
    "map_at_20",
    "map_at_50",
    "map_at_100",
    "map",
    "ndcg_at_10",
    "ndcg_at_20",
    "ndcg_at_50",
    "ndcg_at_100",
    "mrr_at_10",
    "hit_rate_at_10",
    "hit_rate_at_100",
    "same_language_irrelevant_share_at_10",
    "same_language_irrelevant_share_at_20",
    "same_language_irrelevant_share_at_50",
    "same_language_irrelevant_share_at_100",
    "same_language_irrelevant_share_at_100_lang_de",
    "same_language_irrelevant_share_at_100_lang_en",
    "same_language_irrelevant_share_at_100_lang_es",
    "same_language_irrelevant_share_at_100_lang_fr",
    "same_language_irrelevant_share_at_100_lang_pt",
    "same_language_irrelevant_share_at_100_lang_zh",
]


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "default"


def _normalize_dataset_variant(value: str | None) -> str:
    normalized = str(value or DEFAULT_MTEB_VARIANT).strip().lower().replace("-", "_")
    if normalized not in {"multilingual", "cross_language"}:
        raise ValueError(f"Unsupported dataset variant: {value}")
    return normalized


def _infer_language(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    parts = [part for part in raw.replace("-", "_").split("_") if part]
    if not parts:
        return ""
    candidate = parts[-1]
    if 2 <= len(candidate) <= 5 and candidate.isalpha():
        return candidate
    return ""


def _query_language_column(column_names: list[str]) -> str | None:
    if "query_language" in column_names:
        return "query_language"
    if "question_language" in column_names:
        return "question_language"
    if "language" in column_names:
        return "language"
    return None


def _dataset_config_name(
    dataset_repo: str,
    revision: str,
    dataset_variant: str,
    base_config: str,
) -> str:
    subset = _resolve_dataset_subset(dataset_repo, revision, dataset_variant)
    return f"{subset}-{base_config}" if subset is not None else base_config


def _load_corpus_dataset(corpus_repo: str, revision: str):
    """Load a shared `corpus` config from a HF repo or a local dir; expose an `id` column.

    Local dirs use the dry-run export layout (``<dir>/corpus/corpus.parquet`` or
    ``<dir>/corpus.parquet``), so the shared corpus can be tested offline.
    """
    path = Path(corpus_repo)
    if path.is_dir():
        pq = path / "corpus" / "corpus.parquet"
        if not pq.exists():
            pq = path / "corpus.parquet"
        corpus = load_dataset("parquet", data_files=str(pq), split="train")
    else:
        config = _dataset_config_name(corpus_repo, revision, "multilingual", "corpus")
        corpus = load_dataset(corpus_repo, config, split="train", revision=revision)
    if "id" not in corpus.column_names and "_id" in corpus.column_names:
        corpus = corpus.rename_column("_id", "id")
    return corpus


def _resolve_dataset_subset(dataset_repo: str, revision: str, dataset_variant: str) -> str | None:
    variant = _normalize_dataset_variant(dataset_variant)
    dataset_configs = set(get_dataset_config_names(dataset_repo, revision=revision))
    variant_qrels = f"{variant}-qrels"
    if variant_qrels in dataset_configs:
        return variant
    if variant == "multilingual":
        return None
    raise ValueError(
        f"Dataset `{dataset_repo}` does not expose the `{variant}` retrieval variant."
    )


def _dataset_cache_config_names(dataset_repo: str) -> list[str]:
    """Config names materialized in the local datasets cache (offline-safe)."""
    from datasets import config as datasets_config

    cache_root = Path(datasets_config.HF_DATASETS_CACHE) / dataset_repo.replace("/", "___")
    if not cache_root.is_dir():
        return []
    return sorted(path.name for path in cache_root.iterdir() if path.is_dir())


def _resolve_loader_configs(dataset_repo: str, revision: str) -> list[str]:
    """Real dataset config names, robust to strict offline mode.

    ``get_dataset_config_names`` returns ``['default']`` when the hub is unreachable,
    so trust its answer only when it looks real; otherwise fall back to the config
    directories present in the local datasets cache.
    """
    try:
        configs = [
            name
            for name in get_dataset_config_names(dataset_repo, revision=revision)
            if name != "default"
        ]
    except Exception:
        configs = []
    if configs:
        return configs
    return _dataset_cache_config_names(dataset_repo)


def _dataset_task_name(dataset_repo: str, dataset_variant: str) -> str:
    owner, _, name = dataset_repo.partition("/")
    owner_slug = _slugify(owner or "hf")
    name_slug = _slugify(name or dataset_repo)
    variant_slug = _slugify(dataset_variant)
    return f"{owner_slug}_{name_slug}_{variant_slug}_retrieval"


def _default_model_cache_dir() -> Path:
    return Path(__file__).resolve().parents[3] / DEFAULT_MTEB_CACHE_DIR


def _configure_local_model_cache() -> Path:
    cache_dir = _default_model_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    sentence_transformers_dir = cache_dir / "sentence_transformers"
    hub_dir = cache_dir / "hub"
    transformers_dir = cache_dir / "transformers"
    sentence_transformers_dir.mkdir(parents=True, exist_ok=True)
    hub_dir.mkdir(parents=True, exist_ok=True)
    transformers_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_dir)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(sentence_transformers_dir)
    return sentence_transformers_dir


def _detect_query_languages(
    dataset_repo: str,
    revision: str,
    dataset_variant: str,
) -> list[str]:
    query_config = _dataset_config_name(dataset_repo, revision, dataset_variant, "queries")
    queries = load_dataset(dataset_repo, query_config, split="train", revision=revision)
    lang_column = _query_language_column(list(queries.column_names))
    if lang_column is None:
        return [LANGUAGE_TO_MTEB["en"]]

    langs = sorted(
        {
            str(value).strip().lower()
            for value in queries[lang_column]
            if str(value).strip()
        }
    )
    mapped = [LANGUAGE_TO_MTEB[lang] for lang in langs if lang in LANGUAGE_TO_MTEB]
    return mapped or [LANGUAGE_TO_MTEB["en"]]


def build_mteb_task(
    dataset_repo: str = DEFAULT_MTEB_DATASET_REPO,
    *,
    revision: str = "main",
    dataset_variant: str = DEFAULT_MTEB_VARIANT,
    corpus_repo: str = "",
) -> HubDatasetRetrievalTask:
    dataset_variant = _normalize_dataset_variant(dataset_variant)
    subset = _resolve_dataset_subset(dataset_repo, revision, dataset_variant)
    eval_langs = _detect_query_languages(dataset_repo, revision, dataset_variant)
    eval_langs_config = {subset or "default": eval_langs}
    metadata = TaskMetadata(
        name=_dataset_task_name(dataset_repo, dataset_variant),
        dataset={"path": dataset_repo, "revision": revision},
        description=(
            f"Custom {dataset_variant.replace('_', '-')} retrieval evaluation over the "
            f"chemistry-patent QAC dataset `{dataset_repo}`. Cross-lingual relevance "
            "treats every corpus document sharing a question's publication_number as a positive."
        ),
        reference=f"https://huggingface.co/datasets/{dataset_repo}",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=eval_langs_config,
        main_score=DEFAULT_MTEB_MAIN_SCORE,
        prompt=DEFAULT_MTEB_RETRIEVAL_PROMPT,  # short-circuits get_instruction() -> no registry KeyError
        domains=["Chemistry", "Engineering"],
        task_subtypes=["Question Answering Retrieval"],
        license="not specified",
        annotations_creators="LM-generated and reviewed",
        sample_creation="LM-generated and verified",
        is_public=True,
        contributed_by="multi-lingual-qac",
    )
    return HubDatasetRetrievalTask(
        metadata,
        dataset_repo=dataset_repo,
        revision=revision,
        dataset_variant=dataset_variant,
        corpus_repo=corpus_repo,
    )


def _extract_numeric_metrics(result: TaskResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split_rows in result.scores.values():
        for row in split_rows:
            for key, value in row.items():
                if key in {"hf_subset", "languages"}:
                    continue
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
            if metrics:
                return metrics
    return metrics


def _write_summary_reports(
    output_dir: Path,
    dataset_repo: str,
    dataset_variant: str,
    summaries: list[ModelEvaluationSummary],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_md = output_dir / "summary.md"

    payload = {
        "dataset_repo": dataset_repo,
        "dataset_variant": dataset_variant,
        "models": [
            {
                "model_name": item.model_name,
                "model_slug": item.model_slug,
                "dataset_variant": item.dataset_variant,
                "main_score": item.main_score,
                "metrics": item.metrics,
                "output_dir": item.output_dir,
                "eval_languages": item.eval_languages,
                "evaluation_time_seconds": item.evaluation_time_seconds,
            }
            for item in summaries
        ],
    }
    summary_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metric_keys = sorted({key for item in summaries for key in item.metrics})
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model_name",
                "dataset_variant",
                "main_score",
                "evaluation_time_seconds",
                "eval_languages",
                "output_dir",
                *metric_keys,
            ],
        )
        writer.writeheader()
        for item in summaries:
            row: dict[str, Any] = {
                "model_name": item.model_name,
                "dataset_variant": item.dataset_variant,
                "main_score": item.main_score,
                "evaluation_time_seconds": item.evaluation_time_seconds,
                "eval_languages": ", ".join(item.eval_languages),
                "output_dir": item.output_dir,
            }
            row.update(item.metrics)
            writer.writerow(row)

    lines = [
        "# MTEB Evaluation Summary",
        "",
        f"- Dataset: `{dataset_repo}`",
        f"- Variant: `{dataset_variant}`",
        f"- Main score: `{DEFAULT_MTEB_MAIN_SCORE}`",
        "",
        "| Model | Main score | Eval time (s) |",
        "| --- | ---: | ---: |",
    ]
    for item in summaries:
        eval_time = (
            f"{item.evaluation_time_seconds:.1f}"
            if item.evaluation_time_seconds is not None
            else ""
        )
        lines.append(f"| `{item.model_name}` | {item.main_score:.4f} | {eval_time} |")
        if item.metrics:
            metric_parts = [
                f"`{key}`={value:.4f}" for key, value in sorted(item.metrics.items())
            ]
            lines.append(f"| `{item.model_name}` metrics | {'; '.join(metric_parts)} | |")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_summary_models(results_dir: Path) -> tuple[str, str, list[ModelEvaluationSummary]]:
    summary_json = results_dir / "summary.json"
    if not summary_json.exists():
        return _load_raw_result_models(results_dir)

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    dataset_repo = str(payload.get("dataset_repo", DEFAULT_MTEB_DATASET_REPO))
    dataset_variant = _normalize_dataset_variant(payload.get("dataset_variant", DEFAULT_MTEB_VARIANT))
    models_payload = payload.get("models", [])
    summaries: list[ModelEvaluationSummary] = []
    for item in models_payload:
        summaries.append(
            ModelEvaluationSummary(
                model_name=str(item.get("model_name", "")).strip(),
                model_slug=str(item.get("model_slug", "")).strip() or _slugify(str(item.get("model_name", ""))),
                dataset_variant=_normalize_dataset_variant(
                    item.get("dataset_variant", dataset_variant)
                ),
                main_score=float(item.get("main_score", 0.0)),
                metrics={
                    str(key): float(value)
                    for key, value in dict(item.get("metrics", {})).items()
                    if isinstance(value, (int, float))
                },
                output_dir=str(item.get("output_dir", "")).strip(),
                eval_languages=[str(x) for x in item.get("eval_languages", [])],
                evaluation_time_seconds=(
                    float(item["evaluation_time_seconds"])
                    if item.get("evaluation_time_seconds") is not None
                    else None
                ),
            )
        )
    return dataset_repo, dataset_variant, summaries


def _load_raw_result_models(results_dir: Path) -> tuple[str, str, list[ModelEvaluationSummary]]:
    result_files = sorted(
        path
        for path in results_dir.rglob("*.json")
        if path.name not in {"summary.json", "model_meta.json", "model_comparison.json"}
    )
    summaries: list[ModelEvaluationSummary] = []
    dataset_repo = DEFAULT_MTEB_DATASET_REPO
    dataset_variant = DEFAULT_MTEB_VARIANT

    for result_file in result_files:
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        if "scores" not in payload:
            continue
        task_name = str(payload.get("task_name", "")).strip().lower()
        if task_name.endswith("_cross-language_retrieval") or task_name.endswith("_cross_language_retrieval"):
            dataset_variant = "cross_language"
        elif task_name.endswith("_multilingual_retrieval"):
            dataset_variant = "multilingual"

        train_rows = payload.get("scores", {}).get("train", [])
        if not train_rows:
            continue
        first_row = train_rows[0]
        metrics = {
            str(key): float(value)
            for key, value in first_row.items()
            if key not in {"hf_subset", "languages"} and isinstance(value, (int, float))
        }
        main_score = float(
            first_row.get("main_score", metrics.get(DEFAULT_MTEB_MAIN_SCORE, 0.0))
        )
        eval_languages = [str(x) for x in first_row.get("languages", [])]

        model_meta_path = result_file.with_name("model_meta.json")
        model_name = result_file.parent.parent.name.replace("__", "/", 1)
        model_slug = _slugify(model_name)
        if model_meta_path.exists():
            model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
            model_name = str(model_meta.get("name", model_name))
            model_slug = _slugify(model_name)

        summaries.append(
            ModelEvaluationSummary(
                model_name=model_name,
                model_slug=model_slug,
                dataset_variant=dataset_variant,
                main_score=main_score,
                metrics=metrics,
                output_dir=str(result_file.parent),
                eval_languages=eval_languages,
                evaluation_time_seconds=(
                    float(payload["evaluation_time"])
                    if payload.get("evaluation_time") is not None
                    else None
                ),
            )
        )

    if not summaries:
        raise ValueError(
            f"Could not find `summary.json` or any raw MTEB result json files under `{results_dir}`."
        )
    return dataset_repo, dataset_variant, summaries


def _metric_value(item: ModelEvaluationSummary, metric: str) -> float | None:
    if metric == "main_score":
        return item.main_score
    return item.metrics.get(metric)


def _best_metric_values(
    summaries: list[ModelEvaluationSummary],
    metrics: list[str],
) -> dict[str, float]:
    best: dict[str, float] = {}
    for metric in metrics:
        values = [_metric_value(item, metric) for item in summaries]
        numeric_values = [value for value in values if value is not None]
        if numeric_values:
            best[metric] = (
                min(numeric_values)
                if metric.startswith("same_language_irrelevant_share_at_")
                else max(numeric_values)
            )
    return best


def _format_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _metric_label(metric: str) -> str:
    labels = {
        "main_score": "Main score",
        "map": "MAP",
        "same_language_irrelevant_share_at_100": "Same-lang irr@100",
    }
    if metric in labels:
        return labels[metric]
    match = re.fullmatch(r"([a-z]+)_at_(\d+)", metric)
    if match:
        name, cutoff = match.groups()
        display = {
            "recall": "Recall",
            "map": "MAP",
            "ndcg": "nDCG",
            "mrr": "MRR",
            "hit_rate": "Hit",
        }.get(name, name)
        return f"{display}@{cutoff}"
    return metric


def _format_metric_cell(
    item: ModelEvaluationSummary,
    metric: str,
    best: dict[str, float],
) -> str:
    value = _metric_value(item, metric)
    formatted = _format_metric(value)
    if value is not None and metric in best and value == best[metric]:
        return f"**{formatted}**"
    return formatted


def _latex_metric_cell(
    item: ModelEvaluationSummary,
    metric: str,
    best: dict[str, float],
) -> str:
    value = _metric_value(item, metric)
    formatted = _format_metric(value)
    if value is not None and metric in best and value == best[metric]:
        return rf"\textbf{{{formatted}}}"
    return formatted


def _ordered_metric_keys(summaries: list[ModelEvaluationSummary]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for metric in COMPARISON_METRICS:
        if metric not in seen:
            ordered.append(metric)
            seen.add(metric)
    extras = sorted(
        {
            key
            for item in summaries
            for key in item.metrics
            if key not in seen
        }
    )
    ordered.extend(extras)
    return ordered


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _build_markdown_comparison(
    dataset_repo: str,
    ranked: list[ModelEvaluationSummary],
) -> str:
    best = _best_metric_values(ranked, TABLE_METRICS)
    top = ranked[0]
    metric_headers = [_metric_label(metric) for metric in TABLE_METRICS]
    lines = [
        "# MTEB Model Comparison",
        "",
        "## Leaderboard",
        "",
        "### Overview",
        "",
        f"- Dataset: `{dataset_repo}`",
        f"- Models compared: `{len(ranked)}`",
        f"- Best model by `{DEFAULT_MTEB_MAIN_SCORE}`: `{top.model_name}` ({top.main_score:.4f})",
        "",
        "### Ranking",
        "",
        "| Rank | Model | " + " | ".join(metric_headers) + " | Time (s) |",
        "| ---: | --- | " + " | ".join(["---:"] * len(metric_headers)) + " | ---: |",
    ]
    for idx, item in enumerate(ranked, start=1):
        cells = [
            str(idx),
            f"`{item.model_name}`",
            *[_format_metric_cell(item, metric, best) for metric in TABLE_METRICS],
            (
                f"{item.evaluation_time_seconds:.1f}"
                if item.evaluation_time_seconds is not None
                else ""
            ),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "### Metric Winners",
            "",
            "| Metric | Best model | Score |",
            "| --- | --- | ---: |",
        ]
    )
    for metric in TABLE_METRICS:
        winner = (
            min(
                ranked,
                key=lambda item: _metric_value(item, metric)
                if _metric_value(item, metric) is not None
                else float("inf"),
            )
            if metric.startswith("same_language_irrelevant_share_at_")
            else max(
                ranked,
                key=lambda item: _metric_value(item, metric)
                if _metric_value(item, metric) is not None
                else float("-inf"),
            )
        )
        winner_value = _metric_value(winner, metric)
        if winner_value is None:
            continue
        lines.append(
            f"| `{_metric_label(metric)}` | `{winner.model_name}` | {winner_value:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _build_latex_comparison(
    dataset_repo: str,
    ranked: list[ModelEvaluationSummary],
) -> str:
    best = _best_metric_values(ranked, TABLE_METRICS)
    column_spec = "r l " + " ".join(["r"] * len(TABLE_METRICS)) + " r"
    metric_headers = " & ".join(_latex_escape(_metric_label(metric)) for metric in TABLE_METRICS)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        rf"Rank & Model & {metric_headers} & Time (s) \\",
        r"\hline",
    ]
    for idx, item in enumerate(ranked, start=1):
        eval_time = (
            f"{item.evaluation_time_seconds:.1f}"
            if item.evaluation_time_seconds is not None
            else "--"
        )
        lines.append(
            " & ".join(
                [
                    str(idx),
                    r"\texttt{" + _latex_escape(item.model_name) + "}",
                    *[_latex_metric_cell(item, metric, best) for metric in TABLE_METRICS],
                    eval_time,
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            rf"\caption{{MTEB retrieval comparison on \texttt{{{_latex_escape(dataset_repo)}}}. Bold marks the best score per metric.}}",
            r"\label{tab:mteb-model-comparison}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def generate_mteb_comparison_tables(
    *,
    results_dir: str | Path = DEFAULT_MTEB_OUTPUT_DIR,
    output_dir: str | Path = DEFAULT_MTEB_TABLES_DIR,
) -> Path:
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    dataset_repo, dataset_variant, summaries = _load_summary_models(results_path)
    if not summaries:
        raise ValueError(f"No model summaries found in `{results_path}`.")

    ranked = sorted(summaries, key=lambda item: item.main_score, reverse=True)
    all_metric_keys = _ordered_metric_keys(ranked)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_json = output_path / "model_comparison.json"
    comparison_csv = output_path / "model_comparison.csv"
    comparison_md = output_path / "model_comparison.md"
    comparison_tex = output_path / "model_comparison.tex"

    payload = {
        "dataset_repo": dataset_repo,
        "dataset_variant": dataset_variant,
        "results_dir": str(results_path),
        "metrics": all_metric_keys,
        "table_metrics": TABLE_METRICS,
        "models": [
            {
                "rank": idx,
                "model_name": item.model_name,
                "dataset_variant": item.dataset_variant,
                "main_score": item.main_score,
                "evaluation_time_seconds": item.evaluation_time_seconds,
                "output_dir": item.output_dir,
                "metrics": {metric: _metric_value(item, metric) for metric in all_metric_keys},
            }
            for idx, item in enumerate(ranked, start=1)
        ],
    }
    comparison_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with comparison_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rank",
                "model_name",
                "evaluation_time_seconds",
                "output_dir",
                *all_metric_keys,
            ],
        )
        writer.writeheader()
        for idx, item in enumerate(ranked, start=1):
            row: dict[str, Any] = {
                "rank": idx,
                "model_name": item.model_name,
                "main_score": item.main_score,
                "evaluation_time_seconds": item.evaluation_time_seconds,
                "output_dir": item.output_dir,
            }
            for metric in all_metric_keys:
                row[metric] = item.metrics.get(metric, item.main_score if metric == "main_score" else "")
            writer.writerow(row)

    comparison_md.write_text(
        _build_markdown_comparison(dataset_repo, ranked),
        encoding="utf-8",
    )
    comparison_tex.write_text(
        _build_latex_comparison(dataset_repo, ranked),
        encoding="utf-8",
    )
    return output_path


def run_mteb_evaluation(
    models: list[str],
    *,
    dataset_repo: str = DEFAULT_MTEB_DATASET_REPO,
    dataset_variant: str = DEFAULT_MTEB_VARIANT,
    output_dir: str | Path = DEFAULT_MTEB_OUTPUT_DIR,
    revision: str = "main",
    batch_size: int = 32,
    prediction_dir: str | Path | None = None,
    corpus_repo: str = "",
) -> list[ModelEvaluationSummary]:
    if not models:
        raise ValueError("Provide at least one model name for MTEB evaluation.")

    dataset_variant = _normalize_dataset_variant(dataset_variant)
    task = build_mteb_task(
        dataset_repo, revision=revision, dataset_variant=dataset_variant, corpus_repo=corpus_repo
    )
    evaluator = MTEB(tasks=[task])
    base_output_dir = Path(output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    model_cache_dir = _configure_local_model_cache()
    summaries: list[ModelEvaluationSummary] = []
    eval_languages = _detect_query_languages(dataset_repo, revision, dataset_variant)

    # Load queries/qrels (+ swap in the shared corpus) once, up front, so a
    # dataset/corpus error fails fast and clearly instead of being swallowed by the
    # per-model resilience loop (and so every model uses the same swapped corpus).
    try:
        task.load_data()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load evaluation data (dataset_repo={dataset_repo}, "
            f"corpus_repo={corpus_repo or dataset_repo}). If using the shared corpus, "
            "publish it first with `--push-corpus-hf`, or pass --mteb-corpus-repo '' to use "
            f"the dataset's own corpus. Original error: {exc}"
        ) from exc

    failed: list[tuple[str, str]] = []
    for model_name in models:
        model_slug = _slugify(model_name)
        print(f"Evaluating `{model_name}` on `{dataset_repo}` ({dataset_variant})...")
        # A failing model (e.g. gated/missing weights, missing extra package, OOM) is logged
        # as a warning and skipped so the rest of the run still completes.
        try:
            # Prefer MTEB's registry loader (correct per-model prompts / trust_remote_code /
            # ColBERT late interaction); fall back to a plain SentenceTransformer for models
            # MTEB does not know (e.g. SapBERT).
            try:
                model = mteb.get_model(model_name)
                model_meta = mteb.get_model_meta(model_name)
            except Exception:
                model = SentenceTransformer(
                    model_name, cache_folder=str(model_cache_dir), trust_remote_code=True
                )
                model_meta = evaluator.create_model_meta(model)
            # Work around a transformers non-persistent-buffer bug for known-affected models
            # (e.g. gte-multilingual-base): a garbage position_ids buffer otherwise crashes the
            # first encode (OOB IndexError on CPU / "device-side assert" on GPU).
            if model_name in MODELS_NEEDING_POSITION_IDS_REPAIR:
                _repair_position_ids_buffers(model)
            model_output_dir = base_output_dir / model_meta.model_name_as_path() / (
                model_meta.revision or "no_revision_available"
            )
            run_kwargs: dict[str, Any] = {}
            if prediction_dir is not None:
                # Save per-query rankings (one folder per model) for question-level analysis.
                run_kwargs["prediction_folder"] = Path(prediction_dir) / model_slug
            results = evaluator.run(
                model,
                verbosity=2,
                output_folder=str(base_output_dir),
                eval_splits=["train"],
                overwrite_results=True,
                encode_kwargs={"batch_size": batch_size},
                **run_kwargs,
            )
            if not results:
                raise ValueError(f"MTEB returned no results for model `{model_name}`.")

            result = results[0]
            metrics = _extract_numeric_metrics(result)
            summary = ModelEvaluationSummary(
                model_name=model_name,
                model_slug=model_slug,
                dataset_variant=dataset_variant,
                main_score=float(result.main_score),
                metrics=metrics,
                output_dir=str(model_output_dir),
                eval_languages=eval_languages,
                evaluation_time_seconds=result.evaluation_time,
            )
            summaries.append(summary)
        except Exception as exc:  # noqa: BLE001 - resilience: skip the model, keep going
            failed.append((model_name, f"{type(exc).__name__}: {exc}"))
            print(f"\n[WARNING] Skipping `{model_name}` — evaluation failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            print()

    if failed:
        print(f"\n{len(failed)} model(s) skipped due to errors:")
        for name, err in failed:
            print(f"  - {name}: {err}")
    if not summaries:
        print("\n[WARNING] No models evaluated successfully; no summary reports written.")
        return summaries

    _write_summary_reports(base_output_dir, dataset_repo, dataset_variant, summaries)
    return summaries
