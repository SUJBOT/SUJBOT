"""
Loader for ViDoRe V3 benchmark datasets (BEIR format, Arrow files).

Provides VideoreDataset — a lightweight wrapper for datasets stored as
HuggingFace datasets on disk (Arrow tables with queries/, corpus/, and qrels/).

Usage:
    from rag_confidence.data.vidore_loader import load_english_datasets

    datasets = load_english_datasets()
    for name, ds in datasets.items():
        queries = ds.get_queries()        # {query_id: text}
        qrels = ds.get_qrels_dict()       # {query_id: {corpus_id: score}}
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(
    os.environ.get("RAG_CONFIDENCE_DATA_DIR", Path(__file__).parent / "vidore_v3")
)
if not str(DATA_DIR).endswith("vidore_v3"):
    DATA_DIR = DATA_DIR / "vidore_v3"

DATASETS_DIR = DATA_DIR / "data"

# BEIR-format datasets (corpus/queries/qrels subdirectories)
BEIR_DATASETS = [
    "vidore_v3_physics",
    "vidore_v3_energy",
    "vidore_v3_finance_en",
    "vidore_v3_finance_fr",
    "vidore_v3_hr",
    "vidore_v3_industrial",
    "vidore_v3_computer_science",
    "vidore_v3_pharmaceuticals",
    "esg_reports_v2",
    "esg_reports_eng_v2",
    "esg_reports_human_labeled_v2",
    "economics_reports_v2",
    "economics_reports_eng_v2",
    "biomedical_lectures_v2",
    "biomedical_lectures_eng_v2",
]

# English-only BEIR datasets (for cross-domain benchmarks)
ENGLISH_BEIR_DATASETS = [
    "vidore_v3_physics",
    "vidore_v3_energy",
    "vidore_v3_finance_en",
    "vidore_v3_hr",
    "vidore_v3_industrial",
    "vidore_v3_computer_science",
    "vidore_v3_pharmaceuticals",
    "esg_reports_eng_v2",
    "economics_reports_eng_v2",
    "biomedical_lectures_eng_v2",
]

# Non-English datasets
NON_ENGLISH_DATASETS = [
    "vidore_v3_finance_fr",
    "esg_reports_v2",
    "economics_reports_v2",
    "biomedical_lectures_v2",
]


def _load_arrow_table(arrow_dir: Path):
    """Load a HuggingFace datasets Arrow table from disk."""
    from datasets import Dataset

    arrow_files = sorted(arrow_dir.glob("data-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No arrow files in {arrow_dir}")

    # Load first shard (most datasets are single-shard)
    ds = Dataset.from_file(str(arrow_files[0]))

    # If multi-shard, concatenate
    if len(arrow_files) > 1:
        from datasets import concatenate_datasets

        shards = [Dataset.from_file(str(f)) for f in arrow_files[1:]]
        ds = concatenate_datasets([ds] + shards)

    return ds


class VideoreDataset:
    """Lightweight wrapper for a BEIR-format ViDoRe dataset.

    Expected directory structure:
        {dataset_dir}/
        ├── queries/
        │   └── data-00000-of-00001.arrow   (query_id, query, language, ...)
        ├── corpus/
        │   └── data-*.arrow                (corpus documents)
        └── qrels/
            └── data-00000-of-00001.arrow   (query_id, corpus_id, score)
    """

    def __init__(self, name: str, dataset_dir: Optional[Path] = None):
        self.name = name
        self.dataset_dir = dataset_dir or DATASETS_DIR / name

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        self._queries = None
        self._qrels = None
        self._n_corpus = None

    def get_queries(self) -> dict[int, str]:
        """Return {query_id: query_text} mapping."""
        if self._queries is None:
            queries_dir = self.dataset_dir / "queries"
            ds = _load_arrow_table(queries_dir)
            # Handle both column naming conventions (query_id vs query-id)
            qid_col = "query_id" if "query_id" in ds.column_names else "query-id"
            self._queries = {
                int(row[qid_col]): str(row["query"]) for row in ds
            }
        return self._queries

    def get_qrels_dict(self) -> dict[int, dict[int, int]]:
        """Return {query_id: {corpus_id: score}} mapping."""
        if self._qrels is None:
            qrels_dir = self.dataset_dir / "qrels"
            ds = _load_arrow_table(qrels_dir)
            # Handle both column naming conventions
            qid_col = "query_id" if "query_id" in ds.column_names else "query-id"
            cid_col = "corpus_id" if "corpus_id" in ds.column_names else "corpus-id"
            self._qrels = {}
            for row in ds:
                qid = int(row[qid_col])
                cid = int(row[cid_col])
                score = int(row["score"])
                if qid not in self._qrels:
                    self._qrels[qid] = {}
                self._qrels[qid][cid] = score
        return self._qrels

    @property
    def n_queries(self) -> int:
        return len(self.get_queries())

    @property
    def n_corpus(self) -> int:
        if self._n_corpus is None:
            corpus_dir = self.dataset_dir / "corpus"
            if corpus_dir.exists():
                ds = _load_arrow_table(corpus_dir)
                self._n_corpus = len(ds)
            else:
                # Estimate from qrels
                qrels = self.get_qrels_dict()
                all_corpus = set()
                for cids in qrels.values():
                    all_corpus.update(cids.keys())
                self._n_corpus = len(all_corpus)
        return self._n_corpus

    def __repr__(self) -> str:
        return f"VideoreDataset(name={self.name!r}, queries={self.n_queries})"


def load_all_datasets() -> dict[str, VideoreDataset]:
    """Load all BEIR-format datasets that exist on disk."""
    datasets = {}
    for name in BEIR_DATASETS:
        ds_dir = DATASETS_DIR / name
        if not ds_dir.exists():
            logger.warning(f"Dataset not found, skipping: {name}")
            continue
        if not (ds_dir / "queries").exists() or not (ds_dir / "qrels").exists():
            logger.warning(f"Dataset incomplete (missing queries/qrels), skipping: {name}")
            continue
        try:
            datasets[name] = VideoreDataset(name, ds_dir)
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")
    logger.info(f"Loaded {len(datasets)} datasets")
    return datasets


def load_english_datasets() -> dict[str, VideoreDataset]:
    """Load only English BEIR-format datasets."""
    datasets = {}
    for name in ENGLISH_BEIR_DATASETS:
        ds_dir = DATASETS_DIR / name
        if not ds_dir.exists():
            logger.warning(f"Dataset not found, skipping: {name}")
            continue
        if not (ds_dir / "queries").exists() or not (ds_dir / "qrels").exists():
            logger.warning(f"Dataset incomplete, skipping: {name}")
            continue
        try:
            datasets[name] = VideoreDataset(name, ds_dir)
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")
    logger.info(f"Loaded {len(datasets)} English datasets")
    return datasets
