"""
Tests for Generalized QPP vs LLM Benchmark Framework.

Tests cover:
- GeneralQPPExtractor: Feature extraction
- QPPModelFactory: Model creation and training
- BenchmarkDataset: Data loading and manipulation
- GeneralizedBenchmark: Integration tests
"""

import numpy as np
import pytest

from rag_confidence.core.general_qpp_extractor import (
    GeneralQPPExtractor,
    GeneralQPPFeatures,
)
from rag_confidence.core.qpp_model import (
    QPPModelFactory,
    LogisticRegressionModel,
    MLPModel,
    load_model,
)
from rag_confidence.evaluation.benchmark_dataset import (
    BenchmarkDataset,
    InMemoryDataset,
)


class TestGeneralQPPExtractor:
    """Tests for GeneralQPPExtractor."""

    @pytest.fixture
    def extractor(self):
        return GeneralQPPExtractor(tau=0.711)

    @pytest.fixture
    def good_similarities(self):
        """Similarities with clear top results (good retrieval)."""
        return np.concatenate([
            np.array([0.90, 0.85, 0.80, 0.75, 0.60, 0.50, 0.45, 0.40, 0.35, 0.30]),
            0.2 + 0.1 * np.random.random(90)
        ])

    @pytest.fixture
    def bad_similarities(self):
        """Flat similarities (bad retrieval)."""
        return np.concatenate([
            np.array([0.65, 0.64, 0.63, 0.62, 0.61, 0.60, 0.59, 0.58, 0.57, 0.56]),
            0.5 + 0.1 * np.random.random(90)
        ])

    def test_feature_count(self, extractor):
        """Should extract exactly 24 features."""
        assert extractor.n_features() == 24
        assert len(extractor.FEATURE_NAMES) == 24

    def test_feature_tiers(self, extractor):
        """Should have correct feature tier breakdown."""
        tiers = extractor.get_feature_tiers()
        assert len(tiers["distribution"]) == 15  # 14 + gap_concentration
        assert len(tiers["query_text"]) == 5
        assert len(tiers["global_similarity"]) == 4

    def test_extract_returns_dataclass(self, extractor, good_similarities):
        """Extract should return GeneralQPPFeatures dataclass."""
        query = "What are the requirements?"
        features = extractor.extract(query, good_similarities)

        assert isinstance(features, GeneralQPPFeatures)
        assert hasattr(features, "top1_minus_p99")
        assert hasattr(features, "query_char_len")
        assert hasattr(features, "sim_mean_all")

    def test_feature_vector_shape(self, extractor, good_similarities):
        """Feature vector should have shape (24,)."""
        query = "Test query"
        features = extractor.extract(query, good_similarities)
        vector = features.to_vector()

        assert vector.shape == (24,)
        assert vector.dtype == np.float32

    def test_batch_extraction(self, extractor, good_similarities, bad_similarities):
        """Batch extraction should work correctly."""
        queries = ["Query 1", "Query 2"]
        sim_matrix = np.vstack([good_similarities, bad_similarities])

        features = extractor.extract_batch(queries, sim_matrix)

        assert features.shape == (2, 24)

    def test_good_vs_bad_retrieval(self, extractor, good_similarities, bad_similarities):
        """Good retrieval should have different feature values than bad retrieval."""
        query = "Test query"

        good_features = extractor.extract(query, good_similarities)
        bad_features = extractor.extract(query, bad_similarities)

        # Good retrieval should have higher gap features
        assert good_features.top1_minus_p99 > bad_features.top1_minus_p99
        assert good_features.top1_vs_top10_gap > bad_features.top1_vs_top10_gap
        assert good_features.bimodal_gap > bad_features.bimodal_gap

    def test_query_features_language_agnostic(self, extractor, good_similarities):
        """Query features should work for any language."""
        queries = [
            "What is the answer?",  # English
            "Jaká je odpověď?",  # Czech
            "Quelle est la réponse?",  # French
            "答えは何ですか？",  # Japanese
        ]

        for query in queries:
            features = extractor.extract(query, good_similarities)
            # Basic checks - all should have positive char/word counts
            assert features.query_char_len > 0
            assert features.query_word_count > 0

    def test_handles_small_corpus(self, extractor):
        """Should handle small similarity arrays gracefully."""
        query = "Test"
        small_sims = np.array([0.8, 0.7, 0.6, 0.5, 0.4])

        features = extractor.extract(query, small_sims)
        vector = features.to_vector()

        assert vector.shape == (24,)
        assert not np.any(np.isnan(vector))


class TestQPPModelFactory:
    """Tests for QPP model factory and models."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic training data."""
        np.random.seed(42)
        n_train = 200
        n_test = 50
        n_features = 24  # Updated for 24-feature extractor

        X = np.random.randn(n_train + n_test, n_features)
        # Create labels based on first few features
        y = (X[:, 0] + X[:, 1] + 0.5 * np.random.randn(n_train + n_test) > 0).astype(int)

        return {
            "X_train": X[:n_train],
            "y_train": y[:n_train],
            "X_test": X[n_train:],
            "y_test": y[n_train:],
        }

    def test_create_logistic(self):
        """Should create LogisticRegression model."""
        model = QPPModelFactory.create("logistic")
        assert isinstance(model, LogisticRegressionModel)

    def test_create_mlp(self):
        """Should create MLP model."""
        model = QPPModelFactory.create("mlp")
        assert isinstance(model, MLPModel)

    def test_create_with_params(self):
        """Should pass parameters to model."""
        model = QPPModelFactory.create("mlp", hidden_sizes=(32, 16), alpha=0.1)
        assert model.hidden_sizes == (32, 16)
        assert model.alpha == 0.1

    def test_unknown_model_raises(self):
        """Should raise for unknown model type."""
        with pytest.raises(ValueError):
            QPPModelFactory.create("unknown_model")

    def test_list_available(self):
        """Should list available models."""
        models = QPPModelFactory.list_available()
        assert "logistic" in models
        assert "mlp" in models
        assert "xgboost" in models

    def test_fit_and_predict(self, synthetic_data):
        """Model should fit and predict."""
        model = QPPModelFactory.create("logistic")
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])

        assert model.is_fitted

        probas = model.predict_proba(synthetic_data["X_test"])
        assert probas.shape == (50,)
        assert np.all((probas >= 0) & (probas <= 1))

    def test_predict_binary(self, synthetic_data):
        """Model should predict binary labels."""
        model = QPPModelFactory.create("logistic")
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])

        preds = model.predict(synthetic_data["X_test"])
        assert preds.shape == (50,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_save_and_load(self, synthetic_data, tmp_path):
        """Model should save and load correctly."""
        model = QPPModelFactory.create("logistic")
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])

        original_probas = model.predict_proba(synthetic_data["X_test"])

        # Save
        save_path = tmp_path / "model.pkl"
        model.save(save_path)

        # Load
        loaded_model = load_model(save_path)
        loaded_probas = loaded_model.predict_proba(synthetic_data["X_test"])

        np.testing.assert_array_almost_equal(original_probas, loaded_probas)

    def test_mlp_trains_successfully(self, synthetic_data):
        """MLP should train without errors."""
        model = QPPModelFactory.create("mlp", hidden_sizes=(32,), max_iter=100)
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])

        probas = model.predict_proba(synthetic_data["X_test"])
        assert probas.shape == (50,)

    def test_predict_before_fit_raises(self, synthetic_data):
        """Should raise if predict called before fit."""
        model = QPPModelFactory.create("logistic")

        with pytest.raises(ValueError):
            model.predict_proba(synthetic_data["X_test"])


class TestBenchmarkDataset:
    """Tests for benchmark dataset interface."""

    @pytest.fixture
    def synthetic_dataset(self):
        return InMemoryDataset.create_synthetic(
            n_queries=100,
            n_chunks=200,
            positive_rate=0.8,
            random_state=42,
        )

    def test_create_synthetic(self, synthetic_dataset):
        """Should create synthetic dataset."""
        assert synthetic_dataset.n_queries == 100
        assert synthetic_dataset.n_chunks == 200
        assert synthetic_dataset.name == "synthetic"

    def test_describe(self, synthetic_dataset):
        """Should return dataset description."""
        desc = synthetic_dataset.describe()

        assert "n_queries" in desc
        assert "n_chunks" in desc
        assert "positive_rate_k10" in desc
        assert desc["has_chunk_texts"] is True

    def test_train_test_split(self, synthetic_dataset):
        """Should create valid train/test split."""
        split = synthetic_dataset.train_test_split(test_size=0.2, random_state=42)

        assert len(split.train_indices) == 80
        assert len(split.test_indices) == 20
        assert split.random_state == 42

        # No overlap
        assert len(set(split.train_indices) & set(split.test_indices)) == 0

    def test_get_train_data(self, synthetic_dataset):
        """Should return train data correctly."""
        split = synthetic_dataset.train_test_split(test_size=0.2)
        queries, sims, labels = synthetic_dataset.get_train_data(split)

        assert len(queries) == 80
        assert sims.shape[0] == 80
        assert labels.shape[0] == 80

    def test_get_test_data(self, synthetic_dataset):
        """Should return test data correctly."""
        split = synthetic_dataset.train_test_split(test_size=0.2)
        queries, sims, labels = synthetic_dataset.get_test_data(split)

        assert len(queries) == 20
        assert sims.shape[0] == 20
        assert labels.shape[0] == 20

    def test_compute_recall_at_k(self, synthetic_dataset):
        """Should compute binary recall@k."""
        recall = synthetic_dataset.compute_recall_at_k(k=10)

        assert recall.shape == (100,)
        assert set(np.unique(recall)).issubset({0.0, 1.0})

    def test_get_chunks_for_query(self, synthetic_dataset):
        """Should return top-k chunks for query."""
        chunks = synthetic_dataset.get_chunks_for_query(0, top_k=5)

        assert len(chunks) == 5
        assert "chunk_id" in chunks[0]
        assert "similarity" in chunks[0]
        assert "content" in chunks[0]

        # Should be sorted by similarity (descending)
        sims = [c["similarity"] for c in chunks]
        assert sims == sorted(sims, reverse=True)

    def test_in_memory_custom(self):
        """Should create custom in-memory dataset."""
        queries = ["Q1", "Q2", "Q3"]
        similarities = np.random.rand(3, 10)
        relevance = np.zeros((3, 10))
        relevance[0, 0] = 1  # Q1 has relevant chunk at position 0

        dataset = InMemoryDataset(
            queries_list=queries,
            similarity_matrix=similarities,
            relevance_matrix=relevance,
            name="custom",
        )

        assert dataset.n_queries == 3
        assert dataset.n_chunks == 10


class TestGeneralizedBenchmarkIntegration:
    """Integration tests for the full benchmark."""

    @pytest.fixture
    def synthetic_dataset(self):
        return InMemoryDataset.create_synthetic(
            n_queries=100,
            n_chunks=200,
            positive_rate=0.85,
            random_state=42,
        )

    def test_benchmark_qpp_only(self, synthetic_dataset):
        """Should run QPP-only benchmark."""
        from rag_confidence.evaluation.generalized_benchmark import GeneralizedBenchmark

        benchmark = GeneralizedBenchmark(
            dataset=synthetic_dataset,
            qpp_model_types=["logistic"],
            llm_model=None,  # Skip LLM
            k=10,
            test_size=0.2,
        )

        results = benchmark.run()

        assert "logistic" in results.qpp_results
        assert len(results.llm_results) == 0

        qpp = results.qpp_results["logistic"]
        assert "auroc" in qpp.metrics
        assert 0.5 <= qpp.metrics["auroc"] <= 1.0

    def test_benchmark_multiple_qpp_models(self, synthetic_dataset):
        """Should run benchmark with multiple QPP models."""
        from rag_confidence.evaluation.generalized_benchmark import GeneralizedBenchmark

        benchmark = GeneralizedBenchmark(
            dataset=synthetic_dataset,
            qpp_model_types=["logistic", "mlp"],
            llm_model=None,
        )

        results = benchmark.run()

        assert "logistic" in results.qpp_results
        assert "mlp" in results.qpp_results

    def test_benchmark_results_serialization(self, synthetic_dataset, tmp_path):
        """Results should serialize to JSON."""
        from rag_confidence.evaluation.generalized_benchmark import GeneralizedBenchmark

        benchmark = GeneralizedBenchmark(
            dataset=synthetic_dataset,
            qpp_model_types=["logistic"],
        )

        results = benchmark.run()

        # Save
        output_path = tmp_path / "results.json"
        benchmark.save_results(results, output_path)

        # Load and verify
        import json
        with open(output_path) as f:
            loaded = json.load(f)

        assert "metadata" in loaded
        assert "qpp" in loaded
        assert "logistic" in loaded["qpp"]

    def test_benchmark_with_limit(self, synthetic_dataset):
        """Should respect limit parameter."""
        from rag_confidence.evaluation.generalized_benchmark import GeneralizedBenchmark

        benchmark = GeneralizedBenchmark(
            dataset=synthetic_dataset,
            qpp_model_types=["logistic"],
        )

        results = benchmark.run(limit=10)

        assert results.metadata["n_test"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
