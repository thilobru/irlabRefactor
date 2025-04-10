import pytest
import numpy as np

# Assume src is in PYTHONPATH or adjust import path accordingly
from src.evaluation import precision_at_k, average_precision, dcg_at_k, ndcg_at_k, calculate_metrics

# --- Fixtures ---
@pytest.fixture
def eval_data():
    """Provides common data for evaluation metric tests."""
    return {
        'retrieved_ids': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5'], # Length 5
        'relevant_ids': ['doc1', 'doc3', 'doc6'], # doc6 is relevant but not retrieved
        'relevant_grades': {'doc1': 2, 'doc3': 1, 'doc6': 3, 'doc2': 0, 'doc7': 1} # Includes non-retrieved, zero grades
    }

# --- Precision@k Tests ---
@pytest.mark.parametrize("k, expected", [
    (3, 2/3), # doc1, doc3 relevant in top 3
    (5, 2/5), # doc1, doc3 relevant in top 5
    (1, 1/1), # doc1 is relevant
    (0, 0.0),
    (10, 2/10) # k=10 > retrieved length(5). 2 relevant in top 5 (which is top 10). Denominator is k=10. Expected = 0.2
])
def test_precision_at_k(eval_data, k, expected):
    assert precision_at_k(eval_data['retrieved_ids'], eval_data['relevant_ids'], k) == pytest.approx(expected)

def test_precision_at_k_empty_retrieved(eval_data):
    assert precision_at_k([], eval_data['relevant_ids'], k=5) == 0.0

def test_precision_at_k_empty_relevant(eval_data):
    assert precision_at_k(eval_data['retrieved_ids'], [], k=5) == 0.0

# --- Average Precision Tests ---
def test_average_precision_basic(eval_data):
    # Ranks of relevant retrieved docs: 1 (doc1), 3 (doc3)
    # P@1 = 1/1 = 1.0
    # P@3 = 2/3 = 0.666...
    # Total relevant = 3 (doc1, doc3, doc6)
    # AP = (P@1 + P@3) / 3 = (1.0 + 2/3) / 3 = (5/3) / 3 = 5/9
    expected_ap = (1.0 + 2/3) / 3
    assert average_precision(eval_data['retrieved_ids'], eval_data['relevant_ids']) == pytest.approx(expected_ap)

def test_average_precision_no_relevant(eval_data):
    assert average_precision(eval_data['retrieved_ids'], []) == 0.0

def test_average_precision_no_retrieved(eval_data):
     assert average_precision([], eval_data['relevant_ids']) == 0.0

def test_average_precision_all_retrieved_relevant():
     retrieved = ['d1', 'd2']
     relevant = ['d1', 'd2']
     # P@1 = 1/1 = 1.0; P@2 = 2/2 = 1.0; AP = (1.0 + 1.0) / 2 = 1.0
     assert average_precision(retrieved, relevant) == pytest.approx(1.0)

# --- DCG@k Tests ---
@pytest.mark.parametrize("k, expected_dcg", [
    (0, 0.0),
    (3, (2.0 / np.log2(2) + 0.0 / np.log2(3) + 1.0 / np.log2(4))), # 2.5
    (5, (2.0 / np.log2(2) + 0.0 / np.log2(3) + 1.0 / np.log2(4) + 0.0 / np.log2(5) + 0.0 / np.log2(6))) # 2.5
])
def test_dcg_at_k(eval_data, k, expected_dcg):
    assert dcg_at_k(eval_data['retrieved_ids'], eval_data['relevant_grades'], k) == pytest.approx(expected_dcg)

def test_dcg_at_k_no_relevant(eval_data):
    assert dcg_at_k(eval_data['retrieved_ids'], {}, k=5) == 0.0

# --- NDCG@k Tests ---
def test_ndcg_at_k_basic(eval_data):
    # Ideal ranking based on grades: doc6 (3), doc1 (2), doc3 (1), doc7 (1), doc2 (0)
    # IDCG@3 = grade(doc6)/log2(2) + grade(doc1)/log2(3) + grade(doc3)/log2(4)
    ideal_dcg3 = 3.0 / np.log2(2) + 2.0 / np.log2(3) + 1.0 / np.log2(4)
    # DCG@3 = 2.5 (from previous test)
    actual_dcg3 = 2.0 / np.log2(2) + 0.0 / np.log2(3) + 1.0 / np.log2(4)
    expected_ndcg3 = actual_dcg3 / ideal_dcg3 if ideal_dcg3 > 0 else 0.0
    assert ndcg_at_k(eval_data['retrieved_ids'], eval_data['relevant_grades'], k=3) == pytest.approx(expected_ndcg3)

def test_ndcg_at_k_zero_k(eval_data):
    assert ndcg_at_k(eval_data['retrieved_ids'], eval_data['relevant_grades'], k=0) == 0.0

def test_ndcg_at_k_zero_idcg(eval_data):
    # If all relevant items have grade 0, IDCG is 0, NDCG should be 0
    zero_grades = {'doc1': 0, 'doc3': 0}
    assert ndcg_at_k(eval_data['retrieved_ids'], zero_grades, k=3) == 0.0
    # If no relevant items, IDCG is 0, NDCG should be 0
    assert ndcg_at_k(eval_data['retrieved_ids'], {}, k=3) == 0.0

# --- calculate_metrics Tests ---
def test_calculate_metrics(eval_data):
    """Test the main metric calculation function."""
    retrieved_results = list(zip(eval_data['retrieved_ids'], [0.9, 0.8, 0.7, 0.6, 0.5])) # (doc_id, score) tuples
    qrels_for_topic = eval_data['relevant_grades']
    metrics_to_calc = ['precision@3', 'map', 'ndcg@3']

    results = calculate_metrics(retrieved_results, qrels_for_topic, metrics_to_calc)

    # Calculate expected values based on previous tests
    expected_p3 = 2/3
    relevant_ids_for_ap = [doc_id for doc_id, grade in qrels_for_topic.items() if grade > 0] # doc1, doc3, doc6, doc7 -> 4 relevant
    # Ranks of relevant retrieved: 1 (doc1), 3 (doc3)
    expected_map = (1.0 + 2/3) / len(relevant_ids_for_ap) if relevant_ids_for_ap else 0.0 # AP = (P@1 + P@3) / 4 = 5/12

    ideal_dcg3 = 3.0 / np.log2(2) + 2.0 / np.log2(3) + 1.0 / np.log2(4)
    actual_dcg3 = 2.0 / np.log2(2) + 0.0 / np.log2(3) + 1.0 / np.log2(4)
    expected_ndcg3 = actual_dcg3 / ideal_dcg3 if ideal_dcg3 > 0 else 0.0

    assert 'precision@3' in results
    assert results['precision@3'] == pytest.approx(expected_p3)
    assert 'map' in results
    assert results['map'] == pytest.approx(expected_map) # Checks AP for the topic
    assert 'ndcg@3' in results
    assert results['ndcg@3'] == pytest.approx(expected_ndcg3)

def test_calculate_metrics_unsupported(eval_data):
    """Test that unsupported metrics are skipped."""
    retrieved_results = list(zip(eval_data['retrieved_ids'], [0.9, 0.8, 0.7, 0.6, 0.5]))
    qrels_for_topic = eval_data['relevant_grades']
    metrics_to_calc = ['precision@1', 'recall@5', 'ndcg@1'] # recall@5 is not implemented
    results = calculate_metrics(retrieved_results, qrels_for_topic, metrics_to_calc)

    assert 'precision@1' in results
    assert 'recall@5' not in results # Should be skipped
    assert 'ndcg@1' in results

