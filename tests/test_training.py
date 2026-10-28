import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from models import train_model as training


class TrainingTests(unittest.TestCase):
    def transactions(self):
        return pd.DataFrame({
            "transaction_id": ["late", "early", "invalid"],
            "user_id": ["u", "u", "u"],
            "amount": [20.5, 10.25, 99.0],
            "location": ["NY", "CA", "CA"],
            "timestamp": ["2025-01-01T03:00:00Z", "2025-01-01T00:00:00Z", "bad"],
            "label": [1, 0, 1],
        })

    def test_labels_survive_sorting_and_invalid_timestamp_filter(self):
        frame, features, labels = training.build_feature_frame(self.transactions())
        self.assertIsNotNone(labels)
        self.assertEqual(dict(zip(frame.transaction_id, labels)), {"early": 0, "late": 1})
        self.assertNotIn("label", features)
        self.assertEqual(len(frame), 2)

    def test_unlabeled_input_remains_unsupervised(self):
        _, _, labels = training.build_feature_frame(self.transactions().drop(columns="label"))
        self.assertIsNone(labels)

    def test_expired_history_is_excluded_from_window(self):
        frame, _, _ = training.build_feature_frame(self.transactions())
        late = frame.loc[frame.transaction_id == "late"].iloc[0]
        self.assertEqual(late.user_cnt_1h, 0)
        self.assertEqual(late.user_sum_1h, 0)
        self.assertEqual(late.user_cnt_24h, 1)
        self.assertEqual(late.user_sum_24h, 10.25)

    def test_unsupervised_scaler_keeps_fractional_features(self):
        frame = pd.DataFrame({"amount": [1.25, 1.75, 2.5], "hour_sin": [0.25, 0.5, 0.75]})
        artifacts = training.train_unsupervised_model(frame, list(frame.columns))
        np.testing.assert_allclose(artifacts["scaler"].center_, [1.75, 0.5])

    def test_f1_point_uses_matching_threshold(self):
        self.assertEqual(training.best_f1_threshold([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.8]), 0.6)


if __name__ == "__main__":
    unittest.main()
