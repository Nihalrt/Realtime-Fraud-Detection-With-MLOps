import importlib.util
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("alert_sink", Path(__file__).resolve().parents[1] / "services/alert_sink/main.py")
sink = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sink)


class AlertSinkTests(unittest.TestCase):
    def records(self):
        return {"partition": [SimpleNamespace(value={"transaction_id": "t", "user_id": "u", "fraud_score": 0.8, "timestamp": "2025-01-01T00:00:00Z"})]}

    def test_database_commit_precedes_offset_commit(self):
        consumer, conn, order = MagicMock(), MagicMock(), []
        consumer.poll.side_effect = [self.records(), KeyboardInterrupt]
        conn.commit.side_effect = lambda: order.append("db")
        consumer.commit.side_effect = lambda: order.append("kafka")
        with patch.object(sink, "get_consumer", return_value=consumer), patch.object(sink, "get_conn", return_value=conn):
            with self.assertRaises(KeyboardInterrupt):
                sink.main()
        self.assertEqual(order, ["db", "kafka"])
        consumer.close.assert_called_once()
        conn.close.assert_called_once()

    def test_insert_failure_rolls_back_without_committing_offsets(self):
        consumer, conn = MagicMock(), MagicMock()
        consumer.poll.return_value = self.records()
        conn.cursor.return_value.__enter__.return_value.execute.side_effect = RuntimeError("database unavailable")
        with patch.object(sink, "get_consumer", return_value=consumer), patch.object(sink, "get_conn", return_value=conn):
            with self.assertRaises(RuntimeError):
                sink.main()
        conn.rollback.assert_called_once()
        conn.commit.assert_not_called()
        consumer.commit.assert_not_called()
        consumer.close.assert_called_once()
        conn.close.assert_called_once()

    def test_consumer_disables_auto_commit(self):
        with patch.object(sink, "KafkaConsumer") as factory:
            sink.get_consumer()
        self.assertFalse(factory.call_args.kwargs["enable_auto_commit"])


if __name__ == "__main__":
    unittest.main()
