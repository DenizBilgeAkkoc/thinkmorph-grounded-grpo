import errno
import json
import unittest
from unittest.mock import patch
import io_resilience as io

class RecoveryIOTests(unittest.TestCase):
    @patch.object(io.time, 'sleep')
    def test_stale_handle_reopens_and_succeeds(self, sleep):
        calls = []
        def read():
            calls.append(1)
            if len(calls) < 3:
                raise OSError(errno.ESTALE, 'stale handle')
            return {'saved': True}
        self.assertEqual(io.strict_io(read)(), {'saved': True})
        self.assertEqual(len(calls), 3)

    @patch.object(io.time, 'sleep')
    def test_exhausted_required_checkpoint_read_is_fatal(self, sleep):
        def read():
            raise OSError(errno.ESTALE, 'stale handle')
        with self.assertRaises(OSError):
            io.strict_io(read)()
        self.assertEqual(sleep.call_count, 3)

    @patch.object(io.time, 'sleep')
    def test_exhausted_seen_export_is_deferred(self, sleep):
        def export():
            raise OSError(errno.ESTALE, 'stale handle')
        self.assertIsNone(io.optional_monitoring(export)())
        self.assertEqual(sleep.call_count, 3)

    def test_semantic_failures_are_never_suppressed(self):
        for error in (ValueError('checkpoint mismatch'), json.JSONDecodeError('bad JSON', '', 0),
                      PermissionError(errno.EACCES, 'access denied'), RuntimeError('failed check')):
            def corrupt():
                raise error
            with self.assertRaises(type(error)):
                io.optional_monitoring(corrupt)()

    def test_success_preserves_return_value(self):
        self.assertEqual(io.optional_monitoring(lambda: 42)(), 42)

if __name__ == '__main__':
    unittest.main()
