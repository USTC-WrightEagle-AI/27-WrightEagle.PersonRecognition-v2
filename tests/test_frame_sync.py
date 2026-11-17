import unittest
from types import SimpleNamespace

from face_robot.frame_sync import FrameSynchronizer


def message(milliseconds):
    seconds, remainder = divmod(milliseconds, 1000)
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(
        sec=seconds, nanosec=remainder * 1000000)))


class FrameSynchronizerTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.sync = FrameSynchronizer(True, clock=lambda: self.now)

    def test_delayed_depth_matches_rgb_that_was_previously_overwritten(self):
        original = message(1000)
        self.sync.add('rgb', original)
        self.sync.add('rgb', message(1100))
        self.sync.add('depth', message(1000))
        color, depth = self.sync.pop()
        self.assertIs(color, original)
        self.assertEqual(depth.header.stamp.sec, 1)

    def test_depth_can_arrive_first(self):
        depth = message(1000)
        self.sync.add('depth', depth)
        self.assertIsNone(self.sync.pop())
        self.sync.add('rgb', message(1010))
        self.assertIs(self.sync.pop()[1], depth)

    def test_depth_is_not_reused_for_another_face_frame(self):
        self.sync.add('rgb', message(1000))
        self.sync.add('depth', message(1000))
        self.assertIsNotNone(self.sync.pop())
        self.sync.add('rgb', message(1030))
        self.assertIsNone(self.sync.pop())

    def test_multi_second_timestamp_lag_is_not_accepted(self):
        self.sync.add('rgb', message(6000))
        self.sync.add('depth', message(1000))
        self.assertIsNone(self.sync.pop())
        self.assertIn('latest_rgb_minus_depth=+5.000s', self.sync.diagnostics())

    def test_pair_tolerance_is_preserved(self):
        self.sync.add('rgb', message(1000))
        self.sync.add('depth', message(1051))
        self.assertIsNone(self.sync.pop())
        self.sync.add('depth', message(1049))
        self.assertIsNotNone(self.sync.pop())

    def test_stale_color_and_stale_depth_are_rejected(self):
        for first, second in [('rgb', 'depth'), ('depth', 'rgb')]:
            with self.subTest(first=first):
                self.setUp()
                self.sync.add(first, message(1000))
                self.now = 0.51
                self.sync.add(second, message(1000))
                self.assertIsNone(self.sync.pop())

    def test_selects_newest_pair_after_calibration_gate_opens(self):
        for t in (1000, 1100):
            self.sync.add('rgb', message(t))
            self.sync.add('depth', message(t))
        color, _ = self.sync.pop()
        self.assertEqual(color.header.stamp.nanosec, 100000000)
        self.assertIsNone(self.sync.pop())

    def test_buffers_are_bounded_when_one_stream_is_missing(self):
        for t in range(100):
            self.sync.add('rgb', message(t))
        self.assertEqual(len(self.sync.colors), 16)
        self.assertIn('depth: received=0, last_arrival_age=never', self.sync.diagnostics())

    def test_rgb_only_does_not_wait_for_depth(self):
        sync = FrameSynchronizer(False, clock=lambda: self.now)
        sync.add('rgb', message(1000))
        latest = message(1100)
        sync.add('rgb', latest)
        self.assertEqual(sync.pop(), (latest, None))
        self.assertIsNone(sync.pop())


if __name__ == '__main__':
    unittest.main()
