import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('viewer_camera_session', ROOT / 'scripts/camera_session.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DetectionViewerTests(unittest.TestCase):
    def test_default_detection_opens_viewer_but_no_image_does_not(self):
        command = ['bash', str(ROOT / 'posture_recognition/scripts/start_3d.sh')]
        with mock.patch.dict(os.environ, {'DISPLAY': ':0'}):
            self.assertEqual(module.detection_viewer('detection', command)[0], 'bash')
            self.assertIsNone(module.detection_viewer('detection', command + ['--no-image']))
            self.assertIsNone(module.detection_viewer('enroll', command))
            self.assertIsNone(module.detection_viewer('detection', [sys.executable, 'diagnostic.py']))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(module.detection_viewer('detection', command))

    def test_detector_exit_closes_viewer_and_preserves_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'viewer.pid'
            viewer = [sys.executable, '-c',
                      f'import os,time,signal,sys; signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)']
            result = module.run_command([sys.executable, '-c', 'import time; time.sleep(.5); raise SystemExit(7)'], viewer=viewer)
            self.assertEqual(result, 7)
            self.assertTrue(marker.exists())
            with self.assertRaises(ProcessLookupError):
                os.kill(int(marker.read_text()), 0)

    def test_closing_viewer_does_not_end_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'detector-finished'
            app = [sys.executable, '-c',
                   f'import time; from pathlib import Path; time.sleep(.3); Path({str(marker)!r}).touch()']
            self.assertEqual(module.run_command(app, viewer=[sys.executable, '-c', 'pass']), 0)
            self.assertTrue(marker.exists())

    def test_missing_viewer_does_not_break_detection(self):
        self.assertEqual(module.run_command([sys.executable, '-c', 'pass'], viewer=['/nonexistent/rqt-viewer']), 0)

    def test_interrupt_stops_both_viewer_and_detector(self):
        child, viewer = mock.Mock(), mock.Mock()
        child.poll.side_effect = KeyboardInterrupt
        with mock.patch.object(module.subprocess, 'Popen', side_effect=[child, viewer]), \
                mock.patch.object(module, 'stop_process') as stop:
            with self.assertRaises(KeyboardInterrupt):
                module.run_command(['detector'], viewer=['viewer'])
            self.assertEqual(stop.call_args_list, [mock.call(viewer), mock.call(child)])


if __name__ == '__main__':
    unittest.main()
