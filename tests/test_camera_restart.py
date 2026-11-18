import json
from contextlib import ExitStack
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import camera_process
import camera_session
from test_camera_session import FakeCamera, DEFAULT, TARGET


class CameraRestartTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runtime = Path(self.tmp.name)
        self.journal = self.runtime / 'restore.json'
        self.io = FakeCamera()
        self.io.wait_node_gone = mock.Mock()
        self.record = {'camera': {'pid': 123}, 'guardian': None,
                       'owner': {'pid': 122}, 'command': [], 'before': DEFAULT.copy()}
        self.snapshot = self.stack.enter_context(mock.patch.object(camera_process, 'snapshot', return_value=self.record))
        self.stop = self.stack.enter_context(mock.patch.object(camera_process, 'stop_record'))
        self.start = self.stack.enter_context(mock.patch.object(camera_process, 'start_guardian', side_effect=self.start_camera))
        self.stack.enter_context(mock.patch.object(camera_process, 'alive', return_value=True))

    def start_camera(self, runtime, original, values):
        self.io.values.update(values)
        state = {'camera': {'pid': 333}, 'guardian': {'pid': 332}}
        camera_process.write_json(Path(runtime) / 'managed-camera.json', state)
        return state

    def test_restart_and_restore_without_dynamic_parameter_writes(self):
        with mock.patch.object(camera_session, 'run_command', return_value=0):
            result = camera_session.supervise(self.io, self.journal, DEFAULT['serial_no'], TARGET, True,
                                              ['unused'], lease_factory=camera_session.RestartLease)
        self.assertEqual(result, 0)
        self.assertEqual(self.io.values, DEFAULT)
        self.assertEqual(self.io.writes, [])
        self.assertEqual(self.start.call_count, 2)
        self.assertFalse(self.journal.exists())

    def test_real_supervisor_path_restores_on_interrupt(self):
        with mock.patch.object(camera_session, 'run_command', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                camera_session.supervise(self.io, self.journal, DEFAULT['serial_no'], TARGET, True,
                                          ['unused'], lease_factory=camera_session.RestartLease)
        self.assertEqual(self.io.values, DEFAULT)
        self.assertFalse(self.journal.exists())

    def test_saved_restart_record_can_be_recovered(self):
        lease = camera_session.RestartLease(self.io, self.journal)
        lease.prepare(DEFAULT['serial_no'], TARGET, True)
        saved = json.loads(self.journal.read_text())
        self.assertEqual(saved['restart']['before'], DEFAULT)
        camera_session.supervise(self.io, self.journal, '', '', False, [], restore_only=True)
        self.assertEqual(self.io.values, DEFAULT)
        self.assertFalse(self.journal.exists())

    def test_original_launch_exit_does_not_resurrect_camera(self):
        lease = camera_session.RestartLease(self.io, self.journal)
        lease.prepare(DEFAULT['serial_no'], TARGET, True)
        self.start.reset_mock()
        with mock.patch.object(camera_process, 'alive', return_value=False):
            lease.restore()
        self.start.assert_not_called()
        self.assertFalse(self.journal.exists())

    def test_unrecognized_process_is_not_stopped(self):
        self.snapshot.side_effect = RuntimeError('unrecognized parent')
        lease = camera_session.RestartLease(self.io, self.journal)
        with self.assertRaisesRegex(RuntimeError, 'unrecognized'):
            lease.prepare(DEFAULT['serial_no'], TARGET, True)
        self.stop.assert_not_called()
        self.start.assert_not_called()
        self.assertFalse(self.journal.exists())


class GuardianLifetimeTests(unittest.TestCase):
    def test_camera_stops_when_original_owner_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
            state = None
            try:
                original = {'owner': camera_process.identity(owner.pid),
                            'command': [sys.executable, '-c', 'import time; time.sleep(60)'],
                            'cwd': tmp, 'environment': dict(os.environ)}
                state = camera_process.start_guardian(tmp, original, {})
                self.assertTrue(camera_process.alive(state['camera']))
                owner.terminate()
                owner.wait(timeout=3)
                deadline = time.monotonic() + 8
                while camera_process.alive(state['camera']) and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertFalse(camera_process.alive(state['camera']))
            finally:
                if state:
                    camera_process.stop_record(state)
                if owner.poll() is None:
                    owner.kill()
                    owner.wait(timeout=3)


if __name__ == '__main__':
    unittest.main()
