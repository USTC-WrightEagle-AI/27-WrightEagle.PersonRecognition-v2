import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('camera_session', ROOT / 'scripts/camera_session.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DEFAULT = {'serial_no': '260722304986', 'rgb_camera.color_profile': '640,480,15',
           'depth_module.depth_profile': '640,480,15', 'enable_color': True,
           'enable_depth': True, 'align_depth.enable': False}
TARGET = '1280,720,30'


class FakeCamera:
    def __init__(self):
        self.values = DEFAULT.copy()
        self.writes = []
        self.fail = None
        self.live_error = None
        self.found = True
        self.get_client = mock.Mock()
        self.get_client.wait_for_service.return_value = True

    def get(self, keys):
        return {key: self.values[key] for key in keys}

    def wait_parameters(self, **kwargs):
        return self.values.copy()

    def set(self, values):
        for key in ('rgb_camera.color_profile', 'depth_module.depth_profile'):
            if key in values:
                enable = 'enable_color' if key.startswith('rgb_') else 'enable_depth'
                assert self.values[enable] is False, 'profile changed without disabling stream'
        self.writes.append(values.copy())
        if self.fail and self.fail(values):
            self.fail = None
            raise RuntimeError('injected write failure')
        self.values.update(values)

    def discover(self):
        return self.found

    def wait_live(self, *args):
        if self.live_error:
            raise RuntimeError(self.live_error)

    def wait_restored(self, values):
        assert values == self.values


class CameraSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.journal = Path(self.temp.name) / 'restore.json'
        self.io = FakeCamera()
        self.lease = module.ProfileLease(self.io, self.journal)

    def test_switch_and_exact_restore(self):
        self.lease.prepare(DEFAULT['serial_no'], TARGET, True)
        self.assertEqual(json.loads(self.journal.read_text())['before'], DEFAULT)
        self.assertEqual(self.io.values['rgb_camera.color_profile'], TARGET)
        self.assertEqual(self.io.values['depth_module.depth_profile'], TARGET)
        self.assertTrue(self.io.values['align_depth.enable'])
        self.lease.restore()
        self.assertEqual(self.io.values, DEFAULT)
        self.assertFalse(self.journal.exists())

    def test_already_matching_camera_has_no_writes(self):
        self.io.values.update({'rgb_camera.color_profile': TARGET, 'depth_module.depth_profile': TARGET,
                               'align_depth.enable': True})
        self.lease.prepare(DEFAULT['serial_no'], TARGET, True)
        self.lease.restore()
        self.assertEqual(self.io.writes, [])

    def test_rgb_only_does_not_change_depth_or_alignment(self):
        self.lease.prepare(DEFAULT['serial_no'], TARGET, False)
        for key in ('depth_module.depth_profile', 'enable_depth', 'align_depth.enable'):
            self.assertEqual(self.io.values[key], DEFAULT[key])
        self.lease.restore()
        self.assertEqual(self.io.values, DEFAULT)

    def test_wrong_serial_does_not_modify_camera(self):
        with self.assertRaisesRegex(RuntimeError, 'serial differs'):
            self.lease.prepare('another-camera', TARGET, True)
        self.assertEqual(self.io.writes, [])
        self.assertFalse(self.journal.exists())

    def test_rejected_profile_restores_partially_disabled_streams(self):
        self.io.fail = lambda values: values == {'depth_module.depth_profile': TARGET}
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            module.supervise(self.io, self.journal, DEFAULT['serial_no'], TARGET, True, ['unused'])
        self.assertEqual(self.io.values, DEFAULT)
        self.assertFalse(self.journal.exists())

    def test_live_validation_failure_prevents_application_and_restores(self):
        self.io.live_error = 'No depth frames'
        with mock.patch.object(module, 'run_command') as run:
            with self.assertRaisesRegex(RuntimeError, 'No depth'):
                module.supervise(self.io, self.journal, DEFAULT['serial_no'], TARGET, True, ['unused'])
            run.assert_not_called()
        self.assertEqual(self.io.values, DEFAULT)

    def test_restore_after_application_failure_and_interrupt(self):
        for outcome in (17, KeyboardInterrupt()):
            with self.subTest(outcome=outcome):
                with mock.patch.object(module, 'run_command', side_effect=(outcome if isinstance(outcome, BaseException) else None), return_value=outcome):
                    if isinstance(outcome, BaseException):
                        with self.assertRaises(KeyboardInterrupt):
                            module.supervise(self.io, self.journal, DEFAULT['serial_no'], TARGET, True, ['unused'])
                    else:
                        result = module.supervise(self.io, self.journal, DEFAULT['serial_no'], TARGET, True, ['unused'])
                        self.assertEqual(result, outcome)
                self.assertEqual(self.io.values, DEFAULT)
                self.assertFalse(self.journal.exists())

    def test_failed_restore_keeps_recovery_snapshot(self):
        self.lease.prepare(DEFAULT['serial_no'], TARGET, True)
        self.io.fail = lambda values: values == {'rgb_camera.color_profile': '640,480,15'}
        with self.assertRaises(RuntimeError):
            self.lease.restore()
        self.assertTrue(self.journal.exists())
        module.supervise(self.io, self.journal, '', '', False, [], restore_only=True)
        self.assertEqual(self.io.values, DEFAULT)
        self.assertFalse(self.journal.exists())

    def test_existing_recovery_record_blocks_new_session(self):
        self.journal.write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'restore_camera'):
            module.supervise(self.io, self.journal, DEFAULT['serial_no'], TARGET, True, ['unused'])
        self.assertEqual(self.io.writes, [])

    def test_originally_disabled_streams_are_restored_disabled(self):
        self.io.values['enable_color'] = False
        self.io.values['enable_depth'] = False
        original = self.io.values.copy()
        self.lease.prepare(DEFAULT['serial_no'], TARGET, True)
        self.lease.restore()
        self.assertEqual(self.io.values, original)

    def test_temporary_camera_is_closed_instead_of_restoring_default_node(self):
        self.io.found = False
        with mock.patch.object(module, 'start_camera', return_value=mock.sentinel.camera), \
                mock.patch.object(module, 'stop_process') as stop, \
                mock.patch.object(module, 'run_command', return_value=0):
            module.supervise(self.io, self.journal, DEFAULT['serial_no'], TARGET, True, ['unused'])
            stop.assert_called_once_with(mock.sentinel.camera)
        self.assertFalse(self.journal.exists())

    def test_head_sdk_and_other_camera_skip_automatic_ros_changes(self):
        self.assertIsNone(module.settings('enroll', ['--head-backend', 'sdk']))
        self.assertIsNone(module.settings('enroll', ['--camera=0', '--no-depth-filter']))
        self.assertFalse(module.settings('enroll', ['--no-depth-filter'])[2])
        self.assertTrue(module.settings('enroll', ['--no-depth-filter', '--depth-filter'])[2])

    def test_config_path_survives_detector_changing_working_directory(self):
        path = str(Path('my-camera.yaml').resolve())
        for args in (['--config', 'my-camera.yaml'], ['--config=my-camera.yaml']):
            result = module.absolute_config_arguments(['bash', 'start_3d.sh', *args])
            self.assertIn(path, result[-1])

    def test_shell_name_quoting_and_help_without_camera_or_enrollment(self):
        tmp = Path(self.temp.name)
        setup = tmp / 'setup.bash'
        setup.write_text('')
        output = tmp / 'arguments.json'
        fake = tmp / 'fake-python'
        fake.write_text('#!/usr/bin/python3\nimport json,os,sys\nopen(os.environ["TEST_ARGS"],"w").write(json.dumps(sys.argv[1:]))\n')
        fake.chmod(0o755)
        env = dict(os.environ, FACE_ROS_SETUP=str(setup), FACE_PYTHON=str(fake),
                   PERSON_CAMERA_AUTO='0', TEST_ARGS=str(output))
        script = str(ROOT / 'scripts/enroll_face.sh')
        result = subprocess.run(['bash', script, '--no-depth-filter'], input='  TEST ONLY  \n',
                                cwd='/tmp', text=True, capture_output=True, env=env, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(output.read_text())
        self.assertEqual(args[-2:], ['--', 'TEST ONLY'])
        self.assertIn('--no-depth-filter', args)
        result = subprocess.run(['bash', script, '--help'], input='', text=True,
                                capture_output=True, env=env, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text())[-1], '--help')
        self.assertNotIn('即将录入', result.stdout)


if __name__ == '__main__':
    unittest.main()
