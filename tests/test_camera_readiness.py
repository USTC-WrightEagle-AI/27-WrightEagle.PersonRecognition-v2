import sys
from pathlib import Path
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from camera_session import RosCamera, CameraParametersNotReady, PARAMETERS


class CameraReadinessTests(unittest.TestCase):
    def camera(self):
        io = RosCamera.__new__(RosCamera)
        io.node = mock.sentinel.node
        io.rclpy = mock.Mock()
        io.get_client = mock.Mock()
        io.get_client.wait_for_service.return_value = True
        return io

    def test_waits_through_partial_parameters_after_service_appears(self):
        io = self.camera()
        expected = {key: key for key in PARAMETERS}
        io.get = mock.Mock(side_effect=[CameraParametersNotReady('incomplete'),
                                       CameraParametersNotReady('profile not declared'), expected])
        self.assertEqual(io.wait_parameters(timeout=1),expected)
        self.assertEqual(io.get.call_count,3)
        self.assertEqual(io.rclpy.spin_once.call_count,2)

    def test_driver_exit_is_reported_without_requesting_parameters(self):
        io = self.camera()
        io.get = mock.Mock()
        process = mock.Mock(returncode=1)
        process.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError,'driver exited'):
            io.wait_parameters(process=process)
        io.get.assert_not_called()

    def test_missing_parameters_timeout_has_a_specific_diagnostic(self):
        io = self.camera()
        io.get = mock.Mock(side_effect=CameraParametersNotReady('profile not declared'))
        with mock.patch('camera_session.time.monotonic',side_effect=[0,0,2]):
            with self.assertRaisesRegex(RuntimeError,'profile not declared'):
                io.wait_parameters(timeout=1)

    def test_actual_errors_are_not_silently_retried(self):
        io = self.camera()
        io.get = mock.Mock(side_effect=RuntimeError('Unsupported parameter type'))
        with self.assertRaisesRegex(RuntimeError,'Unsupported parameter type'):
            io.wait_parameters()
        self.assertEqual(io.get.call_count,1)


if __name__=='__main__':unittest.main()
