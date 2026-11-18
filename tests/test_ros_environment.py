import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class RosEnvironmentTests(unittest.TestCase):
    def test_desktop_default_survives_ros_setup_injecting_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup=Path(tmp)/'setup.bash'
            setup.write_text('export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"\n')
            overlay=Path(tmp)/'overlay.bash'
            overlay.write_text('export TEST_ROBOT_OVERLAY_LOADED=yes\n')
            for supplied,expected in ((None,'1'),('0','0'),('1','1')):
                with self.subTest(supplied=supplied):
                    env=dict(os.environ,ROS_SETUP=str(setup),PERSON_ROBOT_SETUP=str(overlay))
                    env.pop('ROS_LOCALHOST_ONLY',None)
                    if supplied is not None:env['ROS_LOCALHOST_ONLY']=supplied
                    result=subprocess.run(['bash','-c','source "$1"; printf "%s %s" "$ROS_LOCALHOST_ONLY" "$TEST_ROBOT_OVERLAY_LOADED"',
                                           'env-test',str(ROOT/'scripts/ros_environment.sh')],env=env,text=True,capture_output=True,check=True)
                    self.assertEqual(result.stdout,expected+' yes')


if __name__=='__main__':unittest.main()
