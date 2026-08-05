#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# rospy and cv_bridge are provided by the system ROS installation, not by the
# uv-managed virtualenv, so they have to stay reachable on PYTHONPATH.
PYTHONPATH="${PYTHONPATH:-}:/usr/lib/python3/dist-packages"
export PYTHONPATH

exec uv run --frozen python ros_node.py
