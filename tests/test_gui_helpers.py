# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from faceset_curator.gui import FaceSetCuratorApp, friendly_failure
from faceset_curator.help_content import HELP_TOPICS, matching_help_topics


def test_duration_formatting():
    assert FaceSetCuratorApp._format_duration(42) == "42s"
    assert FaceSetCuratorApp._format_duration(125) == "2m 05s"
    assert FaceSetCuratorApp._format_duration(3725) == "1h 02m"


def test_friendly_cuda_failure_points_to_diagnostics():
    message = friendly_failure("CUDAExecutionProvider failed to load cudnn")
    assert "silently using the CPU" in message
    assert "diagnostics" in message.lower()


def test_friendly_download_failure_explains_resume():
    message = friendly_failure("ConnectionResetError during model download")
    assert "resume" in message


def test_help_search_finds_cuda_performance_guidance():
    matches = matching_help_topics("CUDA GPU")
    assert "performance" in matches
    assert "about" not in matches


def test_help_contains_documentation_and_about_content():
    assert "overview" in HELP_TOPICS
    assert "troubleshooting" in HELP_TOPICS
    assert HELP_TOPICS["about"][0] == "About FSC"
