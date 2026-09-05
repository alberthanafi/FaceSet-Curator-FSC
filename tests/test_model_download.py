# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

import hashlib
import io
import tempfile
from pathlib import Path

from faceset_curator.model_download import _download_once, sha256_file


class FakeResponse:
    status = 206

    def __init__(self, payload: bytes):
        self.stream = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size=-1):
        return self.stream.read(size)


def test_model_download_resumes_partial_file():
    with tempfile.TemporaryDirectory(prefix="fsc-download-test-") as temporary:
        destination = Path(temporary) / "model.zip.part"
        destination.write_bytes(b"first-")
        requests = []

        def opener(request, timeout):
            requests.append((request, timeout))
            return FakeResponse(b"second")

        _download_once("https://example.invalid/model.zip", destination, opener=opener)
        assert destination.read_bytes() == b"first-second"
        assert requests[0][0].get_header("Range") == "bytes=6-"
        assert requests[0][1] == 30


def test_sha256_file_streams_expected_digest():
    with tempfile.TemporaryDirectory(prefix="fsc-hash-test-") as temporary:
        path = Path(temporary) / "sample.bin"
        path.write_bytes(b"verified model")
        assert sha256_file(path, chunk_size=3) == hashlib.sha256(b"verified model").hexdigest()
