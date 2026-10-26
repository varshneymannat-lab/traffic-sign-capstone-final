from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts.restore_inference_models import restore


def main():
    restore(PROJECT/'artifacts/inference_models.zip',PROJECT)
    with tempfile.TemporaryDirectory(prefix='traffic-sign-smoke-') as directory:
        subprocess.run([sys.executable,'-m','src.detect_signs',
                        'demo/mtsd_gallery/input.mp4','--output-dir',directory,'--max-frames','2'],
                       cwd=PROJECT,check=True)
        summary=json.loads((Path(directory)/'summary.json').read_text())
        if summary['frames']!=2 or not (Path(directory)/'annotated.mp4').is_file():
            raise RuntimeError('Container video smoke test did not produce the expected output.')
    print('Verified packaged ONNX inference on two actual video frames.')


if __name__=='__main__':
    main()
