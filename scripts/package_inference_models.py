from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


def main():
    project=Path(__file__).resolve().parents[1]
    files=['models/mtsd/detector.onnx','models/mtsd/detector.pt','models/mtsd/metadata.json',
           'models/mtsd_classifier/classifier.onnx','models/mtsd_classifier/metadata.json']
    manifest={name:{'sha256':hashlib.sha256((project/name).read_bytes()).hexdigest(),
                    'bytes':(project/name).stat().st_size} for name in files}
    output=project/'artifacts'
    output.mkdir(exist_ok=True)
    destination=output/'inference_models.zip'
    temporary=destination.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in files:
            archive.write(project/name,name)
        archive.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip():
            raise ValueError('Inference bundle failed ZIP integrity check.')
    if temporary.stat().st_size>=95*1024*1024:
        raise ValueError('Inference bundle is too large for the planned ordinary Git delivery; use release storage instead.')
    temporary.replace(destination)
    record={'archive':'artifacts/inference_models.zip','bytes':destination.stat().st_size,
            'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'files':manifest,
            'scope':'Selected inference models only; classifier training checkpoints and full data remain DVC-managed.'}
    (output/'manifest.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
