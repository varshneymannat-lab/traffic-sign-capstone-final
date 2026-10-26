from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    project=Path(__file__).resolve().parents[1]
    record=json.loads((project/'pipelines/detection/pretrained.json').read_text())
    destination=project/record['filename']
    if not destination.exists():
        partial=destination.with_suffix('.pt.partial')
        with urllib.request.urlopen(record['url'],timeout=120) as source,partial.open('wb') as output:
            for block in iter(lambda:source.read(1024*1024),b''):
                output.write(block)
        if hashlib.sha256(partial.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Downloaded pretrained detector does not match the recorded SHA256.')
        partial.rename(destination)
    if destination.stat().st_size!=record['bytes'] or hashlib.sha256(destination.read_bytes()).hexdigest()!=record['sha256']:
        raise ValueError('Existing pretrained detector differs from the recorded initial checkpoint.')
    print('Verified official pretrained YOLO11n SHA256.')


if __name__=='__main__':
    main()
