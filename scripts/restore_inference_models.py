from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def restore(bundle,destination):
    with zipfile.ZipFile(bundle) as archive:
        manifest=json.loads(archive.read('MANIFEST.json'))
        if set(archive.namelist())!=set(manifest)|{'MANIFEST.json'}:
            raise ValueError('Bundle files do not match its manifest.')
        payloads={}
        for name,entry in manifest.items():
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts or not name.startswith('models/'):
                raise ValueError('Unsafe bundle member path.')
            target=destination/relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Bundle member would escape the destination.')
            data=archive.read(name)
            if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
                raise ValueError('Bundle file integrity mismatch: '+name)
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=entry['sha256']:
                raise ValueError('Existing different model would be overwritten: '+str(target))
            payloads[target]=data
    for target,data in payloads.items():
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            temporary=target.with_suffix(target.suffix+'.partial')
            temporary.write_bytes(data)
            temporary.replace(target)
    return len(payloads)


def main():
    project=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description='Restore the included verified inference models; no training-data download required.')
    parser.add_argument('--bundle',type=Path,default=project/'artifacts/inference_models.zip')
    parser.add_argument('--destination',type=Path,default=project)
    args=parser.parse_args()
    print(f'Restored or verified {restore(args.bundle,args.destination)} model files.')


if __name__=='__main__':
    main()
