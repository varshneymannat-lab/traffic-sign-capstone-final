from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import struct
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path


class RemoteZipFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        return self.position

    def read(self, length=-1):
        length = min(self.size - self.position, length if length >= 0 else self.size)
        if length <= 0:
            return b''
        if length > 64 * 1024 * 1024:
            raise ValueError('Refusing accidental full-archive download.')
        data = range_bytes(self.url, self.position, self.position + length - 1)
        self.position += len(data)
        return data


def range_bytes(url, start, end):
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise ValueError('Server did not honor the byte range; full download refused.')
                expected = f'bytes {start}-{end}/'
                if not response.headers.get('Content-Range', '').startswith(expected):
                    raise ValueError('Unexpected Content-Range.')
                data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise ValueError('Short or oversized range response.')
                return data
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def fetch_member(url, size, member, destination):
    if destination.exists():
        data = destination.read_bytes()
        if len(data) == member.file_size and zlib.crc32(data) == member.CRC:
            return {'image_key': destination.stem, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        raise ValueError('Existing image failed CRC: ' + str(destination))
    start = member.header_offset
    end = min(size - 1, start + 30 + len(member.filename.encode()) + 65535 + member.compress_size - 1)
    content = range_bytes(url, start, end)
    header = struct.unpack('<4s5H3I2H', content[:30])
    if header[0] != b'PK\x03\x04':
        raise ValueError('Invalid ZIP local header.')
    offset = 30 + header[-2] + header[-1]
    compressed = content[offset:offset + member.compress_size]
    if member.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    elif member.compress_type == zipfile.ZIP_STORED:
        data = compressed
    else:
        raise ValueError('Unsupported ZIP compression.')
    if len(data) != member.file_size or zlib.crc32(data) != member.CRC:
        raise ValueError('ZIP member CRC failed: ' + member.filename)
    temporary = destination.with_suffix('.jpg.partial')
    temporary.write_bytes(data)
    temporary.rename(destination)
    return {'image_key': destination.stem, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description='Download only selected authorised MTSD images using verified ZIP member ranges.')
    parser.add_argument('--links', type=Path, required=True, help='Private local JSON mapping archive names to authorised download URLs. Never commit it.')
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--train-per-class', type=int, default=500)
    parser.add_argument('--test-per-class', type=int, default=70)
    args = parser.parse_args()
    with args.candidates.open() as stream:
        rows = list(csv.DictReader(stream))
    selected = set()
    for split, limit in [('train', args.train_per_class), ('val', args.test_per_class)]:
        for category in sorted({r['sign_label'] for r in rows}):
            keys = sorted({r['image_key'] for r in rows if r['split'] == split and r['sign_label'] == category},
                          key=lambda key: hashlib.sha256(('42/' + key).encode()).hexdigest())
            selected.update(keys[:limit])
    links = json.loads(args.links.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    found = set()
    for archive_name, url in links.items():
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as response:
            size = int(response.headers['Content-Length'])
        reader = RemoteZipFile(url, size)
        with zipfile.ZipFile(reader) as archive:
            members = [m for m in archive.infolist() if m.filename.lower().endswith('.jpg') and Path(m.filename).stem in selected]
        print(f'{archive_name}: {len(members)} selected images; full archive {size / 1e9:.2f} GB', flush=True)
        def download(member):
            return fetch_member(url, size, member, args.output_dir / (Path(member.filename).stem + '.jpg'))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, record in enumerate(pool.map(download, members)):
                key = record['image_key']
                if key in found:
                    raise ValueError('Image key occurs in more than one archive: ' + key)
                found.add(key)
                records.append({**record, 'archive': archive_name})
                if (i + 1) % 100 == 0:
                    print(f'  Verified {i + 1}/{len(members)} selected images', flush=True)
        manifest = {'source': 'Authorised official MTSD v2 fully annotated image archives',
                    'selection_seed': 42, 'selection_scope': 'Official train and validation only; no hidden test labels used.',
                    'image_checks': 'Original ZIP member size and CRC32; per-image SHA256 recorded.',
                    'whole_archive_md5_verified': False,
                    'records': records, 'missing_keys': sorted(selected - found)}
        (args.output_dir / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
    if selected - found:
        raise ValueError(f'{len(selected-found)} selected images were not found in the supplied archives.')
    print(f'Complete: {len(found)} verified images; {sum(r["bytes"] for r in records)/1e9:.2f} GB', flush=True)


if __name__ == '__main__':
    main()
