import hashlib
import json
import zipfile

import pytest

from scripts.restore_inference_models import restore


def make_bundle(tmp_path,name='models/mtsd/model.onnx',content=b'model',digest=None):
    path=tmp_path/'bundle.zip'
    manifest={name:{'bytes':len(content),'sha256':digest or hashlib.sha256(content).hexdigest()}}
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr(name,content)
        archive.writestr('MANIFEST.json',json.dumps(manifest))
    return path


def test_bundle_restore_checks_hash_and_is_repeatable(tmp_path):
    bundle=make_bundle(tmp_path)
    root=tmp_path/'output'
    assert restore(bundle,root)==1
    assert restore(bundle,root)==1
    assert (root/'models/mtsd/model.onnx').read_bytes()==b'model'


def test_unsafe_bundle_path_is_refused(tmp_path):
    with pytest.raises(ValueError,match='Unsafe'):
        restore(make_bundle(tmp_path,name='models/../../escape'),tmp_path/'output')


def test_corrupted_bundle_is_refused_before_writes(tmp_path):
    with pytest.raises(ValueError,match='integrity'):
        restore(make_bundle(tmp_path,digest='wrong'),tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_existing_different_model_is_preserved(tmp_path):
    root=tmp_path/'output'
    target=root/'models/mtsd/model.onnx'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'existing')
    with pytest.raises(ValueError,match='overwritten'):
        restore(make_bundle(tmp_path),root)
    assert target.read_bytes()==b'existing'
