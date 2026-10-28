import base64
from io import BytesIO
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image
import pytest

from src.review_app import decode_image, make_handler


def image_payload():
    stream = BytesIO()
    Image.new('RGB', (40, 30), 'white').save(stream, format='PNG')
    return {'image':'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()}


def test_review_decodes_real_image_dimensions():
    assert decode_image(image_payload()).size == (40, 30)


@pytest.mark.parametrize('payload', [{}, {'image':'https://example.com/image.jpg'}, {'image':'data:image/png;base64,NOT_VALID!'}])
def test_review_rejects_invalid_image_payloads(payload):
    with pytest.raises(ValueError):
        decode_image(payload)


@pytest.fixture
def review_server():
    class Service:
        def detect(self, payload):
            decode_image(payload)
            return {'schema_version':1, 'detections':[]}
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(Service()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown()
    server.server_close()
    thread.join()


def test_review_serves_only_allowlisted_assets(review_server):
    with urlopen(review_server + '/') as response:
        assert b'Traffic Sign Review' in response.read()
        assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']
    with pytest.raises(HTTPError) as failure:
        urlopen(review_server + '/../../README.md')
    assert failure.value.code == 404


def test_review_http_image_request(review_server):
    request = Request(review_server + '/api/detect', data=json.dumps(image_payload()).encode(), headers={'Content-Type':'application/json'})
    with urlopen(request) as response:
        assert json.load(response)['detections'] == []


def test_review_rejects_cross_origin_request(review_server):
    request = Request(review_server + '/api/detect', data=b'{}', headers={'Content-Type':'application/json', 'Origin':'https://unrelated.example'})
    with pytest.raises(HTTPError) as failure:
        urlopen(request)
    assert failure.value.code == 403


def test_review_bad_json_has_structured_error(review_server):
    request = Request(review_server + '/api/detect', data=b'{broken', headers={'Content-Type':'application/json'})
    with pytest.raises(HTTPError) as failure:
        urlopen(request)
    assert failure.value.code == 400
    assert 'error' in json.load(failure.value)
