"""Local-only image detection service for the human-review workspace."""

from __future__ import annotations

import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
MAX_REQUEST_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
CLASSES = ['stop', 'no_entry', 'yield', 'speed_limit_30', 'speed_limit_50']


def decode_image(payload):
    encoded = payload.get('image')
    if not isinstance(encoded, str) or not encoded.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,')):
        raise ValueError('Use a JPEG or PNG image.')
    try:
        raw = base64.b64decode(encoded.split(',', 1)[1], validate=True)
        with Image.open(BytesIO(raw)) as original:
            if original.format not in {'JPEG', 'PNG'} or original.width * original.height > MAX_IMAGE_PIXELS:
                raise ValueError('Use a JPEG or PNG with at most 20 million pixels.')
            image = ImageOps.exif_transpose(original).convert('RGB')
    except (UnidentifiedImageError, OSError, base64.binascii.Error, Image.DecompressionBombError) as exc:
        raise ValueError('The image could not be decoded.') from exc
    return image


class ReviewService:
    def __init__(self, model):
        self.model_path = Path(model)
        self.metadata = json.loads((self.model_path.parent / 'metadata.json').read_text())
        self.detector = None
        self.lock = threading.Lock()

    def detect(self, payload):
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO
        from src.utils.scene_detector import SceneDetector

        image = decode_image(payload)
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        displayed = buffer.getvalue()
        frame = cv2.imdecode(np.frombuffer(displayed, dtype=np.uint8), cv2.IMREAD_COLOR)
        with self.lock:
            started = time.perf_counter()
            cold = self.detector is None
            if cold:
                if not self.model_path.is_file():
                    raise ValueError('Restore models first: python scripts/restore_inference_models.py')
                expected = self.metadata.get('sha256')
                if expected and hashlib.sha256(self.model_path.read_bytes()).hexdigest() != expected:
                    raise ValueError('Detector checksum differs from its metadata.')
                torch.set_num_threads(2)
                model = YOLO(str(self.model_path), task='detect')
                if [model.names[i] for i in range(len(model.names))] != CLASSES:
                    raise ValueError('The model must use the recorded five-category label order.')
                self.detector = SceneDetector(model, self.metadata['image_size'], self.metadata.get('inference_mode', 'sliced'))
            result = self.detector.predict(frame, imgsz=self.metadata['image_size'],
                conf=self.metadata['confidence'], iou=0.7, max_det=100,
                device='cpu', rect=False, verbose=False)[0]
            elapsed = time.perf_counter() - started
        records = []
        for index, box in enumerate(result.boxes):
            category = CLASSES[int(box.cls.item())]
            coordinates = [round(float(v), 3) for v in box.xyxy[0].tolist()]
            records.append({'id':f'model-{index + 1}', 'source':'model',
                'predicted_class':category, 'confidence':float(box.conf.item()),
                'box':coordinates, 'review_class':category, 'review_box':coordinates[:],
                'decision':'pending', 'note':''})
        return {'schema_version':1,
            'image':{'name':str(payload.get('name', 'image.jpg'))[:200],
                     'width':image.width, 'height':image.height,
                     'sha256':hashlib.sha256(displayed).hexdigest(),
                     'data_url':'data:image/jpeg;base64,' + base64.b64encode(displayed).decode('ascii')},
            'model':{'architecture':self.metadata['architecture'], 'format':'ONNX' if self.model_path.suffix=='.onnx' else 'PyTorch',
                     'sha256':self.metadata.get('sha256'), 'confidence':self.metadata['confidence'],
                     'inference_mode':self.metadata.get('inference_mode', 'sliced')},
            'detections':records, 'image_reviewed':False,
            'processing_seconds':elapsed, 'cold_start':cold,
            'timing_scope':'Detector construction on cold start plus prediction. Excludes upload, image decoding, response encoding and human review. Not a controlled benchmark.'}


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type='application/json; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlsplit(self.path).path
            assets = {'/':('index.html','text/html'), '/app.js':('app.js','text/javascript'),
                      '/logic.js':('logic.mjs','text/javascript'), '/style.css':('style.css','text/css'),
                      '/icons.svg':('icons.svg','image/svg+xml')}
            if route == '/api/health':
                self.reply(200, json.dumps({'status':'ok', 'classes':CLASSES}).encode())
            elif route == '/api/sample':
                import cv2
                capture = cv2.VideoCapture(str(ROOT / 'demo/mtsd_gallery/input.mp4'))
                capture.set(cv2.CAP_PROP_POS_FRAMES, 20)
                ok, frame = capture.read()
                capture.release()
                if not ok:
                    self.reply(500, b'{"error":"Sample image is unavailable."}')
                    return
                ok, encoded = cv2.imencode('.jpg', frame)
                if not ok:
                    self.reply(500, b'{"error":"Sample image encoding failed."}')
                    return
                self.reply(200, json.dumps({'name':'MTSD-held-out-road-scene.jpg',
                    'image':'data:image/jpeg;base64,'+base64.b64encode(encoded).decode()}).encode())
            elif route in assets:
                filename, kind = assets[route]
                self.reply(200, (ROOT / 'web' / filename).read_bytes(), kind)
            else:
                self.reply(404, b'{"error":"Not found."}')

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                self.reply(403, b'{"error":"Only same-origin requests are allowed."}')
                return
            if self.path != '/api/detect':
                self.reply(404, b'{"error":"Not found."}')
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    self.reply(413, b'{"error":"Request must be smaller than 15 MB."}')
                    return
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    self.reply(415, b'{"error":"JSON request required."}')
                    return
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('Expected an image request object.')
                result = service.detect(payload)
                self.reply(200, json.dumps(result, allow_nan=False).encode())
            except (ValueError, TypeError) as exc:
                self.reply(400, json.dumps({'error':str(exc)}).encode())
            except Exception as exc:
                self.log_error('Detection failed: %s', repr(exc))
                self.reply(500, b'{"error":"Detection failed. Check the local terminal and model files."}')
    return Handler


def main():
    parser = argparse.ArgumentParser(description='Run the local human-review workspace.')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/mtsd/detector.onnx')
    args = parser.parse_args()
    if args.model.suffix != '.onnx' or not (args.model.parent / 'metadata.json').is_file():
        parser.error('Restore the ONNX model first: python scripts/restore_inference_models.py')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(ReviewService(args.model)))
    print(f'Traffic Sign Review: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
