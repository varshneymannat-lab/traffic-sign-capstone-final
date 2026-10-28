import {CLASSES, LABELS, correctDetection, reviewCsv, validateSession} from './logic.js';

const $ = id => document.getElementById(id);
const canvas = $('scene'), ctx = canvas.getContext('2d');
const image = new Image();
let session = null, selectedId = null, imageData = '', imageName = '', scale = 1, drawing = false, drag = null, dirty = false, busy = false;
const colours = {pending:'#bd7a16', accepted:'#168057', corrected:'#257cbb', rejected:'#7d8782'};
CLASSES.forEach((value, i) => $('category').add(new Option(LABELS[i], value)));
const selected = () => session?.detections.find(row => row.id === selectedId);
const label = category => LABELS[CLASSES.indexOf(category)] || category;
function status(message, kind = '') { $('status').textContent = message; $('status').className = kind; }
function changed() { dirty = true; if (session) session.review_updated_at = new Date().toISOString(); }
function canReplace() { return !dirty || confirm('There are unexported review changes. Replace this image or review?'); }
function setBusy(value) { busy = value; document.body.classList.toggle('busy', value); $('run').disabled = value || !imageData; }
function point(event) { const rect = canvas.getBoundingClientRect(); return [Math.max(0, Math.min(image.width, (event.clientX - rect.left) * image.width / rect.width)), Math.max(0, Math.min(image.height, (event.clientY - rect.top) * image.height / rect.height))]; }
function setScale(value) { scale = Math.max(0.05, Math.min(5, value)); canvas.style.width = `${image.width * scale}px`; canvas.style.height = `${image.height * scale}px`; }
function fit() { if (image.width) setScale(Math.min(($('stage').clientWidth - 22) / image.width, ($('stage').clientHeight - 22) / image.height)); }
function draw() {
  if (!image.width) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(image, 0, 0);
  for (const row of session?.detections || []) {
    if (row.decision === 'rejected' && !$('show-rejected').checked) continue;
    const [x1, y1, x2, y2] = row.review_box;
    const size = Math.max(13, image.width / 80);
    ctx.strokeStyle = colours[row.decision]; ctx.lineWidth = row.id === selectedId ? Math.max(3, image.width / 280) : Math.max(2, image.width / 500);
    ctx.setLineDash(row.decision === 'pending' ? [7, 5] : []); ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); ctx.setLineDash([]);
    const text = `${label(row.review_class)}${row.source === 'manual' ? ' (manual)' : ''}`;
    ctx.font = `${size}px Arial`; const width = ctx.measureText(text).width + 10;
    const x = Math.min(x1, Math.max(0, image.width - width)), y = Math.max(size + 5, y1 - 3);
    ctx.fillStyle = colours[row.decision]; ctx.fillRect(x, y - size - 4, width, size + 7);
    ctx.fillStyle = '#fff'; ctx.fillText(text, x + 5, y - 1);
  }
  if (drag?.end) { const [x, y] = drag.start; ctx.strokeStyle = '#257cbb'; ctx.lineWidth = Math.max(2, image.width / 400); ctx.strokeRect(x, y, drag.end[0] - x, drag.end[1] - y); }
}
function detail() {
  const row = selected(); $('detail').hidden = !row;
  if (!row) return;
  $('detail-title').textContent = `${label(row.review_class)}`;
  $('original-label').textContent = row.source === 'manual' ? 'Manual addition. No model confidence.' : `Model: ${label(row.predicted_class)} / confidence ${row.confidence.toFixed(3)}`;
  $('category').value = row.review_class;
  ['xmin','ymin','xmax','ymax'].forEach((id, i) => { $(id).value = row.review_box[i]; $(id).max = i % 2 ? image.height : image.width; });
  $('note').value = row.note;
  const crop = $('crop').getContext('2d'), [x1,y1,x2,y2] = row.review_box;
  crop.clearRect(0,0,160,120); const ratio = Math.min(160 / (x2-x1),120 / (y2-y1));
  crop.drawImage(image,x1,y1,x2-x1,y2-y1,(160-(x2-x1)*ratio)/2,(120-(y2-y1)*ratio)/2,(x2-x1)*ratio,(y2-y1)*ratio);
}
function render() {
  const rows = session?.detections || [], counts = {pending:0,accepted:0,corrected:0,rejected:0}; rows.forEach(row => counts[row.decision]++);
  $('progress').textContent = `${rows.length - counts.pending} / ${rows.length} reviewed`;
  $('summary').textContent = `Pending ${counts.pending} / Accepted ${counts.accepted} / Corrected ${counts.corrected} / Rejected ${counts.rejected}`;
  $('detections').replaceChildren();
  const filtered = rows.filter(row => $('filter').value === 'all' || row.decision === $('filter').value);
  if (!filtered.length) { const p = document.createElement('p'); p.className = 'empty-list'; p.textContent = rows.length ? 'No matching detections' : 'No predictions'; $('detections').append(p); }
  filtered.forEach(row => {
    const button = document.createElement('button'); button.className = `detection-row${row.id === selectedId ? ' selected' : ''}`; button.dataset.id = row.id;
    const name = document.createElement('span'); name.textContent = label(row.review_class);
    const info = document.createElement('small'); info.textContent = row.source === 'manual' ? 'Manual addition' : `Confidence ${row.confidence.toFixed(3)}`; name.append(info);
    const state = document.createElement('span'); state.className = `state ${row.decision}`; state.textContent = row.decision;
    button.append(name, state); button.onclick = () => { selectedId = row.id; render(); }; $('detections').append(button);
  });
  $('export-json').disabled = !session; $('export-csv').disabled = !session; $('add').disabled = !session; $('image-reviewed').disabled = !session;
  $('image-reviewed').checked = session?.image_reviewed || false; detail(); draw();
}
async function loadImage(data, name, expected = null) {
  const candidate = new Image(); candidate.src = data; await candidate.decode();
  if (candidate.width * candidate.height > 20_000_000) throw new Error('Use an image with at most 20 million pixels.');
  if (expected && (candidate.width !== expected.width || candidate.height !== expected.height)) throw new Error('Embedded image dimensions do not match the review.');
  image.src = data; await image.decode(); imageData = data; imageName = name;
  canvas.width = image.width; canvas.height = image.height; canvas.style.display = 'block'; $('empty-image').hidden = true;
  $('image-name').textContent = name; $('image-meta').textContent = `${image.width} x ${image.height}`;
  $('run').disabled = false; $('timing').textContent = ''; drawing = false; $('stage').classList.remove('drawing'); $('add').classList.remove('primary'); fit(); draw();
}
async function request(url, body) {
  const response = await fetch(url, body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {});
  const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Request failed.'); return data;
}
function download(content, type, name) {
  const url = URL.createObjectURL(new Blob([content], {type})); const a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const protect = action => async (...args) => { try { await action(...args); } catch (error) { status(error.message, 'error'); } };
$('open-image').onclick = () => $('image-file').click();
$('image-file').onchange = protect(async event => {
  const file = event.target.files[0]; event.target.value = ''; if (!file || !canReplace()) return;
  if (file.size > 10 * 1024 * 1024 || !['image/jpeg','image/png'].includes(file.type)) throw new Error('Use a JPEG or PNG file smaller than 10 MB.');
  const data = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = reject; reader.readAsDataURL(file); });
  session = null; selectedId = null; dirty = false; await loadImage(data, file.name); render(); status('Image loaded');
});
$('sample').onclick = protect(async () => {
  if (!canReplace()) return; const sample = await request('/api/sample');
  session = null; selectedId = null; dirty = false; await loadImage(sample.image, sample.name); render(); status('Held-out MTSD sample loaded');
});
$('run').onclick = protect(async () => {
  if (!imageData || !canReplace()) return;
  setBusy(true); status('Running ONNX detector...', 'busy');
  try {
    const data = await request('/api/detect', {image:imageData, name:imageName});
    session = validateSession(data); selectedId = session.detections[0]?.id || null; dirty = false;
    await loadImage(session.image.data_url, session.image.name); render();
    $('timing').textContent = `${session.processing_seconds.toFixed(2)} s${session.cold_start ? ' including model loading' : ' prediction'}`;
    status(`${session.detections.length} predictions / ${session.model.inference_mode} inference`);
  } finally { setBusy(false); }
});
$('filter').onchange = render; $('show-rejected').onchange = draw;
$('correction').onsubmit = protect(event => {
  event.preventDefault(); const row = selected(); if (!row) return;
  const box = ['xmin','ymin','xmax','ymax'].map(id => Number($(id).value));
  Object.assign(row, correctDetection(row, $('category').value, box, image.width, image.height, $('note').value));
  changed(); render(); status(`Review ${row.decision}`);
});
$('reject').onclick = () => { const row = selected(); if (row) { row.decision = 'rejected'; row.note = $('note').value; changed(); render(); status('Prediction rejected'); } };
$('reset').onclick = () => { const row = selected(); if (!row) return; if (row.source === 'manual') { session.detections = session.detections.filter(r => r.id !== row.id); selectedId = null; } else { row.review_class = row.predicted_class; row.review_box = row.box.slice(); row.decision = 'pending'; row.note = ''; } changed(); render(); status('Review decision reset'); };
$('image-reviewed').onchange = () => { if (session) { session.image_reviewed = $('image-reviewed').checked; changed(); } };
$('add').onclick = () => { drawing = !drawing; $('stage').classList.toggle('drawing', drawing); $('add').classList.toggle('primary', drawing); status(drawing ? 'Add-sign mode' : 'Selection mode'); };
canvas.onpointerdown = event => {
  if (!session || busy) return; const p = point(event);
  if (drawing) { drag = {start:p, end:p}; canvas.setPointerCapture(event.pointerId); }
  else { selectedId = [...session.detections].reverse().find(row => (row.decision !== 'rejected' || $('show-rejected').checked) && p[0] >= row.review_box[0] && p[0] <= row.review_box[2] && p[1] >= row.review_box[1] && p[1] <= row.review_box[3])?.id || selectedId; render(); }
};
canvas.onpointermove = event => { if (drag) { drag.end = point(event); draw(); } };
canvas.onpointerup = event => {
  if (!drag) return; const end = point(event), start = drag.start; drag = null;
  const box = [Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])].map(v => Math.round(v * 1000) / 1000);
  if (box[2]-box[0] < 2 || box[3]-box[1] < 2) { draw(); status('Box is too small', 'error'); return; }
  selectedId = `manual-${crypto.randomUUID()}`;
  session.detections.push({id:selectedId, source:'manual', predicted_class:null, confidence:null, box:null, review_class:$('category').value, review_box:box, decision:'pending', note:''});
  drawing = false; $('stage').classList.remove('drawing'); $('add').classList.remove('primary'); changed(); render(); status('Manual box added / pending review');
};
canvas.onpointercancel = () => { drag = null; draw(); };
$('fit').onclick = fit; $('zoom-in').onclick = () => setScale(scale * 1.25); $('zoom-out').onclick = () => setScale(scale / 1.25);
$('export-json').onclick = () => { if (session) { download(JSON.stringify(session,null,2), 'application/json', 'traffic-sign-review.json'); dirty = false; status('Review export prepared'); } };
$('export-csv').onclick = () => { if (session) { download(reviewCsv(session), 'text/csv;charset=utf-8', 'traffic-sign-review.csv'); status('CSV export prepared'); } };
$('import-review').onclick = () => $('review-file').click();
$('review-file').onchange = protect(async event => {
  const file = event.target.files[0]; event.target.value = ''; if (!file || !canReplace()) return;
  if (file.size > 20 * 1024 * 1024) throw new Error('Review file must be smaller than 20 MB.');
  const data = validateSession(JSON.parse(await file.text())); await loadImage(data.image.data_url, data.image.name, data.image);
  session = data; selectedId = data.detections[0]?.id || null; dirty = false; render(); status('Review reopened');
});
window.addEventListener('beforeunload', event => { if (dirty) { event.preventDefault(); event.returnValue = ''; } });
new ResizeObserver(() => { if (!busy) fit(); }).observe($('stage'));
await $('sample').onclick();
