export const CLASSES = ['stop', 'no_entry', 'yield', 'speed_limit_30', 'speed_limit_50'];
export const LABELS = ['Stop', 'No entry', 'Yield', 'Speed limit 30', 'Speed limit 50'];
const DECISIONS = ['pending', 'accepted', 'corrected', 'rejected'];

export function validateBox(box, width, height) {
  if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)) throw new Error('Box coordinates must be four numbers.');
  const [x1, y1, x2, y2] = box;
  if (x1 < 0 || y1 < 0 || x2 > width || y2 > height || x2 <= x1 || y2 <= y1) throw new Error('The box must fit inside the image and have positive width and height.');
  return box.map(v => Math.round(v * 1000) / 1000);
}

export function correctDetection(row, category, box, width, height, note = '') {
  if (!CLASSES.includes(category)) throw new Error('Unsupported sign category.');
  const checked = validateBox(box, width, height);
  const changed = row.source === 'manual' || category !== row.predicted_class || checked.some((v, i) => Math.abs(v - row.box[i]) > 0.001);
  return {...row, review_class:category, review_box:checked, decision:changed ? 'corrected' : 'accepted', note:String(note).slice(0, 2000)};
}

function csvCell(value) {
  if (value === null || value === undefined) return '';
  let text = String(value);
  if (typeof value === 'string' && /^[\s]*[=+@-]/.test(text)) text = "'" + text;
  return '"' + text.replaceAll('"', '""') + '"';
}

export function reviewCsv(session) {
  const header = ['image', 'image_sha256', 'image_reviewed', 'id', 'source', 'decision', 'predicted_class', 'detector_confidence',
    'original_xmin', 'original_ymin', 'original_xmax', 'original_ymax', 'review_class', 'review_xmin', 'review_ymin', 'review_xmax', 'review_ymax', 'note'];
  const rows = session.detections.map(row => [session.image.name, session.image.sha256, session.image_reviewed, row.id, row.source,
    row.decision, row.predicted_class, row.confidence, ...(row.box || ['', '', '', '']), row.review_class, ...row.review_box, row.note]);
  return [header, ...rows].map(row => row.map(csvCell).join(',')).join('\r\n');
}

export function validateSession(value) {
  if (!value || value.schema_version !== 1 || !value.image || !Array.isArray(value.detections) || value.detections.length > 1000) throw new Error('Unsupported review file.');
  const {width, height, data_url} = value.image;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0 || width * height > 20_000_000 ||
      typeof data_url !== 'string' || !/^data:image\/(jpeg|png);base64,[A-Za-z0-9+/=]+$/.test(data_url)) throw new Error('Invalid embedded image.');
  const ids = new Set();
  const detections = value.detections.map(row => {
    if (!row || typeof row.id !== 'string' || ids.has(row.id) || !['model', 'manual'].includes(row.source) || !DECISIONS.includes(row.decision) || !CLASSES.includes(row.review_class)) throw new Error('Invalid review entry.');
    ids.add(row.id);
    if (row.source === 'model' && (!CLASSES.includes(row.predicted_class) || !Number.isFinite(row.confidence) || row.confidence < 0 || row.confidence > 1)) throw new Error('Invalid original prediction.');
    return {...row, box:row.source === 'model' ? validateBox(row.box, width, height) : null,
      confidence:row.source === 'model' ? row.confidence : null,
      predicted_class:row.source === 'model' ? row.predicted_class : null,
      review_box:validateBox(row.review_box, width, height), note:String(row.note || '').slice(0, 2000)};
  });
  return {...value, image:{...value.image, name:String(value.image.name || 'image.jpg').slice(0, 200)}, image_reviewed:value.image_reviewed === true, detections};
}
