import test from 'node:test';
import assert from 'node:assert/strict';
import {correctDetection, reviewCsv, validateSession, validateBox} from '../web/logic.mjs';

const original = () => ({id:'model-1',source:'model',predicted_class:'stop',confidence:.8,box:[1,2,20,22],review_class:'stop',review_box:[1,2,20,22],decision:'pending',note:''});
const session = () => ({schema_version:1,image:{name:'sample.jpg',width:100,height:80,sha256:'example',data_url:'data:image/jpeg;base64,AA=='},image_reviewed:false,detections:[original()]});

test('acceptance preserves the original prediction', () => {
  const row = original(), result = correctDetection(row,'stop',[1,2,20,22],100,80);
  assert.equal(result.decision,'accepted');
  assert.equal(row.decision,'pending');
  assert.deepEqual(result.box,row.box);
});
test('a label or box edit becomes a correction', () => {
  const result = correctDetection(original(),'no_entry',[2,2,20,22],100,80,'Corrected category');
  assert.equal(result.decision,'corrected');
  assert.equal(result.predicted_class,'stop');
  assert.equal(result.review_class,'no_entry');
});
test('manual additions have no fabricated model score', () => {
  const row = {...original(),source:'manual',box:null,predicted_class:null,confidence:null};
  assert.equal(correctDetection(row,'yield',[1,2,20,22],100,80).decision,'corrected');
});
test('invalid and out-of-bounds geometry is rejected', () => {
  for (const box of [[-1,0,20,20],[0,0,101,20],[5,5,3,9],[0,0,NaN,20]]) assert.throws(() => validateBox(box,100,80));
});
test('CSV retains rejected rows, raw predictions and corrections', () => {
  const value = session(); value.detections[0].decision = 'rejected'; value.detections[0].note = '=HYPERLINK("external")';
  const csv = reviewCsv(value);
  assert.match(csv,/original_xmin/); assert.match(csv,/review_xmin/); assert.match(csv,/"rejected"/);
  assert.ok(csv.includes("'=HYPERLINK"));
});
test('review sessions round-trip decisions', () => {
  const value = session(); value.detections[0].decision = 'accepted'; value.image_reviewed = true;
  assert.deepEqual(validateSession(JSON.parse(JSON.stringify(value))),value);
});
test('review import blocks remote images, duplicate IDs and unsupported categories', () => {
  const remote = session(); remote.image.data_url = 'https://external.example/image.jpg'; assert.throws(() => validateSession(remote));
  const duplicate = session(); duplicate.detections.push(original()); assert.throws(() => validateSession(duplicate));
  const invalid = session(); invalid.detections[0].review_class = 'traffic_light'; assert.throws(() => validateSession(invalid));
});
