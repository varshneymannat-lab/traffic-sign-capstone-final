import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');

const root=path.resolve(process.env.REVIEW_TEST_OUTPUT || 'outputs/review_browser');
await fs.mkdir(root,{recursive:true});
const browser=await chromium.launch({...(process.env.REVIEW_BROWSER_EXECUTABLE ? {executablePath:process.env.REVIEW_BROWSER_EXECUTABLE} : {}),headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
const errors=[]; page.on('pageerror',e=>errors.push(e.message));
await page.goto(process.env.REVIEW_URL || 'http://127.0.0.1:8765');
await page.waitForFunction(()=>document.querySelector('#scene').width>300 && !document.querySelector('#run').disabled);
await page.screenshot({path:path.join(root,'desktop_loaded.png'),fullPage:true});
await page.click('#run');
await page.waitForFunction(()=>!document.body.classList.contains('busy') && /predictions/.test(document.querySelector('#status').textContent),{timeout:120000});
assert.equal(await page.locator('#status.error').count(),0);
const predictions=await page.locator('.detection-row').count();
assert.ok(predictions>0,'The recorded sample should produce an actual model prediction.');
await page.screenshot({path:path.join(root,'desktop_inference.png'),fullPage:true});
if(predictions){
  await page.click('#accept'); assert.match(await page.locator('#summary').textContent(),/Accepted 1/);
  await page.click('#reset');
  await page.selectOption('#category','yield'); await page.click('#accept');
  assert.match(await page.locator('#summary').textContent(),/Corrected 1/);
  await page.click('#reject'); assert.match(await page.locator('#summary').textContent(),/Rejected 1/);
  await page.click('#show-rejected');
  await page.click('#reset'); assert.match(await page.locator('#summary').textContent(),/Corrected 0/);
}
await page.click('#add');
const frame=await page.locator('#scene').boundingBox();
await page.mouse.move(frame.x+frame.width*.4,frame.y+frame.height*.3);
await page.mouse.down();
await page.mouse.move(frame.x+frame.width*.5,frame.y+frame.height*.5);
await page.mouse.up();
await page.selectOption('#category','stop'); await page.click('#accept');
assert.match(await page.locator('#original-label').textContent(),/No model confidence/);
assert.match(await page.locator('#summary').textContent(),/Corrected 1/);
await page.check('#image-reviewed');
await page.fill('#note','Browser smoke-test correction, not ground truth.'); await page.click('#accept');
await page.screenshot({path:path.join(root,'desktop_review.png'),fullPage:true});
const csvDownload=page.waitForEvent('download'); await page.click('#export-csv'); const csv=await csvDownload; await csv.saveAs(path.join(root,'smoke_review.csv'));
const jsonDownload=page.waitForEvent('download'); await page.click('#export-json'); const json=await jsonDownload; await json.saveAs(path.join(root,'smoke_review.json'));
const record=JSON.parse(await fs.readFile(path.join(root,'smoke_review.json'),'utf8'));
assert.equal(record.image_reviewed,true); assert.equal(record.detections.filter(d=>d.source==='manual'&&d.confidence===null).length,1);
await page.setInputFiles('#review-file',path.join(root,'smoke_review.json'));
await page.waitForFunction(()=>document.querySelector('#status').textContent==='Review reopened');
await page.setViewportSize({width:390,height:844}); await page.waitForTimeout(300);
assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
await page.screenshot({path:path.join(root,'mobile_review.png'),fullPage:true});
assert.deepEqual(errors,[]);
await fs.writeFile(path.join(root,'verification.json'),JSON.stringify({checked_at:new Date().toISOString(),real_model_predictions:predictions,
  actual_onnx_inference:true,checks:['Desktop image load','Actual ONNX inference','Correct/reject/reset','Add missed box','Manual score remains null','CSV and JSON export','JSON review reopen','Mobile layout without horizontal overflow','No page errors'],
  note:'Functional browser smoke test. The scripted manual box and label changes are test actions, not verified ground truth or a user-efficiency study.'},null,2));
console.log(JSON.stringify({real_model_predictions:predictions,checks_passed:9,output:root}));
await browser.close();
