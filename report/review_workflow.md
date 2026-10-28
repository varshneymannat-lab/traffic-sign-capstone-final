# Human Review Workflow

The local workspace supports review of road-survey images. It runs the existing five-category ONNX detector with the validation-selected tiled-inference settings. It does not retrain the detector or change the published test results.

## Start

From the project root, with the Python environment activated:

```bash
python scripts/restore_inference_models.py
python -m src.review_app --port 8765
```

Open http://127.0.0.1:8765 . The service binds only to this computer. No images are sent to an external service. Supported uploads are JPEG or PNG, smaller than 10 MB and at most 20 million pixels. The automatically loaded sample comes from frame 20 of the existing real-image gallery input. It illustrates a visible prediction for reviewing the workflow, not a newly collected photograph, random performance sample or independent evaluation set.

## Review an Image

1. Select **Open image**, or retain the included sample, then select **Run detector**.
2. Select a predicted box. **Confirm** accepts an unchanged label and box. Changing the category or coordinates before confirming records a correction.
3. **Reject** retains the original prediction but excludes it from accepted review decisions. **Reset** returns a model prediction to pending, or removes a manually added box.
4. Select **Add missed sign** and drag a rectangle around a supported sign. Choose its category and confirm. A manual addition has no model confidence score.
5. Inspect the complete road image, including regions without predictions. Mark **Full image reviewed** only after that inspection. This checkbox is a reviewer assertion, not proof of completeness.
6. **Save review** downloads JSON including the image, model metadata, original predictions and review decisions. **Open review** reopens it. **Export CSV** downloads the detection-level audit table.

Use the JSON review to retain image-level review status, especially when there are no detections. A CSV with no detections contains only its column header. Exporting a CSV does not replace saving the JSON session. Changes remain in browser memory until exported; the tool does not create a central database.

## Records and Limitations

Original class, score and coordinates remain separate from reviewed class, box and decision. Rejected predictions remain visible in the record. Human additions are marked manual with null confidence. These are review decisions, not automatically validated ground truth. Exported images are embedded as a normalised JPEG for reopening; image SHA256 identifies those embedded bytes.

The five exact trained MTSD variants remain the detection scope. Review cannot reveal an invisible or unreadable sign, and the crop display does not restore missing source detail. Unsupported categories and broader international variants are not new trained classes. The first request includes model loading, and displayed request timing is not the controlled experiment benchmark.

The interface has functional desktop/mobile browser checks, real ONNX inference, review-action and export/reopen checks. No manual-versus-assisted user study, guaranteed time savings, production authentication or multi-user storage is claimed. It is a local student application, not a public network service.

## Repeat the Software Checks

```bash
pytest tests/test_review_app.py -q
node --test tests/review_logic.test.mjs
```

For the optional browser check, install the development test dependency with `npm install`, then `npx playwright install chromium`. Start the local Python service in another terminal and run `npm run test:browser`. It writes screenshots and a functional receipt under outputs/review_browser. This test deliberately changes labels and draws an arbitrary box to exercise controls; its exported test review is not a corrected dataset or valid ground truth. The interface itself does not require Node or npm to run.
