.PHONY: install eda prepare train compare eval export predict infer test tensorboard restore detection-eda detector-export classifier-report report

install:
	python3 -m pip install -r requirements.txt

restore:
	python3 scripts/restore_inference_models.py

prepare:
	python3 scripts/prepare_mtsd_crops.py

eda:
	$(MAKE) detection-eda

compare:
	python3 scripts/compare_classifiers.py --data-dir data/mtsd_crops --output-dir outputs/mtsd_classifier_comparison --epochs 5 --image-size 128 --device auto

train:
	python3 -m src.train

eval:
	python3 -m src.eval

export:
	python3 scripts/export_selected.py

predict:
	python3 -m src.predict_image $(IMAGE)

infer:
	python3 -m src.detect_signs $(INPUT) --output-dir $(OUTPUT)

detection-eda:
	python3 scripts/eda_mtsd.py
	python3 scripts/review_mtsd_annotations.py
	python3 scripts/build_mtsd_eda_report.py

detector-export:
	python3 scripts/export_detector.py

classifier-report:
	python3 scripts/build_classifier_results.py

report:
	python3 scripts/build_final_report.py

test:
	pytest -q

tensorboard:
	tensorboard --logdir outputs
