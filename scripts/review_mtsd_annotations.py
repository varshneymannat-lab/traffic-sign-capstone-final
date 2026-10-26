"""Check selected labels against source annotations and illustrate the review cases."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report/mtsd_annotation_review'
SOURCE = ROOT / 'data/raw/mtsd/mtsd_v2_fully_annotated'
TARGET = '#ffe600'
OTHER = '#00d9ff'


def expected_labels(annotation, mapping):
    return [(mapping[o['label']], o['bbox']) for o in annotation['objects'] if o['label'] in mapping]


def compare_labels(annotation, rows, mapping):
    expected = expected_labels(annotation, mapping)
    if Counter(c for c, box in expected) != Counter(r['sign_label'] for r in rows):
        return False
    remaining = list(expected)
    for row in rows:
        normalized = [float(row[k]) for k in ('x_center','y_center','box_width','box_height')]
        found = None
        for i, (category, box) in enumerate(remaining):
            values = [(box['xmin']+box['xmax'])/2/annotation['width'],
                      (box['ymin']+box['ymax'])/2/annotation['height'],
                      (box['xmax']-box['xmin'])/annotation['width'],
                      (box['ymax']-box['ymin'])/annotation['height']]
            if category == row['sign_label'] and all(abs(a-b)<1e-7 for a,b in zip(values, normalized)):
                found = i
                break
        if found is None:
            return False
        remaining.pop(found)
    return not remaining


def font(size):
    path=Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    if not path.exists():
        from matplotlib.font_manager import findfont
        path=Path(findfont('DejaVu Sans'))
    return ImageFont.truetype(str(path),size)


def source_panel(key, mapping):
    annotation = json.loads((SOURCE / 'annotations' / (key+'.json')).read_text())
    original = Image.open(SOURCE / 'images' / (key+'.jpg')).convert('RGB')
    scene = ImageOps.contain(original,(1200,800))
    draw = ImageDraw.Draw(scene)
    details = []
    for i,obj in enumerate(annotation['objects'],1):
        b=obj['bbox']; colour=TARGET if obj['label'] in mapping else OTHER
        box=[b['xmin']*scene.width/original.width,b['ymin']*scene.height/original.height,
             b['xmax']*scene.width/original.width,b['ymax']*scene.height/original.height]
        draw.rectangle(box,outline=colour,width=4)
        x,y=box[:2]
        tx,ty=x,max(0,y-30)
        if key=='-82y-KcR6EtdxdkOv5zzuA' and i in {1,2}:
            tx,ty=x-50,y+(20 if i==1 else -45)
            draw.line((tx+30,ty+14,x,y),fill=colour,width=3)
        draw.rectangle((tx,ty,tx+30,ty+28),fill=colour)
        draw.text((tx+4,ty),str(i),font=font(24),fill='black')
        details.append({'id':i,'source_label':obj['label'],'selected':obj['label'] in mapping,
                        'original_box_width':b['xmax']-b['xmin'], 'original_box_height':b['ymax']-b['ymin'],
                        'width_at_1280':(b['xmax']-b['xmin'])*1280/original.width,
                        'height_at_1280':(b['ymax']-b['ymin'])*1280/original.width})
    canvas=Image.new('RGB',(1200,scene.height+60),'white')
    canvas.paste(scene,(0,0))
    ImageDraw.Draw(canvas).text((14,scene.height+15),'Yellow: selected category    Cyan: other source category    Numbers: annotation IDs',font=font(23),fill='black')
    canvas.save(OUT/(key+'_all_source.jpg'),quality=95)
    selected=[obj for obj in annotation['objects'] if obj['label'] in mapping]
    crop_canvas=Image.new('RGB',(1200,300),'white')
    painter=ImageDraw.Draw(crop_canvas)
    for i,obj in enumerate(selected):
        b=obj['bbox']; crop=original.crop((int(b['xmin']),int(b['ymin']),int(b['xmax']+1),int(b['ymax']+1)))
        tile=ImageOps.contain(crop,(210,210),method=Image.Resampling.NEAREST)
        left=35+i*380
        crop_canvas.paste(tile,(left,12))
        label=mapping[obj['label']].replace('_',' ')
        painter.text((left,230),label,font=font(25),fill='black')
        painter.text((left,265),f'Original crop: {crop.width} x {crop.height} px',font=font(21),fill='black')
    crop_canvas.save(OUT/(key+'_target_crops.png'))
    return {'image_key':key,'all_annotations':details,'note':'Source annotations, not model predictions. Crops enlarged with nearest-neighbour sampling.'}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    mapping=dict(OmegaConf.load(ROOT/'pipelines/detection/classes.yaml').mtsd_labels)
    with (ROOT/'data/mtsd/annotations_split.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    images=defaultdict(list)
    for row in rows:
        images[row['image_key']].append(row)
    mismatches=[]; other=Counter(); source_count=0
    for key, retained in images.items():
        annotation=json.loads((SOURCE/'annotations'/(key+'.json')).read_text())
        if not compare_labels(annotation,retained,mapping):
            mismatches.append(key)
        source_count+=len(annotation['objects'])
        other.update(o['label'] for o in annotation['objects'] if o['label'] not in mapping)
    families=('--stop--','--no-entry--','--yield--','--maximum-speed-limit-30--','--maximum-speed-limit-50--')
    unmapped_related={label:n for label,n in other.items() if any(f in label for f in families)}
    record={'prepared_scenes_checked':len(images),'retained_target_boxes':len(rows),
            'source_annotation_count':source_count,'mismatched_scenes':mismatches,
            'non_target_source_annotations':sum(other.values()),'non_target_label_counts':dict(other),
            'unmapped_related_source_labels':unmapped_related,
            'policy':'Only the five exact mapped source labels are prediction targets. Other labelled signs are background for this closed-set task.',
            'limitations':'Comparison verifies retention and geometry of supplied target annotations. It cannot discover signs missing from the original annotations, establish complete manual semantic review, or justify merging unreviewed label variants.',
            'examples':[source_panel(k,mapping) for k in ['-82y-KcR6EtdxdkOv5zzuA','-5ihylTtxA3-CGKOhyhDmQ']]}
    (OUT/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
    (OUT/'README.md').write_text(f'''# Annotation Review

Compared all {len(images):,} prepared scenes and {len(rows):,} retained target boxes with original MTSD annotations. Mismatched scenes: {len(mismatches)}. Target category counts and normalized box coordinates are checked, not merely the existence of label files.

The first review example contains one selected speed-30 sign and three other source-annotated signs. The large road-bump and priority-for-oncoming-traffic signs are outside the five categories; they are not missing from the source. The numbered image shows all source boxes, with yellow for selected targets and cyan for other categories.

The second example contains two selected speed-50 signs. Their original boxes are approximately 57-59 pixels across, but are about 14 pixels across after the existing 1280-pixel preparation. Enlarged crops illustrate available original detail, not reconstructed detail. Increasing label font size only fixes presentation readability.

Other source classes are deliberately background in this closed-set five-category detector. A prediction on one of them is not automatically correct. This policy is different from losing an annotation for an in-scope sign. No cross-class relabelling is used to hide missing labels.

The category names abbreviate five exact g1 source labels. Related but unmapped source variants also occur: {json.dumps(unmapped_related, sort_keys=True)}. They were not merged during the recorded experiments. This is a real coverage limitation, including related stop/speed variants, not evidence that all visual variants of those five categories are labelled as targets. A broader semantic task needs reviewed remapping and retraining; test labels must not be silently revised after seeing results.

The audit cannot prove that the source contains every real sign, and does not substitute for a complete manual annotation review. Difficult and unsupported categories remain limitations. Whole-image versus overlapping-tile inference was selected on internal validation only; measured accuracy and timing are recorded in the final technical report.

Images and annotations: Mapillary Traffic Sign Dataset v2. Source access: https://www.mapillary.com/dataset/trafficsign . Preserve the distributed dataset licence.
''')
    print(json.dumps({k:v for k,v in record.items() if k not in {'non_target_label_counts','examples'}},indent=2))
    if mismatches:
        raise SystemExit('Source/retained target mismatch requires correction before submission.')


if __name__=='__main__':
    main()
