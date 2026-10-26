from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser(description='Create reproducible EDA for the prepared real-scene detection task.')
    parser.add_argument('--data-dir',type=Path,default=Path('data/mtsd'))
    parser.add_argument('--output-dir',type=Path,default=Path('report/mtsd_eda'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    data = pd.read_csv(args.data_dir / 'annotations_split.csv')
    summary = json.loads((args.data_dir / 'dataset_summary.json').read_text())
    (args.output_dir / 'dataset_summary.json').write_text(json.dumps(summary,indent=2))
    classes = summary['classes']
    figure,axes = plt.subplots(1,2,figsize=(11,3.6))
    x = np.arange(len(classes))
    for offset,split,colour in zip([-.25,0,.25],['train','val','test'],['#17604d','#c56c39','#527d9e']):
        axes[0].bar(x+offset,[summary['image_counts_per_class'][split].get(c,0) for c in classes],width=.25,label=split,color=colour)
    axes[0].set_xticks(x,labels=[c.replace('speed_limit_','speed ').replace('_',' ') for c in classes],rotation=20)
    axes[0].set_ylabel('Distinct scenes containing category')
    axes[0].legend()
    expected_sizes = [416,640]
    approximate_pixels = {str(s):(np.sqrt(data.box_width*data.box_height)*s).tolist() for s in expected_sizes}
    for size,colour in zip(expected_sizes,['#527d9e','#17604d']):
        axes[1].hist(approximate_pixels[str(size)],bins=np.linspace(0,150,31),alpha=.55,color=colour,label=f'{size}-pixel square')
    axes[1].set(xlabel='Approximate sign side length (pixels)',ylabel='Annotated boxes')
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / 'distribution_and_sign_size.png',dpi=160)
    plt.close(figure)
    gallery = Image.new('RGB',(920,760),'white')
    draw = ImageDraw.Draw(gallery)
    provenance=[]
    seen_scenes=set()
    scene_examples=[]
    for category_index,category in enumerate(classes):
        draw.text((10,category_index*150+10),category.replace('_',' '),fill='black')
        candidates = data[(data.sign_label==category)&(data.split=='train')].head(5)
        for index,(_,row) in enumerate(candidates.iterrows()):
            with Image.open(args.data_dir / row.image_path) as source:
                image=source.convert('RGB')
            width,height=image.size
            x,y,bw,bh=[float(row[k]) for k in ['x_center','y_center','box_width','box_height']]
            box=(int((x-bw/2)*width),int((y-bh/2)*height),max(1,int((x+bw/2)*width)),max(1,int((y+bh/2)*height)))
            crop=image.crop(box)
            if min(crop.size)<1:
                raise ValueError('Empty annotation crop in EDA.')
            resized=ImageOps.contain(crop,(110,110),method=Image.Resampling.NEAREST)
            gallery.paste(resized,(195+index*140,category_index*150+30))
            provenance.append({'image_key':row.image_key,'category':category,'split':'train','prepared_crop_size':list(crop.size)})
            if row.image_key not in seen_scenes and not any(r['category']==category for r in scene_examples):
                scene=image.copy()
                painter=ImageDraw.Draw(scene)
                targets=data[data.image_key==row.image_key]
                for _,target in targets.iterrows():
                    tx,ty,tw,th=[float(target[k]) for k in ['x_center','y_center','box_width','box_height']]
                    rectangle=((tx-tw/2)*width,(ty-th/2)*height,(tx+tw/2)*width,(ty+th/2)*height)
                    painter.rectangle(rectangle,outline='#ffeb00',width=3)
                    painter.text((max(0,rectangle[0]),max(0,rectangle[1]-15)),target.sign_label,fill='#ffeb00',stroke_width=1,stroke_fill='black')
                destination=args.output_dir / ('ground_truth_'+category+'.jpg')
                scene.save(destination,quality=95)
                scene_examples.append({'category':category,'image_key':row.image_key,'file':destination.name,'type':'ground_truth_not_model_prediction'})
                seen_scenes.add(row.image_key)
    gallery.save(args.output_dir / 'real_category_examples.png')
    means=[]
    dimensions=[]
    for path in sorted(data.image_path.unique()):
        with Image.open(args.data_dir / path) as image:
            dimensions.append(list(image.size))
            means.append(float(np.asarray(image.convert('L').resize((64,64))).mean()))
    stats={'prepared_scenes':len(dimensions),'prepared_boxes':len(data),'all_images_decoded':True,
           'brightness_mean_gray_range':[min(means),max(means)],
           'approximate_object_side_pixels_median':{s:float(np.median(values)) for s,values in approximate_pixels.items()},
           'object_size_caveat':'sqrt(normalized box area) times square size is a relative-size proxy, not actual letterboxed pixel size; aspect ratios change that size.',
           'group_count_by_split':data.groupby('split').group_id.nunique().to_dict(),
           'exclusions':Counter(r['reason'] for r in summary['exclusions']),
           'class_mapping_review':'Spot-checked real training crops: stop includes international text variants; no-entry is the horizontal white bar; yield is the inverted red triangle; speeds are round 30/50 signs. This is not a manual audit of every annotation.'}
    (args.output_dir / 'eda_statistics.json').write_text(json.dumps(stats,indent=2))
    (args.output_dir / 'example_provenance.json').write_text(json.dumps({'crops':provenance,'scenes':scene_examples},indent=2))
    rows='\n'.join(f"| {c.replace('_',' ')} | {summary['image_counts_per_class']['train'][c]} | {summary['image_counts_per_class']['val'][c]} | {summary['image_counts_per_class']['test'][c]} |" for c in classes)
    text=f'''# EDA: Real Road Scenes from MTSD

## Data Source and Task

The source is the fully annotated Mapillary Traffic Sign Dataset v2, not generated drawings or pasted sign crops. The selected task locates and classifies five categories in road photographs. The authorised annotation archive matches its supplied MD5 checksum. Selected image files were checked using original ZIP member CRC32 and saved SHA256 digests; full image-archive MD5 checks are not claimed for range-downloaded subsets.

## Distinct Scene Counts

| Category | Training | Internal Validation | Held-out Local Test |
| --- | ---: | ---: | ---: |
{rows}

There are {summary['split_images']['train']} training, {summary['split_images']['val']} validation and {summary['split_images']['test']} test scenes. A scene may contain multiple categories, so category counts do not sum to scene totals. All five target boxes in each retained photograph are included.

## Split and Duplicate Checks

Official training scenes supply internal training/validation. The selected official validation scenes are reserved for final local testing, not tuning; this is not a hidden official-test benchmark score. Exact decoded pixels and 64-bit full-scene difference hashes screen duplicates. Visually similar hashes stay in one internal group. Three training scenes in groups overlapping held-out data were excluded. These checks do not prove independent physical-sign identity without sequence metadata.

## Visual and Size Checks

![Enlarged real training crops](real_category_examples.png)

These are the first five retained training crops per category, enlarged with nearest-neighbour sampling; enlargement does not add detail. Original crop sizes and source keys are recorded in example_provenance.json. Many signs occupy a small part of a large road scene. Blur, shadows, viewing angle, and international text variants are visible.

The global MTSD speed labels identify the displayed number and sign family. The application does not infer or convert national speed units from the shape alone. GTSRB speed examples come from the German benchmark; no claim is made that every global round 30/50 sign uses km/h.

![Category counts and object-size proxy](distribution_and_sign_size.png)

The sign-size chart uses a normalized-area proxy, not measured letterboxed dimensions. It motivates comparing detector resolutions, without predicting which will win. All {len(dimensions)} prepared images decoded successfully. Mean grayscale brightness across thumbnails ranges from {min(means):.1f} to {max(means):.1f} on a 0-255 scale.

## Annotation Policy and Limitations

Panoramas are excluded, including wraparound rectangles that require a different representation. Scenes with selected target signs flagged ambiguous, occluded or out-of-frame are conservatively excluded. This simplifies the first experiment but makes its scope easier than unrestricted road video. Other sign categories remain in the photographs as background for the five-class task; they must not be relabelled as one of the targets. Selected scenes contain target signs, so the study does not estimate false alarms per hour on sign-free drives. No synthetic scenes or duplicate copies are used to inflate class counts.

## Source

[Official dataset access](https://www.mapillary.com/dataset/trafficsign) and [Ertler et al., ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680069.pdf). Preserve the distributed dataset licence and attribution. Temporary account-specific download links are private and are not published with the project.
'''
    (args.output_dir / 'README.md').write_text(text)
    print(json.dumps(stats,indent=2))


if __name__ == '__main__':
    main()
