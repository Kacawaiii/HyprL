"""Compose the nine forward/backward captures; Pillow is the only dependency."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('report_dir', type=Path)
parser.add_argument('--tag', choices=('desktop', 'mobile'))
parser.add_argument('--kind', choices=('pullback', 'warp'))
args = parser.parse_args()
for tag in ([args.tag] if args.tag else ('desktop', 'mobile')):
    width, height = (480, 300) if tag == 'desktop' else (260, 563)
    for kind in ([args.kind] if args.kind else ('pullback', 'warp')):
        data = json.loads((args.report_dir / f'norezoom-{tag}-{kind}-sizes.json').read_text())
        for direction in ('forward', 'backward'):
            sheet = Image.new('RGB', (width * 3, (height + 46) * 3 + 36), '#101217')
            draw = ImageDraw.Draw(sheet)
            draw.text((12, 10), f'No rezoom / {tag} / {kind} / {direction}', fill='#eee7db')
            records = [r for r in data['records'] if r['direction'] == direction]
            for i, record in enumerate(records):
                frame = args.report_dir / f'norezoom-{tag}-{kind}-{direction}-{i}.jpg'
                with Image.open(frame) as source:
                    image = source.resize((width, height), Image.Resampling.LANCZOS)
                x, y = i % 3 * width, i // 3 * (height + 46) + 36
                sheet.paste(image, (x, y))
                radius = record['planetAngularRadiusPx']
                subject = record.get('monolithHeightPx')
                detail = f'monolith h={subject:.2f}px' if subject is not None else 'orbit'
                if kind == 'warp':
                    detail = f"singularity r={record['singularityRingRadiusPx']:.2f}px"
                draw.text((x + 8, y + height + 5), f"{i + 1:02}  t={record['t']:.3f}  {detail}", fill='#ccc6bc')
                draw.text((x + 8, y + height + 23), f'planet angular r={radius:.2f}px', fill='#a9b7c7')
            sheet.save(args.report_dir / f'norezoom-{tag}-{kind}-{direction}-sheet.jpg', quality=92)
