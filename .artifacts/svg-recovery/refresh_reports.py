from pathlib import Path
import json,hashlib,subprocess
import fitz
from PIL import Image,ImageDraw
P=Path('writings/iceeict2027-final');R=P/'review'
old=json.loads((R/'final-verification.json').read_text(encoding='utf-8'))
sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(); data=(P/'main.pdf').read_bytes(); digest=hashlib.sha256(data).hexdigest(); doc=fitz.open(stream=data,filetype='pdf')
assert len(doc)==6 and len(data)<3000000
fonts={f[0]:f for page in doc for f in page.get_fonts(full=True)}
assert all(doc.extract_font(x)[3] for x in fonts)
assert all(f[2]!='Type3' for f in fonts.values())
assert all(abs(page.rect.width-595.28)<.1 and abs(page.rect.height-841.89)<.1 for page in doc)
meta=json.loads((P/'figure-sources/pipeline-metadata.json').read_text(encoding='utf-8'))
minpt=min(s['size'] for page in doc for block in page.get_text('dict')['blocks'] if 'lines' in block for line in block['lines'] for s in line['spans'] if 'TimesNewRoman' in s['font'])
assert minpt>=7.99
for name in ['requirements-verdict.json','requirements-verdict.md','changes.md','task-status.json']:
 f=R/name;s=f.read_text(encoding='utf-8').replace(old['paper_source_commit'],sha).replace(old['pdf_sha256'],digest).replace(format(old['bytes'],','),format(len(data),',')).replace(f"among {old['fonts']} final fonts",f'among {len(fonts)} final fonts').replace(f"All {old['fonts']} font programs",f'All {len(fonts)} font programs');f.write_text(s,encoding='utf-8')
old.update(paper_source_commit=sha,pdf_sha256=digest,pages=6,bytes=len(data),fonts=len(fonts),minimum_placed_figure_text_pt=minpt)
f=fitz.open(P/'figures/method.pdf');old['figures']['method.pdf'].update(sha256=hashlib.sha256((P/'figures/method.pdf').read_bytes()).hexdigest(),width_pt=f[0].rect.width,height_pt=f[0].rect.height,minimum_font_pt=min(z['size'] for b in f[0].get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for z in line['spans']),fonts=sorted({z[3] for z in f[0].get_fonts()}))
old['method_placed_height_pt']=f[0].rect.height*251.06/f[0].rect.width
old['notes'].append('Actual recovered HTML/SVG generator replaces the rejected Matplotlib reconstruction; original source is preserved alongside typography-only derivation.')
(R/'final-verification.json').write_text(json.dumps(old,indent=2)+'\n',encoding='utf-8')
thumbs=[]
for n,page in enumerate(doc,1):
 pix=page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False);pix.save(R/f'rendered-pages/page-{n}.png');im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);im.thumbnail((470,665));thumbs.append(im)
canvas=Image.new('RGB',(1440,1390),'#ddd');draw=ImageDraw.Draw(canvas)
for i,im in enumerate(thumbs):x=(i%3)*480;y=(i//3)*695;canvas.paste(im,(x+5,y+22));draw.text((x+8,y+3),f'Page {i+1}',fill='black')
canvas.save(R/'contact-sheet.png')
print(json.dumps({'source':sha,'hash':digest,'pages':6,'bytes':len(data),'fonts':len(fonts),'min_text':minpt},indent=2))
# Render the actual supplied method PDF beside the source-derived SVG export.
ref=Path('C:/Users/User/.codex/worktrees/2a6e/lorenz1960-pinn/writings/iceeict2027-submission/figures/method.pdf')
comparison=Image.new('RGB',(1800,2200),'white');d=ImageDraw.Draw(comparison)
for i,(f,label) in enumerate([(ref,'Original supplied diagram'),(P/'figures/method.pdf','Original source with revised typography')]):
 dd=fitz.open(f);pix=dd[0].get_pixmap(matrix=fitz.Matrix(3,3),alpha=False);im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);im.thumbnail((860,2140));comparison.paste(im,(20+900*i,45));d.text((20+900*i,10),label,fill='black')
comparison.save(R/'method-original-vs-restored.png')
prior=Image.open('.artifacts/method-restore/prior-comparison.png').crop((0,782,1040,1876));prior=prior.resize((1800,round(prior.height*1800/1040)));allfigs=Image.new('RGB',(1800,2200+prior.height),'white');allfigs.paste(comparison,(0,0));allfigs.paste(prior,(0,2200));allfigs.save(R/'figure-comparison.png')
