from pathlib import Path
import json,hashlib,zipfile,subprocess
P=Path('writings/iceeict2027-final')
files=sorted(f for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc' and (f.parts[len(P.parts)] in ['review','figures','figure-sources'] or f.name in ['main.tex','main.pdf','manuscript.txt','references.bib','main.bbl','IEEEtran.cls','IEEEtran.bst','ablation-table.tex','README.md']))
v=json.loads((P/'review/final-verification.json').read_text(encoding='utf-8'))
manifest={'paper_source_commit':v['paper_source_commit'],'pdf_sha256':v['pdf_sha256'],'pages':6,'note':'Source/PDF frozen at reviewed commit; later commits contain audit and packaging only.','files':{f.relative_to(P).as_posix():{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files}}
(P/'delivery-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
with zipfile.ZipFile('writings/iceeict2027-final.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in files+[P/'delivery-manifest.json']:z.write(f,'iceeict2027-final/'+f.relative_to(P).as_posix())
names=['main.tex','main.pdf','manuscript.txt','references.bib','main.bbl','IEEEtran.cls','IEEEtran.bst','ablation-table.tex','figures/method.pdf','figures/plateau-results.pdf','figures/solution_vs_reference.pdf']
with zipfile.ZipFile('writings/iceeict2027-manuscript.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n in names:z.write(P/n,n)
print('packaged',len(files),'manifest files and',len(names),'upload files')
