"""Build the manuscript table from the pinned local PR37 artifacts."""
from pathlib import Path
import contextlib
import importlib.util
import io
import sys
import subprocess
root=Path(sys.argv[1])
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()=='babc47aab9f7cdbe8fde832cdd1e670eab1f9210'
spec=importlib.util.spec_from_file_location('report',root/'scripts/ablation_report.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with contextlib.redirect_stdout(io.StringIO()): d=m.load()
assert len(d)==16
rows=[]
for group,label in [('r4_candidate','Candidate'),('r1_no_causal','No causal'),('r2_no_warm','No warm start'),('r6_sequential','Shared network'),('r3_causal_single','Causal single'),('F1','Full interval')]:
    g=d[d.group==group]; assert len(g)>0
    r=g.rmse_combined_l2
    def fmt(v):
        if v>=.01:return f'{v:.3f}'
        a,b=f'{v:.2e}'.split('e');return a+r'\!\times\!10^{'+str(int(b))+'}'
    rr='--' if len(g)==1 else '$'+fmt(r.min())+'$--$'+fmt(r.max())+'$'
    residual='--' if group=='F1' else '$'+fmt(g.residual.median())+'$'
    rows.append(label+' & '+str(len(g))+' & $'+fmt(r.median())+'$ & '+rr+' & '+residual+chr(92)*2)
out=r'''\begin{table*}[t]
\caption{Ablation and failure-case results on 13,265 evaluation times. RMSE is the combined state norm; residual MSE is the median dense-grid value. Candidate seeds are 0--4; no-causal, no-warm-start, and shared-network seeds are 1--3. Both full-interval cases use seed 0. Candidate seed 0 is the earlier run of record; F1 is a separate reference.}
\label{tab:ablations}
\centering\footnotesize
\begin{tabular}{lcccc}
\toprule
Configuration & Seeds & Median RMSE & RMSE range & Residual MSE\\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table*}
'''
Path('writings/conference-paper/ablation-table.tex').write_text(out)
d.to_csv('docs/manuscript-revision/verified-ablation-values.csv',index=False)
print(out)
