"""Restore the recovered nine-step pipeline at readable single-column size."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib._mathtext as math_layout
# Keep mathematical subscripts readable at the final printed size.
math_layout.SHRINK_FACTOR=1.0
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Polygon,Circle
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'figures'
WIDTH,HEIGHT=251.06,543.0; FONT='Times New Roman'
FONT_FILE=Path(font_manager.findfont(FONT,fallback_to_default=False))
plt.rcParams.update({'font.family':FONT,'font.size':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'custom','mathtext.rm':FONT,'mathtext.it':FONT+':italic','mathtext.bf':FONT+':bold','mathtext.fallback':None})
fig=plt.figure(figsize=(WIDTH/72,HEIGHT/72),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,WIDTH),ylim=(HEIGHT,0));ax.axis('off')
INK,LINE='#334155','#475569';ORANGE,BLUE,PURPLE,TEAL,GREEN='#c2410c','#1e40af','#6d28d9','#0d9488','#065f46'
texts=[]
def label(x,y,text,size=8.5,weight='normal',color=INK,region=None):
 a=ax.text(x,y,text,ha='center',va='center',fontsize=size,fontweight=weight,color=color,zorder=5);texts.append((a,region));return a
def box(x,y,w,h,title,lines=(),fill='#fff',edge=LINE,number=None,title_y=None,line_y=None,title_size=9.3):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=3',linewidth=.7,edgecolor=edge,facecolor=fill,zorder=2);ax.add_patch(p)
 if title:label(x+w/2,title_y if title_y is not None else y+8,title,title_size,'bold',region=p)
 for s,ty in zip(lines,line_y if line_y is not None else [y+20+10*i for i in range(len(lines))]):label(x+w/2,ty,s,region=p)
 if number is not None:
  cx,cy=x+8,y+8;ax.add_patch(Circle((cx,cy),5.8,facecolor=edge,edgecolor='none',zorder=4));label(cx,cy,str(number),8.5,'bold','#fff',p)
 return p
def arrow(points,color=LINE,dash=False):
 xs,ys=zip(*points)
 if len(points)>2:ax.plot(xs[:-1],ys[:-1],color=color,linewidth=.75,linestyle=(0,(3,2)) if dash else '-',zorder=3)
 ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=5.8,linewidth=.75,color=color,linestyle=(0,(3,2)) if dash else '-',shrinkA=0,shrinkB=.5,zorder=3))
def down(x,y0,y1):arrow([(x,y0),(x,y1)])
def diamond(cx,cy,hw,hh,lines,offsets):
 p=Polygon([(cx,cy-hh),(cx+hw,cy),(cx,cy+hh),(cx-hw,cy)],closed=True,facecolor='#fefce8',edgecolor='#a16207',linewidth=.8,zorder=2);ax.add_patch(p)
 for i,(s,dy) in enumerate(zip(lines,offsets)):label(cx,cy+dy,s,8.5,'bold' if i==0 else 'normal',region=p)
 return p
box(2,2,247,58,'Problem setup',[r'$\dot x=-0.1yz,\quad\dot y=1.6xz,\quad\dot z=-0.75xy$',r'$u_0=(0.5,\,0.75,\,1.0)$; 0 ≤ t ≤ 13.26446','27 windows · 1,536 points/window · float64'],'#fff7ed',ORANGE,1,line_y=[25,38,51])
ax.add_patch(FancyBboxPatch((2,66),247,378,boxstyle='round,pad=0,rounding_size=4',facecolor='white',edgecolor=GREEN,linewidth=.7,linestyle=(0,(3,2)),zorder=0))
ax.add_patch(FancyBboxPatch((10,61),190,12,boxstyle='round,pad=0,rounding_size=5',facecolor=GREEN,edgecolor='none',zorder=4))
label(105,67,'REPEAT FOR EACH WINDOW  k = 0, …, 26',8.5,'bold','#fff');down(125.5,60,76)
box(15,76,220,62,'PINN for window k',fill='#eff6ff',edge=BLUE,number=2)
# Preserve the original Input → Hidden → Output → Hard IC subblocks.
for x,w,t,lines in [(20,27,'Input',[r'$t$']),(54,39,'Hidden',['4 × 60','tanh']),(100,38,'Output',[r'$N_k(t)$','3 states']),(145,87,'Hard IC',[r'$\hat u_k=u_{0,k}$',r'$+(t-t_k)N_k(t)$'])]:
 box(x,94,w,42,t,lines,'#fff','#93c5fd',title_size=8.5,title_y=100,line_y=[112,128][:len(lines)])
for a,b in [(47,54),(93,100),(138,145)]:arrow([(a,111),(b,111)])
down(125.5,138,140)
box(15,140,220,27,'ODE residual',[r'$r_k=d\hat u_k/dt-f(\hat u_k)$; derivatives by autodiff'],'#f5f3ff',PURPLE,3,line_y=[159]);down(125.5,167,174)
box(15,174,220,54,'Causal residual loss',[r'$L_i=\mathrm{mean}(r_i^2),\quad w_i=\exp(-\varepsilon\Sigma_{j<i}L_j)$',r'$L=\mathrm{mean}(w_iL_i)$; detach weights for gradients','ε: 0.01 → 0.1 → 1 → 10'],'#f5f3ff',PURPLE,4,line_y=[196,212,223.5]);down(125.5,228,234)
box(15,234,220,27,'Adam optimization',['Learning rate × 0.9 every 1,000 updates'],'#f0fdfa',TEAL,5,line_y=[253]);down(77,261,263)
diamond(77,290,62,27,['Stage complete?',r'Verified $\min\,w>0.99$','or 4,000 steps'],[-10,0,11])
diamond(189,290,43,24,['All ε values','used?'],[-5,5])
arrow([(139,290),(146,290)]);label(142,279,'YES',8.5,'bold')
arrow([(15,290),(8,290),(8,201),(15,201)]);label(18,303,'NO',8.5,'bold')
arrow([(189,314),(189,324)]);label(213,318,'NO',8.5,'bold')
arrow([(168,302),(168,318),(77,318),(77,324)]);label(114,311,'YES',8.5,'bold')
box(149,324,82,27,'Next ε',['Keep parameters'],'#fff7ed',ORANGE,line_y=[343],title_size=9.0)
arrow([(231,337),(241,337),(241,201),(235,201)],ORANGE,True)
box(15,324,125,33,'L-BFGS',['≤ 1,000 iterations','Unweighted residual MSE'],'#f0fdfa',TEAL,6,line_y=[343,352],title_size=9.0);down(77,357,365)
box(15,365,125,25,'Save window',['Checkpoint · history · prediction'],'#f8fafc',LINE,7,line_y=[383],title_size=9.0);down(77,390,392)
diamond(77,415,58,23,['Last window?',r'$k=26$'],[-5,6])
box(149,397,82,46,'Next window',[r'$u_{0,k+1}=\hat u_k(t_{k+1})$','Copy parameters',r'$k\leftarrow k+1$'],'#ecfdf5',GREEN,line_y=[416,427.5,437],title_size=9.0)
arrow([(135,415),(149,415)]);label(142,404,'NO',8.5,'bold')
arrow([(231,415),(247,415),(247,104),(235,104)],GREEN,True)
arrow([(77,438),(77,451)]);label(103,439,'YES',8.5,'bold')
box(2,451,247,42,'Reference evaluation',[r'DOP853 · 13,265 points · rtol $10^{-10}$ · atol $10^{-12}$','Evaluation only; not used for optimization'],'#eff6ff',BLUE,8,line_y=[474,487]);down(125.5,493,500)
box(2,500,247,39,'Evaluation outputs',['MAE · RMSE · maximum error · residual MSE','Invariant drift · training cost'],'#f8fafc',LINE,9,line_y=[520,532])
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for artist,region in texts:
 assert artist.get_fontsize()>=8.5
 bb=artist.get_window_extent(renderer)
 corners=[[bb.x0,bb.y0],[bb.x0,bb.y1],[bb.x1,bb.y0],[bb.x1,bb.y1]]
 assert all(0<=x<=WIDTH and 0<=y<=HEIGHT for x,y in ax.transData.inverted().transform(corners)),artist.get_text()
 if region is not None:
  path=region.get_path().transformed(region.get_transform())
  assert all(path.contains_point(point) for point in corners),(artist.get_text(),ax.transData.inverted().transform(corners),region.get_extents())
for i,(first,region) in enumerate(texts):
 if region is None:continue
 for second,other_region in texts[i+1:]:
  if region is other_region:
   assert not first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)),(first.get_text(),ax.transData.inverted().transform(first.get_window_extent(renderer).get_points()),second.get_text(),ax.transData.inverted().transform(second.get_window_extent(renderer).get_points()))
OUT.mkdir(exist_ok=True)
fig.savefig(OUT/'method.pdf',metadata={'Title':'Sequential windowed PINN training','Author':'','Subject':'Nine-step method pipeline','Keywords':''})
fig.savefig(OUT/'method.png',dpi=300);fig.savefig(Path(__file__).with_name('pipeline.svg'));plt.close(fig)
import fitz
pdf=fitz.open(OUT/'method.pdf')
sizes=[s['size'] for b in pdf[0].get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
fonts=[{'name':f[3],'embedded':len(pdf.extract_font(f[0])[3])>0,'type':f[2]} for f in pdf[0].get_fonts(full=True)]
assert min(sizes)>=8.49,min(sizes)
assert all(f['embedded'] and f['type']!='Type3' and 'TimesNewRoman' in f['name'] for f in fonts),fonts
assert abs(pdf[0].rect.width-WIDTH)<.01
metadata={'source':Path(__file__).name,'derived_from':'Recovered pipeline_figure_original.py and pipeline-original.html','width_pt':WIDTH,'height_pt':HEIGHT,'minimum_pdf_text_size_pt':min(sizes),'fonts':fonts,'font_file':FONT_FILE.name,'font_file_sha256':hashlib.sha256(FONT_FILE.read_bytes()).hexdigest(),'structure':'Original nine numbered sections, four network subblocks, three decision diamonds and labelled return paths','protocol':{'windows':27,'window_indices':[0,26],'network':'four hidden layers of 60 tanh units; separate network per window','hard_initial_state':True,'points_per_window':1536,'causal_adam_epsilon':[.01,.1,1,10],'causal_weights_detached_for_gradient':True,'stage_threshold':'verified post-update minimum causal weight > 0.99','stage_cap_adam_updates':4000,'lbfgs_objective':'unweighted residual mean square','lbfgs_iterations_per_window_at_most':1000,'handoff':'predicted endpoint and copied trained parameters','reference':'DOP853, 13265 evaluation times; not used for optimization'},'validation':{'labels_inside_node_shapes':True,'text_inside_page':True,'no_text_overlap_within_nodes':True},'pdf_sha256':hashlib.sha256((OUT/'method.pdf').read_bytes()).hexdigest()}
Path(__file__).with_name('pipeline-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8');print(json.dumps(metadata,indent=2))
