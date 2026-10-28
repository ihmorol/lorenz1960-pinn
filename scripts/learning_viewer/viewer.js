const DATA=/* VIEWER_DATA */;
const $=id=>document.getElementById(id),config={responsive:true,displayModeBar:false};
let frame=DATA.frames.length-1,cursor=DATA.frames.at(-1).evaluation_count,timer=null,busy=false,scene='learning';
const camera={eye:{x:1.15,y:1.25,z:.9}},font={family:'system-ui,sans-serif',size:13,color:'#435565'};
function xyz(v){return {x:v.map(p=>p[0]),y:v.map(p=>p[1]),z:v.map(p=>p[2])}}
function base(title){return {title:{text:title,font:{size:15}},font,paper_bgcolor:'#fff',plot_bgcolor:'#fff',margin:{l:64,r:25,t:45,b:50},uirevision:'keep'}}
function space(title,bounds){return {...base(title),margin:{l:0,r:0,t:36,b:0},scene:{camera,aspectmode:'cube',xaxis:{title:{text:'x'},range:bounds[0]},yaxis:{title:{text:'y'},range:bounds[1]},zaxis:{title:{text:'z'},range:bounds[2]}},showlegend:false}}
function curve(points,name,color,width=4){return {type:'scatter3d',mode:'lines',...xyz(points),name,line:{color,width},hovertemplate:'x %{x:.4f}<br>y %{y:.4f}<br>z %{z:.4f}<extra>'+name+'</extra>'}}
function stop(){clearInterval(timer);timer=null;$('play').textContent='Play this window'}
function windowInfo(){const f=DATA.frames[frame];$('window').value=frame;$('interval').innerHTML='Selected interval<br><strong>'+f.start.toFixed(5)+' to '+f.stop.toFixed(5)+'</strong><br>Total interval: 0 to '+DATA.frames.at(-1).stop.toFixed(5);$('counts').textContent=f.adam_count.toLocaleString()+' Adam updates + '+(f.evaluation_count-f.adam_count).toLocaleString()+' L-BFGS evaluations';$('stageRows').innerHTML=f.stages.map(s=>'<div class="stage-row">epsilon '+s.eps+': '+s.adam_steps.toLocaleString()+' Adam updates<br>'+(s.threshold_met?'Threshold reached':'Iteration cap reached')+'</div>').join('');$('cursor').max=f.evaluation_count;$('cursor').value=cursor;$('step').textContent='Window '+f.window+' of 27 | selected time '+f.start.toFixed(5)+' to '+f.stop.toFixed(5)+' | total time 13.26446';}
function metrics(){const f=DATA.frames[frame],n=Math.min(cursor,f.evaluation_count-1),complete=cursor===f.evaluation_count,eps=f.stages.find(s=>cursor<s.stop);$('phase').textContent=complete?'Window training complete':cursor<f.adam_count?'Adam update '+(cursor+1).toLocaleString()+' | epsilon '+eps.eps:'L-BFGS evaluation '+(cursor-f.adam_count+1).toLocaleString();$('cursorLabel').textContent='Local evaluation '+cursor.toLocaleString()+' / '+f.evaluation_count.toLocaleString();$('loss').innerHTML=f.objectives[n].toExponential(3)+'<span>Recorded objective'+(complete?' (last evaluation)':cursor<f.adam_count?' before Adam update':' at L-BFGS evaluation')+'</span>';let r=-1;f.history_t.forEach((t,i)=>{if(t<=cursor)r=i});$('rmse').innerHTML=r<0?'Not recorded yet<span>Reference RMSE</span>':f.history_error[r].toExponential(3)+'<span>Recorded prefix RMSE at local evaluation '+f.history_t[r]+'</span>';$('cursor').value=cursor;}
async function geometry(){const f=DATA.frames[frame],zoom=$('focus').value==='window',ref=zoom?f.window_reference:DATA.reference,pred=zoom?f.window_xyz:f.xyz;let bounds=DATA.bounds;if(zoom){const pts=[...ref,...pred];bounds=[0,1,2].map(i=>{const v=pts.map(p=>p[i]),lo=Math.min(...v),hi=Math.max(...v),pad=Math.max((hi-lo)*.12,.001);return [lo-pad,hi+pad]});}$('trajectoryCaption').textContent=zoom?'Dashed gray = ODE reference for this window. Solid orange = this window final prediction. Zoom changes axis ranges.':'Dashed gray = reference over the full interval. Solid orange = completed predictions for windows 1 through '+f.window+'. Other windows are not displayed as trained.';const traces=[];if($('showReference').checked){const c=curve(ref,'ODE reference (dashed gray)','#64748b',5);c.line.dash='dash';traces.push(c)}if($('showPrediction').checked)traces.push(curve(pred,'PINN final prediction (orange)','#d97706',5));await Plotly.react('trajectory',traces,{...space('',bounds),uirevision:zoom?'window-'+frame:'full',showlegend:true,legend:{orientation:'h',x:.01,y:1,itemclick:false,itemdoubleclick:false}},config);}
async function learning(){windowInfo();metrics();const f=DATA.frames[frame],kind=$('errorKind').value;let x,y,all,title,xlabel,ylabel,shapes=[],annotations=[];
if(kind==='loss'){x=Array.from({length:Math.min(cursor+1,f.evaluation_count)},(_,i)=>i);y=f.objectives.slice(0,x.length);all=f.objectives;title='Recorded learning objective - window '+f.window;xlabel='Local loss-evaluation index';ylabel='Objective (log scale)';shapes=[{type:'line',x0:f.adam_count,x1:f.adam_count,y0:0,y1:1,yref:'paper',line:{color:'#8b9aa4',dash:'dash',width:1}}];annotations=[{x:f.adam_count,y:1,yref:'paper',text:'L-BFGS begins',showarrow:false,xanchor:'left',font:{size:11}}];$('historyCaption').textContent='Teal = recorded loss. Dashed boundary = switch from causal Adam objective to raw residual L-BFGS objective. These objectives differ; a jump does not necessarily mean worse predictions.';
}else if(kind==='history'){x=f.history_t.filter(t=>t<=cursor);y=f.history_error.slice(0,x.length);all=f.history_error;title='Recorded reference RMSE - window '+f.window;xlabel='Local loss-evaluation index';ylabel='Prefix state RMSE (log scale)';$('historyCaption').textContent='Only recorded reference samples are shown. The reference is evaluated less often than the loss; no intermediate error values are invented.';
}else{x=f.time;y=f[kind];all=DATA.frames.at(-1)[kind];title=kind==='error'?'Completed-prefix error':'Completed-prefix physics residual';xlabel='Physical time t';ylabel=kind==='error'?'Error norm (log scale)':'Residual norm (log scale)';$('historyCaption').textContent='This curve evaluates the completed checkpoint prefix over physical time. It is not a historical optimizer frame.';}
const logs=all.filter(v=>v>0).map(v=>Math.log10(v));const layout={...base(title),uirevision:kind+'-'+frame,xaxis:{title:{text:xlabel},range:kind==='loss'||kind==='history'?[0,f.evaluation_count]:[0,DATA.time.at(-1)]},yaxis:{title:{text:ylabel},type:'log',range:[Math.floor(Math.min(...logs))-.2,Math.ceil(Math.max(...logs))+.2]},shapes,annotations};await Plotly.react('errorPlot',[{x,y,type:'scatter',mode:kind==='history'?'lines+markers':'lines',marker:{size:4},line:{color:'#197e78',width:2},hovertemplate:'Index/time %{x:.3f}<br>Recorded value %{y:.4g}<extra></extra>'}],layout,config);$('learningInsight').textContent='Learning playback follows actual recorded evaluations in window '+f.window+'. The 3D view stays at the completed solution because intermediate parameter checkpoints are unavailable. '+($('focus').value==='window'?'Selected-window zoom changes axis ranges.':'Full-trajectory axes stay fixed.');}
async function setFrame(value){stop();frame=Math.max(0,Math.min(DATA.frames.length-1,value));cursor=DATA.frames[frame].evaluation_count;windowInfo();if(scene==='layers')await layers();else await Promise.all([geometry(),learning()]);}
DATA.frames.forEach((f,i)=>{const option=document.createElement('option');option.value=i;option.textContent='Window '+f.window+' / 27';$('window').append(option)});
$('window').onchange=()=>setFrame(Number($('window').value));$('back').onclick=()=>setFrame(frame-1);$('next').onclick=()=>setFrame(frame+1);
$('cursor').oninput=()=>{stop();cursor=Number($('cursor').value);learning()};$('duration').onchange=stop;
$('play').onclick=()=>{if(timer){stop();return}if(['error','residual'].includes($('errorKind').value))$('errorKind').value='loss';if(cursor===DATA.frames[frame].evaluation_count)cursor=0;learning();$('play').textContent='Pause';const startCursor=cursor,startTime=performance.now(),durationMs=Number($('duration').value)*1000;timer=setInterval(async()=>{if(busy)return;busy=true;try{cursor=Math.min(DATA.frames[frame].evaluation_count,startCursor+Math.floor(DATA.frames[frame].evaluation_count*(performance.now()-startTime)/durationMs));await learning();if(cursor===DATA.frames[frame].evaluation_count)stop()}finally{busy=false}},500)};
$('focus').onchange=()=>{stop();geometry();learning()};$('showReference').onchange=geometry;$('showPrediction').onchange=geometry;$('resetCamera').onclick=()=>Plotly.relayout('trajectory',{'scene.camera':camera});$('errorKind').onchange=()=>{stop();learning()};


let currentCase;

async function reference(){

 currentCase=DATA.cases.find(v=>v.k===Number($('k').value)&&v.key===$('initial').value);const c=currentCase;

 $('coefficients').textContent='k = '+c.k+', l = 1 · A = '+c.c[0].toFixed(4)+', B = '+c.c[1].toFixed(4)+', C = '+c.c[2].toFixed(4)+' · initial state ('+c.ic.join(', ')+')';

 $('caseExplanation').textContent=c.explanation;

 const colors=['#197e78','#c57536','#74609d'];

 const family=DATA.cases.filter(v=>v.key===c.key).flatMap(v=>v.xyz);

 const familyBounds=[0,1,2].map(i=>{const v=family.map(p=>p[i]),lo=Math.min(...v),hi=Math.max(...v),pad=Math.max((hi-lo)*.08,.005);return [lo-pad,hi+pad]});

 const component=$('component').value,shown=component==='all'?[0,1,2]:[Number(component)];

 const lines=shown.map(i=>({x:c.t,y:c.xyz.map(p=>p[i]),name:['x','y','z'][i],type:'scatter',mode:'lines',line:{color:colors[i],width:2.5}}));

 const stateBounds=[Math.min(...shown.map(i=>familyBounds[i][0])),Math.max(...shown.map(i=>familyBounds[i][1]))];

 await Promise.all([Plotly.react('reference3d',[curve(c.xyz,'ODE reference (teal)','#197e78',5),{type:'scatter3d',mode:'markers',...xyz([c.xyz[0]]),marker:{size:5,color:'#c57536'},name:'Selected physical time (orange)',showlegend:true}],{...space('',familyBounds),uirevision:'reference-'+c.key,showlegend:true,legend:{orientation:'h',x:0,y:1}},config),Plotly.react('referenceTime',lines,{...base('State values over physical time'),uirevision:c.key+'-'+component,xaxis:{title:{text:'Physical time t'},range:[0,10]},yaxis:{title:{text:'State value'},range:stateBounds},legend:{orientation:'h',y:-.25,x:0},margin:{l:58,r:22,t:50,b:80},shapes:[]},config)]);

 $('referenceEvidence').textContent='Computed separately with the existing DOP853 reference solver, rtol 1e−10 / atol 1e−12. Maximum absolute drift of two quadratic invariants: '+c.drift.toExponential(2)+'. No PINN training.';

 await markTime();

}

async function markTime(){const i=Number($('physicalTime').value),c=currentCase;if(!c)return;$('timeLabel').textContent='t = '+c.t[i].toFixed(2);await Promise.all([Plotly.restyle('reference3d',{x:[[c.xyz[i][0]]],y:[[c.xyz[i][1]]],z:[[c.xyz[i][2]]]},[1]),Plotly.relayout('referenceTime',{shapes:[{type:'line',x0:c.t[i],x1:c.t[i],y0:0,y1:1,yref:'paper',line:{color:'#9aabb3',width:1,dash:'dot'}}]})])}

$('k').onchange=reference;$('initial').onchange=reference;$('component').onchange=reference;$('physicalTime').oninput=markTime;

async function layers(){
 const idx=Number($('layer').value),layer=DATA.frames[frame].layers[idx];
 const limit=Math.max(1e-10,...DATA.frames.flatMap(f=>f.layers[idx].w.flat()).map(Math.abs));
 const blimit=Math.max(1e-10,...DATA.frames.flatMap(f=>f.layers[idx].b).map(Math.abs));
 $('layerEpoch').textContent='Window '+(frame+1)+' / 27 - final network';
 await Promise.all([Plotly.react('weights',[{type:'heatmap',z:layer.w,zmin:-limit,zmax:limit,colorscale:[[0,'#be6934'],[.5,'#f2f2ef'],[1,'#197e78']],colorbar:{thickness:12},hovertemplate:'Source %{x}<br>Receiver %{y}<br>Weight %{z:.4g}<extra></extra>'}],{...base('Final weight matrix'),xaxis:{title:{text:'Source neuron (zero-based)'}},yaxis:{title:{text:'Receiving neuron (zero-based)'},autorange:'reversed',dtick:idx===4?1:10}},config),Plotly.react('biases',[{type:'bar',x:layer.b.map((_,i)=>i),y:layer.b,marker:{color:'#197e78'}}],{...base('Final bias vector'),xaxis:{title:{text:'Receiving neuron (zero-based)'}},yaxis:{range:[-blimit*1.1,blimit*1.1],title:{text:'Bias'}}},config)]);
 $('layerNote').textContent='These are final parameters of a separate network for window '+(frame+1)+'. The color scale is shared across all windows for this layer. Parameter magnitude alone does not measure output importance.';
}
$('layer').onchange=layers;

async function switchScene(button){stop();scene=button.dataset.scene;document.querySelectorAll('.scene').forEach(s=>s.hidden=s.id!==scene);document.querySelectorAll('[data-scene]').forEach(b=>b.setAttribute('aria-selected',b===button));$('windowControls').hidden=scene==='reference';$('learningControls').hidden=scene!=='learning';$('referenceControls').hidden=scene!=='reference';$('layerControls').hidden=scene!=='layers';document.body.classList.remove('controls-open');$('sidebarOpen').setAttribute('aria-expanded','false');if(scene==='reference')await reference();else if(scene==='layers')await layers();else await Promise.all([geometry(),learning()]);}
document.querySelectorAll('[data-scene]').forEach(button=>button.onclick=()=>switchScene(button));
$('clean').onclick=()=>{const on=document.body.classList.toggle('presentation');$('clean').setAttribute('aria-pressed',on);$('clean').textContent=on?'Show evidence details':'Presentation mode'};
$('sidebarOpen').onclick=()=>{const on=document.body.classList.toggle('controls-open');$('sidebarOpen').setAttribute('aria-expanded',on)};$('sidebarClose').onclick=()=>{document.body.classList.remove('controls-open');$('sidebarOpen').setAttribute('aria-expanded','false')};
$('finalMetric').textContent='Final full-run checkpoint RMSE: '+DATA.finalRmse.toExponential(3)+'. Recorded RMSE above is the last available training sample.';
$('sampling').textContent='All 67,908 recorded loss evaluations are included across 27 windows. Whole-trajectory view uses 566 probes; selected-window view uses dense checkpoint geometry. Intermediate parameter snapshots are unavailable locally.';
$('provenance').textContent='Source: '+DATA.source+' | final RMSE '+DATA.finalRmse.toExponential(6)+' | checkpoint '+DATA.checkpointSha256.slice(0,12)+' | no retraining';
const observer=new ResizeObserver(entries=>{for(const entry of entries){const el=entry.target;if(el.classList.contains('js-plotly-plot')&&el.offsetWidth)requestAnimationFrame(()=>Plotly.Plots.resize(el))}});document.querySelectorAll('.plot').forEach(el=>observer.observe(el));
windowInfo();geometry();learning();
