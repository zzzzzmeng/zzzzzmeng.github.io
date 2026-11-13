'use strict';
const $ = id => document.getElementById(id);
const palette = ['#a5f4c3','#c7baff','#ffd087','#a9d5ff'];
let mode=0, scenario=0, step=0, gpu=0, timer=null;
const current = () => lessons[mode].cases[scenario].steps;
let previousDiagram='';
function node(tag,classes,value) {
  const el=document.createElement(tag);
  if(classes) el.className=classes;
  if(value!=null) el.textContent=value;
  return el;
}
function renderMatrix(m) {
  const box=node('section','matrix');
  const header=node('header');
  header.append(node('b','',m.name),node('span','shape','['+m.values.length+'×'+m.values[0].length+']'));
  const cells=node('div','cells');
  cells.style.gridTemplateColumns='repeat('+m.values[0].length+',auto)';
  m.values.forEach((row,i)=>row.forEach((v,j)=>{
    const c=typeof m.color==='function'?m.color(i,j):m.color;
    const cell=node('button','cell'+(c==null?'':' c'+c),Number(v.toFixed(3)));
    cell.title='行 '+i+' · 列 '+j+(c==null?'':' · GPU '+c);
    cell.setAttribute('aria-label',m.name+' 行 '+i+' 列 '+j+' 值 '+Number(v.toFixed(3)));
    cell.onclick=()=>{
      $('scalar').replaceChildren(node('strong','',m.name+'['+i+','+j+']'),
        node('code','formula','值 = '+v.toFixed(6)),
        node('p','',m.note||'按当前矩阵的行列定义读取。该元素属于本帧显示的张量。'));
    };
    cells.append(cell);
  }));
  box.append(header,cells);
  if(m.note)box.append(node('small','',m.note));
  return box;
}
function renderTiles(rows){
  const block=node('div','tile-block');
  for(const row of rows){
    const line=node('div','tile-row');
    for(const t of row)line.append(node('span','tile'+(t.owner==null?'':' c'+t.owner),t.label));
    block.append(line);
  }
  return block;
}
function renderSplit(d){
  const panel=node('div','split-example');
  const heading=node('div','split-example-title');
  heading.append(node('strong','',d.title),node('span','shape',d.axis));
  const origin=node('div','split-origin');
  origin.append(node('span','shape','原始布局'),renderTiles(d.source));
  const shards=node('div','split-shards');
  d.parts.forEach((rows,i)=>{
    const b=node('button','split-device c'+i);
    b.dataset.gpu=i;
    b.append(node('span','gpu-id','GPU '+i),renderTiles(rows),node('small','',d.notes[i]));
    b.onclick=()=>{gpu=i;render();};
    shards.append(b);
  });
  const flow=node('div','split-flow');
  const arrow=node('span','split-arrow','→');
  arrow.setAttribute('aria-hidden','true');
  flow.append(origin,arrow,shards);
  panel.append(heading,flow);
  return panel;
}
function renderExecution(l,s){
  const meta=impl[l.label];
  $('source-title').textContent=l.name+' · 切分、执行与梯度归属';
  $('contract').replaceChildren(...[
    ['切分对象',meta.cut],['张量布局',meta.shape],['本示例配置',meta.boundary]
  ].map(([title,text])=>{
    const row=node('div','contract-row');
    row.append(node('strong','',title),node('span','',text));
    return row;
  }));
  $('source-route').textContent='前向 '+meta.forward+' / 反向 '+meta.backward;
  $('source-code').textContent=meta.code;
  $('source-link').href='https://github.com/NVIDIA/Megatron-LM/blob/'+revision+'/megatron/core/'+meta.file+'#L'+meta.line;
  $('source-foot').textContent=meta.foot;
  $('execution').replaceChildren(...s.roles.map((role,i)=>{
    const card=node('div','exec-card'+(gpu===i?' selected':''));
    card.style.setProperty('--gpu',palette[i]);
    card.append(node('span','gpu-id','GPU '+i),node('strong','',role),node('code','formula',s.formula));
    return card;
  }),node('p','exec-detail',s.detail),node('p','exec-comm',s.comm||'设备内本地计算；该步骤无需跨卡传输。'));
  const backward=s.phase==='反向';
  $('scalar').replaceChildren(node('strong','',backward?'梯度如何得到':'矩阵乘法怎样得到一个元素'));
  if(l.label==='TP'&&scenario===0){
    const examples=backward?
      ['行切反向 dA₀[0,0] = 1×1 + 1×0 = 1',
       '列切反向 dX₀[0,0] = 1×1 + 1×2 = 3',
       '输入梯度求和 dX[0,1] = 1 + 2 = 3']:
      ['升维 Z₀[0,1] = 1×2 + 2×1 = 4',
       '激活 A₁[0,1] = ReLU(−1) = 0',
       '降维 Y₁[0,0] = 2×1 + 0×0 = 2',
       '输出求和 Y[0,0] = 1 + 2 = 3'];
    $('scalar').append(node('code','formula',examples[backward?Math.min(step-4,2):Math.min(step,3)]));
  }else{
    $('scalar').append(node('code','formula',s.formula),node('p','',backward?
      'dX 传给前一计算，dW 累积到本地参数。读下方源码时区分输入梯度的归约与参数梯度的同步。':
      '输出中的不同位置应拼接；同一位置的不同计算贡献应求和。具体操作看当前通信条。'));
  }
  $('pipeline').replaceChildren();
  if(l.label==='PP'&&scenario===1){
    const table=node('table','timeline');
    const top=node('tr');top.append(node('th','','阶段'),...schedule.map((_,i)=>node('th','',i+1)));table.append(top);
    [0,1].forEach(rank=>{
      const row=node('tr');
      row.append(node('th','','GPU '+rank),...schedule.map((slice,i)=>node('td',(i===step?'active ':'')+(slice[rank].startsWith('B')?'bwd':'fwd'),slice[rank])));
      table.append(row);
    });
    $('pipeline').append(table,node('p','axis-guide','F 是前向，B 是反向，下标为 microbatch。此图展示依赖顺序，不模拟实际计算耗时。'));
  }
}
function stop(){
  clearInterval(timer);
  timer=null;
}
function jump(i,focus){
  stop();
  step=Math.min(current().length-1,Math.max(0,i));
  render();
  if(focus) $('steps').children[step].focus();
}
function render(){
  const l=lessons[mode], s=current()[step];
  [...$('modes').children].forEach((btn,i)=>btn.setAttribute('aria-pressed',i===mode));
  $('tag').textContent=l.label+' / '+l.name;
  $('name').textContent=s.title;
  $('overview').textContent=l.overview;
  renderExecution(l,s);
  $('caseLabel').hidden=l.cases.length<2;
  const diagramKey=l.label+':'+scenario;
  if(previousDiagram!==diagramKey){
    $('splits').replaceChildren(...splitDiagrams[l.label][scenario].map(renderSplit));
    previousDiagram=diagramKey;
  }
  for(const el of $('splits').querySelectorAll('[data-gpu]')){
    el.setAttribute('aria-pressed',Number(el.dataset.gpu)===gpu);
  }
  $('phase').textContent=s.phase;
  $('phase').dataset.phase=s.phase;
  $('forward').setAttribute('aria-pressed',s.phase==='前向');
  $('backward').setAttribute('aria-pressed',s.phase==='反向');
  $('matrices').replaceChildren(...s.matrices.map(renderMatrix));
  $('wire').replaceChildren(...(s.comm?[node('i','signal'),node('span','',s.comm)]:[node('span','shape','本步骤仅做本地计算')]));
  $('wire').classList.toggle('live',Boolean(s.comm));
  $('cards').replaceChildren(...s.roles.map((role,i)=>{
    const b=node('button','gpu');
    b.style.setProperty('--gpu',palette[i]);
    b.setAttribute('aria-pressed',i===gpu);
    b.append(node('span','gpu-id','GPU '+i),node('strong','',role));
    b.onclick=()=>{gpu=i;render();$('cards').children[i].focus();};
    return b;
  }));
  $('steps').replaceChildren(...current().map((f,i)=>{
    const b=node('button','step');
    if(i===step)b.setAttribute('aria-current','step');
    b.append(node('span','',String(i+1).padStart(2,'0')),node('strong','',f.title),node('small','step-phase',f.phase));
    b.onclick=()=>jump(i,true);
    return b;
  }));
  $('inspector').replaceChildren(node('div','eyebrow','GPU '+gpu+' / 当前分工'),node('h3','',s.roles[gpu]),node('p','',s.detail),node('code','formula',s.formula));
  $('prev').disabled=step===0;
  $('next').disabled=step===current().length-1;
  $('play').textContent=timer?'Ⅱ 暂停':'▶ 播放';
  $('counter').textContent=(step+1)+' / '+current().length;
  $('seek').max=current().length-1;
  $('seek').value=step;
}
function loadMode(i){
  stop();
  mode=i;scenario=0;step=0;gpu=0;
  $('cases').replaceChildren(...lessons[i].cases.map((c,k)=>{
    const o=node('option','',c.label);o.value=k;return o;
  }));
  render();
}
lessons.forEach((l,i)=>{
  const b=node('button','',l.label+' '+l.name);
  b.onclick=()=>loadMode(i);
  $('modes').append(b);
});
$('cases').onchange=e=>{scenario=Number(e.target.value);jump(0);};
$('forward').onclick=()=>jump(0);
$('backward').onclick=()=>jump(current().findIndex(s=>s.phase==='反向'));
$('prev').onclick=()=>jump(step-1);
$('next').onclick=()=>jump(step+1);
$('reset').onclick=()=>jump(0);
$('seek').oninput=e=>jump(Number(e.target.value));
$('play').onclick=()=>{
  if(timer){stop();render();return;}
  if(step===current().length-1)step=0;
  timer=setInterval(()=>{
    step++;
    if(step===current().length-1)stop();
    render();
  },1800);
  render();
};
loadMode(0);
