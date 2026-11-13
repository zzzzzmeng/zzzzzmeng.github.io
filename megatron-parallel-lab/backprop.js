'use strict';
const replaceTail = (steps,start,...frames) => steps.splice(start,steps.length-start,...frames);
const back = (title,detail,formula,matrices,roles,comm='') => frame(title,detail,formula,matrices,roles,comm,'反向');
const softmax = row => {
  const exps=row.map(v=>Math.exp(v-Math.max(...row)));
  return exps.map(v=>v/exps.reduce((a,b)=>a+b,0));
};

// Attention uses ordinary MHA, one feature per head and a causal mask.
// Derive all frames from the same QKV and W_o so forward and backward agree.
const headWo=[[[1,2]],[[3,1]]];
const attn=ids.map(i=>{
  const qkv=cols(X,i,i+1);
  const p=mm(qkv,transpose(qkv)).map((r,j)=>softmax(r.map((v,k)=>k<=j?v:-Infinity)));
  const a=mm(p,qkv), wo=headWo[i], da=mm(dY,transpose(wo));
  const dv=mm(transpose(p),da), dp=mm(da,transpose(qkv));
  const ds=p.map((r,j)=>{
    const mean=r.reduce((sum,w,k)=>sum+w*dp[j][k],0);
    return r.map((w,k)=>w*(dp[j][k]-mean));
  });
  const dq=mm(ds,qkv), dk=mm(transpose(ds),qkv);
  const dx=add(add(dq,dk),dv).map(([v])=>i===0?[v,0]:[0,v]);
  return {p,a,dv,ds,dq,dk,da,dx,y:mm(a,wo),dw:mm(transpose(a),dY)};
});
const attnValues = key => attn.map(h=>h[key]);
const attnY=add(...attnValues('y')), attnDX=add(...attnValues('dx'));
replaceTail(heads,1,
  frame('各卡本地算注意力','第一行只能看第一个 token，第二行对两个位置做 softmax。每张卡都拥有本地头的完整序列。','Aᵢ = softmax(mask(QᵢKᵢᵀ)) × Vᵢ',[...pair('P',attnValues('p')),...pair('A',attnValues('a'))],['头 0：两个 token','头 1：两个 token']),
  frame('输出投影求和','各卡把自己的头输出乘 Wₒ 中对应的行，得到两个隐藏特征的部分和。','Y = A₀Wₒ₀ + A₁Wₒ₁',[matrix('Wₒ',[[1,2],[3,1]],i=>i),...pair('Y',attnValues('y')),matrix('Y',attnY)],['头 0 贡献完整输出','头 1 贡献完整输出'],'GPU 0 ⇄ GPU 1 · All-reduce SUM · Y'),
  back('输出投影把梯度送回各头','取损失为 Y 所有元素之和。两卡用同一份 dY，各自计算本地头的 dA 和输出投影权重梯度。','dAᵢ = dY × Wₒᵢᵀ；dWₒᵢ = Aᵢᵀ × dY',[matrix('dY',dY),...pair('dA',attnValues('da')),...pair('dWₒ · ',attnValues('dw'))],['反传到头 0','反传到头 1']),
  back('各头本地反传到 Q、K、V','先反传 A=P×V，得到 dV 和 dP。dP 再经过 softmax 得到打分梯度 dS，最后由 QKᵀ 的两条分支算出 dQ、dK。因果屏蔽位置的 dS 为 0。','dV=PᵀdA；dP=dAVᵀ；dQ=dSK；dK=dSᵀQ',[...pair('dV',attnValues('dv')),...pair('dS',attnValues('ds')),...pair('dQ',attnValues('dq')),...pair('dK',attnValues('dk'))],['头 0 的 QKV 梯度','头 1 的 QKV 梯度']),
  back('QKV 投影合并输入梯度','每卡将三路 QKV 对输入的贡献加起来，再把两卡对同一个 X 的贡献求和。示例权重为单位阵，所以每卡只给一个输入特征贡献梯度。','dX = Σᵢ(dQᵢWqᵢᵀ + dKᵢWkᵢᵀ + dVᵢWvᵢᵀ)',[...pair('dX',attnValues('dx')),matrix('完整 dX',attnDX)],['拿到完整 dX','拿到完整 dX'],'GPU 0 ⇄ GPU 1 · All-reduce SUM · dX')
);

const dpDY=dpY.map((y,i)=>y.map(([v],j)=>[v-targets[i][j][0]]));
replaceTail(dp,2,
  back('由本地损失算出 dY','本地损失取平方误差之和的一半。每个样本的梯度就是预测减目标。','dYᵢ = Yᵢ − 目标ᵢ',pair('dY',dpDY),['本地样本 0–1','本地样本 2–3']),
  back('各卡计算本地参数梯度','两卡算的是同一份权重对应位置的梯度。数值不同，因为用的是不同样本。','dWᵢ = Xᵢᵀ × dYᵢ',pair('dW',dpGrad),['本地 dW₀','本地 dW₁']),
  back('同步到相同的全局梯度','本例全局损失取两个本地损失的平均。参数梯度先求和，再除以 2。','dW = (dW₀ + dW₁) / 2',[...pair('dW',dpGrad),matrix('平均 dW',mean)],['获得全局 dW','获得全局 dW'],'GPU 0 ⇄ GPU 1 · All-reduce · 参数梯度'),
  frame('用相同梯度更新','优化器现在才改权重。两卡用相同的梯度和学习率，因此得到相同的新权重。','W ← W − 0.1 dW',[matrix('旧 W',dpW),matrix('新 W',dpW.map((r,i)=>[r[0]-.1*mean[i][0]]))],['更新副本 0','更新副本 1'],'','更新')
);

const stageWeights=[[[1,0],[0,2]],[[2,1],[0,1]]];
const micros=[[[1,2]],[[2,1]]];
const ppDY=[[1,1]], ppDA=mm(ppDY,transpose(stageWeights[1]));
const stageGrads=micros.map(x=>[mm(transpose(x),ppDA),mm(transpose(mm(x,stageWeights[0])),ppDY)]);
const ppGradTotal=ids.map(i=>add(stageGrads[0][i],stageGrads[1][i]));
replaceTail(pp,4,...[1,0].flatMap(i=>[
  back('μ'+i+'：后段算出边界梯度','所有前向结束后，本演示依次反传 μ1、μ0。后段算本地权重梯度，然后把 dA 送回前段。','dW₁=AᵀdY；dA=dYW₁ᵀ',[shard('dY',ppDY,1),shard('本 microbatch 的 dW₁',stageGrads[i][1],1),shard('回传 dA',ppDA,0)],['接收 μ'+i+' 的 dA','反传 μ'+i+' 的后段'],'GPU 1 → GPU 0 · P2P · μ'+i+' 的 dA'),
  back('μ'+i+'：前段继续反传','前段取回与该 microbatch 对应的输入，计算本地权重梯度及输入梯度。','dW₀=XᵀdA；dX=dAW₀ᵀ',[shard('本 microbatch 的 dW₀',stageGrads[i][0],0),shard('dX',mm(ppDA,transpose(stageWeights[0])),0)],['反传 μ'+i+' 的前段','已算完本 microbatch'])
]),back('按阶段累积参数梯度','两个 microbatch 的梯度都已传完，各阶段只累加自己持有的权重梯度。','dW(阶段) = dW(μ0) + dW(μ1)',pair('阶段 dW',ppGradTotal),['保留阶段 0 的 dW','保留阶段 1 的 dW']));

const cpK=[[0],[0],[0],[0]], cpV=[[1],[2],[3],[4]], cpP=[[.25,.25,.25,.25]];
const cpDScore=[[-.375,-.125,.125,.375]], cpDV=transpose(cpP), cpDK=transpose(cpDScore);
replaceTail(cp,0,
  frame('按 token 位置切序列','两卡处理同一条序列的不同位置。用一维的 K/V 演示，重点跟踪位置 3 的一个查询。','GPU 0：t₀,t₁；GPU 1：t₂,t₃',[matrix('K',cpK,i=>split(i)),matrix('V',cpV,i=>split(i))],['位置 0–1 的 K/V','位置 2–3 的 K/V']),
  frame('位置 3 先读本地 KV','令位置 3 的 Q=1。它能看见自己和之前的三个位置，先取自己设备上的位置 2、3。','本地 scores = 1 × [0,0]',[shard('本地 K',cpK.slice(2),1),shard('本地 V',cpV.slice(2),1)],['负责位置 0–1','读取位置 2–3']),
  frame('读取远端 KV','要完成位置 3 的注意力，还需要位置 0、1 的 K 和 V。图中只追踪这个查询的依赖方向。','远端 scores = 1 × [0,0]',[shard('远端 K',cpK.slice(0,2),0),shard('远端 V',cpV.slice(0,2),0)],['提供位置 0–1 的 KV','读取位置 0–1 的 KV'],'GPU 0 → GPU 1 · KV 块'),
  frame('在所有可见位置归一化','四个打分都是 0，所以权重各为 0.25。块式实现也要得到同一个跨序列 softmax。','A₃ = 0.25 × (1+2+3+4) = 2.5',[matrix('位置 3 的 P',cpP,(_,j)=>split(j)),shard('位置 3 的输出',[[2.5]],1)],['保留本地位置的输出','位置 3 的输出为 2.5']),
  back('位置 3 生成四份 KV 梯度','只给位置 3 一个上游梯度 dA=1。dV 是注意力权重，dscore 是 softmax 反向的结果；此处 Q=1，所以 dK=dscore，所有 K 为 0，所以 dQ=0。','dV=Pᵀ；dscore=P⊙(Vᵀ−2.5)',[matrix('对四个位置的 dV',cpDV,i=>split(i)),matrix('对四个位置的 dK',cpDK,i=>split(i))],['等待远端位置的贡献','计算本查询的所有贡献']),
  back('梯度回到 KV 所属卡','把位置 0、1 的贡献还给 GPU 0。若还有其他查询用到了同一 KV，所属卡把所有贡献累加后再反传本地投影。','dKV(位置) = Σ 各查询的贡献',[...pair('dV · GPU ',[cpDV.slice(0,2),cpDV.slice(2)]),...pair('dK · GPU ',[cpDK.slice(0,2),cpDK.slice(2)])],['累加位置 0–1 的 dKV','保留位置 2–3 的 dKV'],'GPU 1 → GPU 0 · 远端 dK / dV')
);

replaceTail(ep,4,
  back('把输出梯度送回专家','令每个输出收到 dY=1。前向 combine 的逆向把梯度送回当时执行计算的专家。','dY(t₁,t₂) → E₀；dY(t₀,t₃) → E₁',[shard('E₀ 收到 dY',[[1],[1]],0,'t₁、t₂'),shard('E₁ 收到 dY',[[1],[1]],1,'t₀、t₃')],['收到 t₁、t₂ 的 dY','收到 t₀、t₃ 的 dY'],'GPU 0 ⇄ GPU 1 · All-to-all · 专家输出梯度'),
  back('各专家在本地反传','简化专家 y=w×x。E₀ 的参数梯度是 2+3=5，E₁ 是 1+4=5。对每个输入的梯度则等于该专家的权重。','dW=Σ(x×dY)；dX=w×dY',[...pair('专家 dW',[[[5]],[[5]]]),...pair('专家内 dX',[[[2],[2]],[[3],[3]]])],['E₀ 算参数梯度和 dX','E₁ 算参数梯度和 dX']),
  back('把输入梯度送回 token 原位','按 dispatch 的逆向交换各自处理的 dX，再还原输入 token 顺序。参数梯度留在专家所属卡。','还原 t₀,t₁,t₂,t₃ 的顺序',[matrix('输入 dX',[[3],[2],[2],[3]],i=>split(i))],['t₀、t₁ 的 dX','t₂、t₃ 的 dX'],'GPU 0 ⇄ GPU 1 · All-to-all · token 输入梯度')
);

replaceTail(sp,0,
  frame('层边界沿序列切','用与 TP MLP 相同的输入和权重。边界上每张卡只保留一行，但这一行的隐藏特征完整。','X₀=X[0:1]；X₁=X[1:2]',[matrix('X',X,i=>i),...pair('本地 X',X.map(r=>[r]))],['token 0 / 完整特征','token 1 / 完整特征']),
  frame('列切前汇集序列','每张卡持有部分中间特征的权重，所以都要对两个 token 计算自己的特征。先按序列行拼接 X。','完整 X = concat(X₀,X₁)',[matrix('两卡都收到 X',X),...pair('本地 A',A)],['所有 token / 特征 0–1','所有 token / 特征 2–3'],'GPU 0 ⇄ GPU 1 · All-gather · 输入激活'),
  frame('行切后求和并分回序列','行切结果都是 [2×2] 的部分和。归约后 GPU 0 只保留第 0 行，GPU 1 只保留第 1 行。','Y(本地) = (Y₀ + Y₁)[本地 token]',[...pair('部分 Y',Y),...pair('输出行 ',fullY.map(r=>[r]))],['保留 token 0','保留 token 1'],'GPU 0 ⇄ GPU 1 · Reduce-scatter · 输出激活'),
  back('行切反向先汇集 dY','边界上每卡先拿一行上游梯度。行切权重在前向处理了两个 token，反向需要两行 dY，所以先拼接。','完整 dY = concat(dY₀,dY₁)',[...pair('上游行 ',dY.map(r=>[r])),matrix('两卡都收到 dY',dY)],['得到所有 token 的 dY','得到所有 token 的 dY'],'GPU 0 ⇄ GPU 1 · All-gather · dY'),
  back('反传本地激活与权重','用完整 dY 算本地降维梯度，经过 ReLU 后回到升维层。算升维权重梯度需要原输入 X，实际实现可能再次汇集保存的输入行。','dAᵢ=dY W₂ᵢᵀ；dW₁ᵢ=XᵀdZᵢ',[...pair('dZ',dZ),...pair('dW₁ · ',dZ.map(z=>mm(transpose(X),z)))],['求本地特征的梯度','求本地特征的梯度']),
  back('dX 求和后只留本地行','升维层给每个 token 的输入都贡献梯度。按对应位置求和后，再分回序列边界，恢复每卡一行。','dX(本地) = (dX₀ + dX₁)[本地 token]',[...pair('部分 dX',dx),...pair('输入梯度行 ',add(...dx).map(r=>[r]))],['收到 token 0 的 dX','收到 token 1 的 dX'],'GPU 0 ⇄ GPU 1 · Reduce-scatter · dX')
);
