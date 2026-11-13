'use strict';
const transpose = a => a[0].map((_,i)=>a.map(r=>r[i]));
const mm = (a,b) => a.map(r=>transpose(b).map(c=>r.reduce((v,x,i)=>v+x*c[i],0)));
const add = (a,b) => a.map((r,i)=>r.map((x,j)=>x+b[i][j]));
const cols = (a,i,j) => a.map(r=>r.slice(i,j));
const matrix = (name,values,color=null,note='') => ({name,values,color,note});
const frame = (title,detail,formula,matrices,roles,comm='',phase='前向') => ({title,detail,formula,matrices,roles,comm,phase});
const lesson = (label,name,overview,cases) => ({label,name,overview,cases});
const example = (label,steps) => ({label,steps});
const shard = (name,values,color,note) => matrix(name,values,color,note);
const X = [[1,2],[2,1]], W1 = [[1,2,0,1],[0,1,1,-1]], W2 = [[1,0],[0,1],[1,1],[0,1]];
const ids = [0,1], split = j => j<2?0:1;
const Wup = ids.map(i=>cols(W1,i*2,i*2+2)), Wdown=ids.map(i=>W2.slice(i*2,i*2+2));
const Z = Wup.map(w=>mm(X,w)), A=Z.map(z=>z.map(r=>r.map(x=>Math.max(x,0))));
const Y = A.map((a,i)=>mm(a,Wdown[i])), fullY=add(...Y);
const dY = [[1,1],[1,1]], dA = Wdown.map(w=>mm(dY,transpose(w)));
const dZ = Z.map((z,i)=>z.map((r,t)=>r.map((x,k)=>x>0?dA[i][t][k]:0)));
const dx = dZ.map((z,i)=>mm(z,transpose(Wup[i])));
const pair = (name,values,note='') => values.map((v,i)=>shard(name+i,v,i,note));
const tp = [
 frame('第一层按列切','X 的两行是两个 token。每张卡都读取完整 X，但只持有 W₁ 的两列。','Zᵢ = X × W₁ᵢ',[matrix('X',X,null,'两卡复制'),matrix('W₁',W1,(_,j)=>split(j),'列 = 中间特征'),...pair('Z',Z)],['W₁ 的第 0–1 列','W₁ 的第 2–3 列']),
 frame('本地做激活','ReLU 把负数置零。每个元素可以独立计算，分片继续留在各卡。','Aᵢ = max(Zᵢ, 0)',[...pair('Z',Z),...pair('A',A)],['激活特征 0–1','激活特征 2–3']),
 frame('第二层按行切','W₂ 的行与中间特征一一对应。两卡分别计算相同两个输出特征的一部分。','Yᵢ = Aᵢ × W₂ᵢ',[matrix('W₂',W2,i=>split(i),'行 = 中间特征'),...pair('Y',Y)],['W₂ 的第 0–1 行','W₂ 的第 2–3 行']),
 frame('合并前向输出','Y₀ 和 Y₁ 的列代表同一组输出特征，逐元素求和后两张卡都得到完整 Y。','Y = Y₀ + Y₁',[...pair('Y',Y),matrix('Y',fullY,null,'两卡都得到完整输出')],['收到完整 Y','收到完整 Y'],'GPU 0 ⇄ GPU 1 · All-reduce SUM · Y [2×2]'),
 frame('行切层的反向','取损失为 Y 所有元素之和。两卡收到同一份 dY，各自计算自己的权重梯度和 dA。','dW₂ᵢ = Aᵢᵀ × dY；dAᵢ = dY × W₂ᵢᵀ',[matrix('dY',dY),...pair('dA',dA),...pair('dW₂·',A.map(a=>mm(transpose(a),dY)))],['本地 dW₂₀ 与 dA₀','本地 dW₂₁ 与 dA₁'],'','反向'),
 frame('激活与列切层反向','ReLU 在负数位置阻断梯度。各卡的 dW₁ 是完整的参数分片梯度，dX 则只是对同一个输入的贡献。','dXᵢ = dZᵢ × W₁ᵢᵀ',[...pair('dZ',dZ),...pair('dX',dx),...pair('dW₁·',dZ.map(z=>mm(transpose(X),z)))],['求本地 dW₁₀ 和 dX₀','求本地 dW₁₁ 和 dX₁'],'','反向'),
 frame('合并输入梯度','同一个 X 参与了两个分支的计算。把两份 dX 逐元素相加，前一层才能拿到完整梯度。','dX = dX₀ + dX₁',[...pair('dX',dx),matrix('dX',add(...dx))],['完整 dX 回传','完整 dX 回传'],'GPU 0 ⇄ GPU 1 · All-reduce SUM · dX [2×2]','反向')
];
const heads = [
 frame('QKV 按头切','两卡拿同样的两个 token。示例每个头只有一个特征，Wq、Wk、Wv 都按头切列。','Qᵢ = X × Wqᵢ；K、V 同理',[matrix('X',X),matrix('Wq = Wk = Wv',[[1,0],[0,1]],(_,j)=>j),...pair('Q=K=V · ',[cols(X,0,1),cols(X,1,2)])],['头 0 的 Q、K、V','头 1 的 Q、K、V']),
 frame('各卡本地算注意力','第一行只能看第一个 token。第二行对两个位置做 softmax。每张卡拥有自己那个头的完整序列。','Aᵢ = softmax(mask(QᵢKᵢᵀ)) × Vᵢ',[...pair('P',[[[1,0],[.119,.881]],[[1,0],[.731,.269]]]),...pair('A',[[[1],[1.881]],[[2],[1.731]]])],['头 0：两位置','头 1：两位置']),
 frame('输出投影求和','各卡把自己的头输出乘 Wo 中对应的行，得到两个隐藏特征的部分和。','Y = A₀ × Wo₀ + A₁ × Wo₁',[matrix('Wo',[[1,2],[3,1]],i=>i),...pair('Y',[[[1,2],[1.881,3.762]],[[6,2],[5.193,1.731]]]),matrix('Y',[[7,4],[7.074,5.493]],null,'显示四舍五入后的数值')],['头 0 贡献完整输出','头 1 贡献完整输出'],'GPU 0 ⇄ GPU 1 · All-reduce SUM · Y'),
 frame('输出梯度回到各头','Wo 的反向先把梯度映射回本地头。接着本地反传经过 V、softmax、Q 和 K。','dAᵢ = dY × Woᵢᵀ',[matrix('dY',dY),...pair('dA',[[[3],[3]],[[4],[4]]])],['反传头 0','反传头 1'],'','反向'),
 frame('QKV 输入梯度合并','两组头都依赖同一个输入。各卡先计算 Q、K、V 对 X 的本地梯度，再跨卡相加。','dX = Σᵢ(dQᵢWqᵢᵀ + dKᵢWkᵢᵀ + dVᵢWvᵢᵀ)',[matrix('X',X,null,'同一份输入由两卡共享')],['头 0 的 dX 贡献','头 1 的 dX 贡献'],'GPU 0 ⇄ GPU 1 · All-reduce SUM · dX','反向')
];
const dpW=[[1],[2]], dpXs=[[[1,0],[0,1]],[[2,0],[0,2]]], targets=[[[0],[1]],[[1],[3]]];
const dpY=dpXs.map(x=>mm(x,dpW));
const dpGrad=dpXs.map((x,i)=>mm(transpose(x),dpY[i].map((r,j)=>[r[0]-targets[i][j][0]])));
const mean=add(...dpGrad).map(r=>[r[0]/2]);
const dp=[
 frame('样本分给不同副本','每张卡都持有完整 W。训练 batch 的前两行交给 GPU 0，后两行交给 GPU 1。','X → X₀, X₁；W₀ = W₁ = W',[matrix('W',dpW),matrix('全局 X',dpXs.flat(),i=>split(i)),...pair('本地 X',dpXs)],['样本 0–1 / 完整 W','样本 2–3 / 完整 W']),
 frame('各自计算预测','两个副本用同一套权重，分别为自己收到的样本预测。','Yᵢ = Xᵢ × W',[...pair('Y',dpY),...pair('目标 ',targets)],['算本地损失','算本地损失']),
 frame('得到本地权重梯度','为简化手算，每卡的损失取两个样本平方误差之和的一半。两个副本的梯度位置相同，值可以不同。','dWᵢ = Xᵢᵀ × (Yᵢ − 目标ᵢ)',pair('dW',dpGrad),['样本 0–1 的梯度','样本 2–3 的梯度'],'','反向'),
 frame('同步梯度再更新','对两个本地损失取平均。先将梯度求和再除以 2，两卡更新到同样的新权重。','dW = (dW₀ + dW₁) / 2；W ← W − 0.1 dW',[...pair('dW',dpGrad),matrix('平均 dW',mean),matrix('新 W',dpW.map((r,i)=>[r[0]-.1*mean[i][0]]))],['用相同梯度更新','用相同梯度更新'],'GPU 0 ⇄ GPU 1 · All-reduce · 参数梯度','更新')
];
const pp=[
 frame('按层分配设备','示例用两层线性变换代替两段 Transformer。GPU 0 持有第一段，GPU 1 持有第二段。','f(X) = (X × W₀) × W₁',[shard('W₀',[[1,0],[0,2]],0),shard('W₁',[[2,1],[0,1]],1)],['阶段 0 / 第一段','阶段 1 / 第二段']),
 frame('第一个 microbatch 出发','阶段 0 处理 μ0。它把中间激活交给阶段 1。','[1,2] × W₀ = [1,4]',[shard('μ0 输入',[[1,2]],0),shard('μ0 激活',[[1,4]],0)],['前向 μ0','等待激活'],'GPU 0 → GPU 1 · 点对点 · μ0 激活'),
 frame('两个阶段同时工作','GPU 1 继续算 μ0。此时 GPU 0 开始算 μ1。这里展示一种先填充流水线的简化调度。','μ0：[1,4] × W₁ = [2,5]',[shard('μ1 激活',[[2,2]],0),shard('μ0 输出',[[2,5]],1)],['前向 μ1','前向 μ0'],'GPU 0 → GPU 1 · 点对点 · μ1 激活'),
 frame('阶段 1 继续算 μ1','阶段 1 收到第二份激活并计算输出。阶段 0 在这个时间片没有前向工作。','μ1：[2,2] × W₁ = [4,4]',[shard('μ1 输出',[[4,4]],1)],['暂时空闲','前向 μ1']),
 frame('反向沿原路返回','令 μ0 的输出梯度为 [1,1]。阶段 1 算出激活梯度并发回阶段 0；其他 microbatch 同理。','dA = [1,1] × W₁ᵀ = [3,1]',[shard('dY',[[1,1]],1),shard('dA',[[3,1]],0)],['接收并反传 μ0','发送 μ0 的 dA'],'GPU 1 → GPU 0 · 点对点 · μ0 激活梯度','反向')
];
const cp=[
 frame('按 token 位置切序列','两卡持有相同的投影权重，但 GPU 0 处理位置 0、1，GPU 1 处理位置 2、3。示例为一维 Q/K/V。','Q、K、V 的行 = token 位置',[matrix('Q = K = V',[[1],[2],[3],[4]],i=>split(i))],['位置 0–1 的 Q/K/V','位置 2–3 的 Q/K/V']),
 frame('先用本地 KV 块','位置 3 的 Q=4。对本地 K=[3,4] 的分数是 [12,16]。因果掩码允许它看见全部四个位置。','scores(local) = 4 × [3,4]',[shard('本地 K、V',[[3],[4]],1),shard('本地分数',[[12,16]],1)],['计算位置 0–1','计算位置 2–3']),
 frame('交换所需 KV','位置 3 还要看 GPU 0 的位置 0、1。这个两卡因果例子只画该查询所需的数据方向，实际 ring 可双向调度不同块。','scores(remote) = 4 × [1,2]',[shard('远端 K、V',[[1],[2]],0),shard('远端分数',[[4,8]],1)],['提供位置 0–1 的 KV','保留 Q，读取远端 KV'],'GPU 0 → GPU 1 · KV 块'),
 frame('合并注意力结果','把所有允许的分数纳入同一个 softmax。块式实现可以维护归一化统计量，无需把全矩阵都保存。','softmax([4,8,12,16]) · [1,2,3,4] ≈ 3.981',[matrix('位置 3 的分数',[[4,8,12,16]],(_,j)=>split(j)),shard('位置 3 输出',[[3.981]],1)],['保留自己的输出位置','保留自己的输出位置']),
 frame('反向归还 KV 梯度','同一个远端 K 或 V 可能被多个查询使用。各卡把贡献回传给 KV 所属卡并累加。具体调度由注意力后端决定。','dKV(所属卡) = Σ 查询分片贡献',[matrix('KV 所有权',[[0,0,1,1]],(_,j)=>split(j),'单元格为 GPU 编号')],['累加自己的 dKV','回传远端 dKV 贡献'],'GPU 1 → GPU 0 · 远端 KV 的梯度贡献','反向')
];
const ep=[
 frame('路由器为 token 选专家','用 Top-1 示例。每卡起初有两个 token，路由器为它们选择专家 0 或专家 1。','expert(token) = argmax(router logits)',[matrix('路由分数',[[.1,.9],[.8,.2],[.7,.3],[.2,.8]],(i)=>[1,0,0,1][i],'行是 token 0–3，列是专家 0–1')],['持有 token 0、1 / 专家 0','持有 token 2、3 / 专家 1']),
 frame('把 token 发给专家','专家 0 收到 token 1、2。专家 1 收到 token 0、3。这里每张卡只持有一个专家。','按专家编号重排并发送',[shard('专家 0 收到 ID',[[1],[2]],0),shard('专家 1 收到 ID',[[0],[3]],1)],['处理 token 1、2','处理 token 0、3'],'GPU 0 ⇄ GPU 1 · All-to-all · token 激活'),
 frame('专家独立计算','为方便手算，每个专家用一个缩放代替 MLP。输入 token 的标量分别是 1、2、3、4。','专家 0 ×2；专家 1 ×3',[shard('专家 0 输出',[[4],[6]],0),shard('专家 1 输出',[[3],[12]],1)],['本地专家输出 4、6','本地专家输出 3、12']),
 frame('输出回到原始位置','将结果送回 token 原来所在的设备，再恢复顺序。本例省略门控权重。','按 token ID 还原输出',[matrix('全局输出',[[3],[4],[6],[12]],i=>split(i))],['恢复 token 0、1','恢复 token 2、3'],'GPU 0 ⇄ GPU 1 · All-to-all · 专家输出'),
 frame('梯度沿路由反向流动','输出梯度先送回专家，各专家算本地参数梯度，再把输入梯度送回原 token 的位置。此处只展示专家计算分支。','d输出 → 专家反向 → d输入',[matrix('每个 token 的 d输入',[[3],[2],[2],[3]],i=>split(i),'假设每个输出的上游梯度为 1')],['原位置收到 d输入','原位置收到 d输入'],'两次 All-to-all · 送 d输出，返回 d输入','反向')
];
const sp=[
 frame('层边界沿序列切','SP 与 TP 使用同一组卡。在归一化等层边界，每卡只保留部分 token 的完整隐藏特征。','X₀ = X[0:2]；X₁ = X[2:4]',[matrix('X',[[1,2],[2,1],[3,2],[2,3]],i=>split(i))],['位置 0–1','位置 2–3']),
 frame('列切前汇集序列','列切线性层的每个权重分片都要为所有 token 计算自己负责的输出特征。','沿 token 行拼成完整 X',[matrix('完整 X',[[1,2],[2,1],[3,2],[2,3]],i=>split(i))],['收到完整序列，算本地特征','收到完整序列，算本地特征'],'GPU 0 ⇄ GPU 1 · All-gather · 激活'),
 frame('行切后求和并分回序列','每卡的部分和覆盖所有 token。Reduce-scatter 对同位置求和，并只把对应行留给各卡。','Yᵢ = (Y₀(partial) + Y₁(partial))[本地行]',[shard('局部和 0',[[1,0],[2,0],[3,0],[4,0]],0),shard('局部和 1',[[0,1],[0,2],[0,3],[0,4]],1),matrix('归约后的 Y',[[1,1],[2,2],[3,3],[4,4]],i=>split(i))],['输出行 0–1','输出行 2–3'],'GPU 0 ⇄ GPU 1 · Reduce-scatter · 输出激活'),
 frame('反向交换所需梯度','行切层反向先汇集 dY。列切层算出的 dX 贡献则求和后按序列分回去。计算权重梯度时也可能重新汇集输入。','dY: all-gather；dX: reduce-scatter',[matrix('dX 布局',[[1,1],[1,1],[1,1],[1,1]],i=>split(i),'示意分片，不对应具体 loss')],['保留 dX 行 0–1','保留 dX 行 2–3'],'All-gather + Reduce-scatter · TP 组','反向')
];
const lessons=[
 lesson('TP','张量并行','切开同一层的权重，让多张卡合作处理同一批 token。',[example('MLP · 列切与行切',tp),example('Self Attention · 按头切',heads)]),
 lesson('DP','数据并行','复制模型，每张卡处理不同样本，再同步参数梯度。',[example('两个模型副本',dp)]),
 lesson('PP','流水线并行','把模型层分给不同阶段，用 microbatch 流动填充流水线。',[example('两阶段、两个 microbatch',pp)]),
 lesson('CP','上下文并行','沿序列位置切开长上下文，注意力需要访问其他卡负责的 KV。',[example('两卡因果注意力',cp)]),
 lesson('EP','专家并行','把 MoE 的专家放到不同卡，按 token 的路由选择交换数据。',[example('两个专家、Top-1 路由',ep)]),
 lesson('SP','序列并行','和 TP 配合，在层边界沿序列切分激活，减少重复保存。',[example('层边界切序列',sp)])
];
