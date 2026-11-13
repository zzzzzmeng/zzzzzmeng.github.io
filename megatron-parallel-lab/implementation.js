'use strict';
const revision='581248531984920e92f076946ad4629d359442a4';
const impl={
 TP:{cut:'权重的输出列 / 输入行（数学 XW 记法）',shape:'X[T,H] 复制 → 中间激活[T,F/2] → 输出[T,H] 复制',boundary:'普通 TP，关闭 SP；PyTorch weight 按 [输出,输入] 存储，源码切分轴与 XW 记法相反。',file:'tensor_parallel/layers.py',line:1353,forward:'ColumnParallelLinear.forward → RowParallelLinear.forward',backward:'LinearWithGradAccumulationAndAsyncCommunication.backward',code:'# 中文注释，节选关键调用\n# 列切保留输出分片，gather_output=False\noutput = output_parallel\n# 行切合并相同位置的部分和\noutput_ = reduce_from_tensor_model_parallel_region(\n    output_parallel, group=self.tp_group\n)\n# 反向：列切输入梯度汇合\ngrad_input = grad_output.matmul(weight)\nhandle = torch.distributed.all_reduce(\n    grad_input, group=tp_group, async_op=True\n)',foot:'融合 QKV 的本地反向把三路输入贡献加起来再归约；行切前向 SUM 的反向直接传递同一份 dY。'},
 DP:{cut:'全局 batch 的样本轴',shape:'X[B,…] → 每副本 X[B/2,…]；参数 W 在两卡复制',boundary:'普通 DDP，等量样本。本示例 loss 是本地平方误差之和的一半，全局再平均两个本地 loss。',file:'distributed/param_and_grad_buffer.py',line:766,forward:'各 rank 使用各自样本，独立执行模型 forward',backward:'start_grad_sync → finish_grad_sync',code:'# 中文注释，普通 DDP 分支节选\n# 梯度 bucket 含多份参数梯度\ntorch.distributed.all_reduce(\n    bucket.grad_data, op=reduce_op,\n    group=communication_group, async_op=async_op\n)\n# 分布式优化器分支改用 reduce-scatter\n# SUM/AVG 与额外缩放由训练配置决定',foot:'不同有效 token 数时应核对全局 loss 的归一化。真实通信按 bucket 触发；分布式优化器还涉及参数汇集。'},
 PP:{cut:'模型深度，即不同的 Transformer 层',shape:'阶段 0 的 X → 边界激活 A → 阶段 1 的 Y',boundary:'数值示例用两层线性变换代替两段 Block；另有按 Megatron 非交错调度依赖构造的 1F1B 示意。时间片不代表实际耗时。',file:'pipeline_parallel/schedules.py',line:2301,forward:'forward_backward_pipelining_without_interleaving',backward:'send_forward_recv_backward → backward_step → send_backward_recv_forward',code:'# 源码节选与中文注释\nnum_warmup_microbatches = (\n    p2p_communicator.total_stages\n    - p2p_communicator.current_stage - 1\n)\n# 稳定段，发送新激活，接收旧 microbatch 的梯度\noutput_tensor_grad = p2p_communicator.send_forward_recv_backward(\n    output_tensor, send_tensor_shapes,\n    p2p_communicator.is_pp_last_stage\n)\n# 反向后，将边界梯度送回前序阶段',foot:'参数与参数梯度留在所属阶段；跨阶段传激活和激活梯度。'},
 CP:{cut:'同一条序列的上下文位置',shape:'X[S,B,H] → 每卡 S/2 个查询位置；本地 Q 读取所需远端 KV',boundary:'第一个例子为连续布局手算，第二个例子为本地 per-sequence balancing 镜像分块。注意力内核由 Transformer Engine 等后端执行。',file:'utils.py',line:2695,forward:'get_batch_on_this_cp_rank → CP-aware Attention',backward:'Attention 后端回传并累加远端 KV 梯度贡献',code:'# per-sequence balancing 源码节选\n# 序列预先 reshape 为 2 * cp_size 块\nindex[0].fill_(cp_rank)\nindex[1].fill_(2 * cp_size - cp_rank - 1)\nval = val.index_select(seq_dim, index)\n# TE 接入所用的通信组\nextra_kwargs["cp_group"] = pg_collection.cp\n# cp_comm_type 根据配置选择，如 p2p',foot:'本地投影权重是共享参数，CP 查询分片的参数梯度还需要按配置归约，涉及 DP×CP 组。KV 贡献回传与参数梯度归约分开理解。'},
 EP:{cut:'MoE 的专家集合，token 随路由跨卡',shape:'原始 token → 按专家置换 → 专家计算 → 返回原始 token 位置',boundary:'Top-1、专家 TP=1、门控权重固定为 1；标量缩放代替专家 MLP。本演示聚焦专家分支，实际 router 与负载均衡损失另有梯度。',file:'transformer/moe/token_dispatcher.py',line:720,forward:'dispatch_preprocess → token_dispatch → experts → token_combine',backward:'_AllToAll.backward 交换 input/output split sizes，沿路由返回',code:'# dispatcher 源码节选，中文注释\n# 发送给对应专家\nglobal_input_tokens = all_to_all(\n    self.ep_group, permutated_local_input_tokens,\n    self.output_splits, self.input_splits,\n    use_nccl_stream=self.use_nccl_stream\n)\n# combine 使用反方向的 split sizes\n# 此实现还单独交换 global_probs',foot:'真实 dispatcher 还传概率或元数据，因此不能将“两次激活 all-to-all”直接等同于全部底层通信调用数。'},
 SP:{cut:'TP 组内，层边界激活的序列行',shape:'[S/2,B,H] → AG → [S,B,H] → 本地特征分片 → RS → [S/2,B,H]',boundary:'与 TP=2 同组，权重切法和 TP 相同；归一化等边界部分保留本地序列。',file:'tensor_parallel/layers.py',line:787,forward:'列切前 all-gather → 行切后 reduce-scatter',backward:'行切 dY all-gather → 列切 dX reduce-scatter',code:'# 前向的序列汇集，源码节选\ndist_all_gather_func(all_gather_buffer, input, group=tp_group)\ntotal_input = all_gather_buffer\n# 行切输出\noutput_ = reduce_scatter_to_sequence_parallel_region(\n    output_parallel, group=self.tp_group\n)\n# 反向：计算 wgrad 可能重新汇集输入\n# dX 使用 dist_reduce_scatter_func',foot:'输入重聚合和复制的归一化参数梯度同步也要计入真实执行。SP 和 CP 在注意力阶段的序列布局不同。'}
};

// Source-corresponding CP partition map, alongside the contiguous numeric example.
const balanced=[
 frame('序列先分成四个块','S=8，CP=2。per-sequence balancing 将序列分成 2×CP 个块，再按首尾镜像取块。','块号 = [cp_rank, 2×CP−cp_rank−1]',[matrix('四个块的 token 位置',[[0,1],[2,3],[4,5],[6,7]],i=>[0,1,1,0][i])],['块 0、3 → 位置 0,1,6,7','块 1、2 → 位置 2,3,4,5']),
 frame('两卡的因果查询工作量均衡','第 j 个查询可看 j+1 个位置。把早期与晚期块搭配，避免连续布局的后半卡独自承担更多配对。这里只统计有效 QK 配对。','GPU 0：1+2+7+8=18；GPU 1：3+4+5+6=18',[shard('GPU 0 可见位置数',[[1,2,7,8]],0),shard('GPU 1 可见位置数',[[3,4,5,6]],1)],['本地 Q 位置 0,1,6,7','本地 Q 位置 2,3,4,5']),
 frame('查询留本地，按需取得其他块 KV','位置 7 在 GPU 0，它还需要 GPU 1 的位置 2–5。位置 5 在 GPU 1，则需要 GPU 0 的位置 0–1。图示逻辑依赖，具体分块轮次由后端决定。','本地 Q × 全部因果可见 KV',[matrix('KV 的位置所有者',[[0,0,1,1,1,1,0,0]],(_,j)=>[0,0,1,1,1,1,0,0][j])],['读取位置 2–5 的 KV','读取位置 0–1 的 KV'],'GPU 0 ⇄ GPU 1 · KV 分块交换'),
 back('反向将远端 KV 贡献归还原位置','每个查询先在本地得到 dQ 及所用 KV 的梯度贡献。远端贡献回到其位置所属卡，和本卡查询贡献相加。参数梯度再按共享参数组同步。','dKV_j = Σ 可见查询 i 的贡献',[matrix('dKV 所属位置',[[0,1,2,3,4,5,6,7]],(_,j)=>[0,0,1,1,1,1,0,0][j])],['累加位置 0,1,6,7 的 dKV','累加位置 2,3,4,5 的 dKV'],'GPU 0 ⇄ GPU 1 · 远端 dK/dV 贡献')
];
lessons[3].cases.push(example('源码布局 · 镜像分块',balanced));
splitDiagrams.CP.push([{title:'源码中的镜像分块',axis:'S=8 · CP=2 · 先切 4 块再配首尾',source:[strip(['t₀','t₁','t₂','t₃','t₄','t₅','t₆','t₇'],[0,0,1,1,1,1,0,0])],parts:[grid([['t₀','t₁'],['t₆','t₇']],[[0,0],[0,0]]),grid([['t₂','t₃'],['t₄','t₅']],[[1,1],[1,1]])],notes:['块 0、3','块 1、2']}]);

// Abstract 1F1B timeline, dependencies enforced for each microbatch.
const schedule=[
 ['F0','—'],['F1','F0'],['等待 dA₀','B0'],['B0','F1'],
 ['F2','B1'],['B1','F2'],['等待 dA₂','B2'],['B2','—']
];
const ppTimeline=schedule.map((row,i)=>frame('1F1B 时间片 '+(i+1),
 '按两阶段、三个 microbatch 的依赖构造。阶段 0 的操作序列为 F0,F1,B0,F2,B1,B2；阶段 1 为 F0,B0,F1,B1,F2,B2。横轴是教学时间片，实际执行可重叠通信。',
 'Fμ：前向激活向后；Bμ：激活梯度向前',
 [matrix('时间片 '+(i+1)+' / 阶段编号',[[0,1]],(_,j)=>j,'两卡当前任务见设备卡片')],row,
 row.some(x=>x.startsWith('B'))?'GPU 1 → GPU 0 · 按 μ 配对 dA':'GPU 0 → GPU 1 · 按 μ 配对激活',
 row.some(x=>x.startsWith('B'))?'反向':'前向'));
lessons[2].cases.push(example('源码调度 · 1F1B 时间线',ppTimeline));
splitDiagrams.PP.push(splitDiagrams.PP[0]);

