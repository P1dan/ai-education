
COGNITIVE_THRESHOLD = {"COMFORT": 0.8, "LEARNING": 0.4, "PANIC": 0.4}

KNOWLEDGE_GRAPH = {
    "神经网络基础": {"pre_reqs": [], "next": ["感知机", "激活函数"], "cluster": "基础认知", "stage": 1, "complexity": 1},
    "感知机": {"pre_reqs": ["神经网络基础"], "next": ["多层感知机"], "cluster": "网络结构", "stage": 2, "complexity": 2},
    "激活函数": {"pre_reqs": ["神经网络基础"], "next": ["反向传播"], "cluster": "核心机制", "stage": 2, "complexity": 2},
    "多层感知机": {"pre_reqs": ["感知机"], "next": ["损失函数"], "cluster": "网络结构", "stage": 3, "complexity": 3},
    "损失函数": {"pre_reqs": ["多层感知机"], "next": ["梯度下降"], "cluster": "优化基础", "stage": 3, "complexity": 3},
    "链式法则": {"pre_reqs": [], "next": ["反向传播"], "cluster": "数学基础", "stage": 2, "complexity": 2},
    "梯度下降": {"pre_reqs": ["损失函数"], "next": ["反向传播"], "cluster": "优化基础", "stage": 4, "complexity": 4},
    "反向传播": {"pre_reqs": ["激活函数", "梯度下降", "链式法则"], "next": ["卷积神经网络"], "cluster": "核心机制", "stage": 5, "complexity": 5},
    "卷积神经网络": {"pre_reqs": ["反向传播"], "next": ["池化层", "卷积核"], "cluster": "高级结构", "stage": 6, "complexity": 5},
    "池化层": {"pre_reqs": ["卷积神经网络"], "next": [], "cluster": "高级结构", "stage": 7, "complexity": 4},
    "卷积核": {"pre_reqs": ["卷积神经网络"], "next": [], "cluster": "高级结构", "stage": 7, "complexity": 4},
}