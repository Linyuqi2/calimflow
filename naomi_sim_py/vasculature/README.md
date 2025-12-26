# 血管生成模块 (Vasculature Module)

## 📋 MATLAB源码分析

### 主函数：`simulatebloodvessels.m`

#### **函数签名**
```matlab
[neur_ves, vasc_params, neur_ves_all] = simulatebloodvessels(vol_params, vasc_params)
```

#### **输入参数**
- `vol_params`: 体积参数（尺寸、分辨率等）
- `vasc_params`: 血管参数（尺寸、频率、权重等）

#### **输出参数**
- `neur_ves`: 神经区域内的血管（3D二进制数组）
- `vasc_params`: 更新后的血管参数
- `neur_ves_all`: 完整血管体积（可选）

---

## 🔄 执行流程

### **1. 输入解析和参数检查**
```matlab
vol_params = check_vol_params(vol_params);
vasc_params = check_vasc_params(vasc_params);
```

### **2. 参数设置和缩放**
```matlab
% 考虑体积分辨率进行缩放
vp.depth_surf = vasc_params.depth_surf * vres;
vp.mindists = vasc_params.vesFreq * vres / 2;
vp.maxcappdist = 2 * vasc_params.vesFreq(3) * vres;
vp.vesSize = vasc_params.vesSize * vres;
```

### **3. 计算血管数量**
```matlab
% 不同类型血管的数量计算
nv.nsource = max(round(...), 0);  % 源节点
nv.nvert = max(round(...), 0);    % 垂直血管
nv.nsurf = max(round(...), 0);    % 表面血管
nv.ncapp = max(round(...), 0);    % 毛细血管
```

### **4. 生成主要血管**
```matlab
[nodes, nv] = growMajorVessels(nv, np, vp);
```
- 生成垂直和表面血管的节点

### **5. 节点转连接**
```matlab
conn = nodesToConn(nodes);
```
- 将节点结构转换为连接结构

### **6. 表面血管位置调整**
```matlab
% 根据血管直径调整表面位置
for i = 1:length(conn)
    if surface_vessel_condition
        nodes(conn(i).start).pos(3) = adjusted_position;
    end
end
```

### **7. 创建主要血管体积**
```matlab
[neur_ves, conn] = connToVol(nodes, conn, nv);
```
- 将连接转换为3D体积

### **8. 生成毛细血管**
```matlab
[nodes, conn, nv] = growCapillaries(nodes, conn, neur_ves, nv, vp, vres);
```

### **9. 添加毛细血管到体积**
```matlab
cappidxs = find(cellfun(@isempty, {conn.locs}));
[neur_ves, ~] = connToVol(nodes, conn, nv, cappidxs, neur_ves);
```

---

## 🏗️ Python实现架构

### **核心数据结构**

#### **节点结构 (Node)**
```python
@dataclass
class VascNode:
    pos: np.ndarray      # 3D位置 [x, y, z]
    type: str           # 类型: 'source', 'vert', 'surf', 'capp', etc.
    conn: List[int]     # 连接的节点索引
    weight: float       # 权重
```

#### **连接结构 (Connection)**
```python
@dataclass
class VascConnection:
    start: int          # 起始节点索引
    ends: int           # 结束节点索引
    weight: float       # 连接权重
    locs: Optional[np.ndarray]  # 连接路径点
    type: str           # 连接类型
```

#### **血管网络参数 (VascNetwork)**
```python
@dataclass
class VascNetwork:
    vol_sz: np.ndarray      # 体积尺寸
    size: np.ndarray        # 缩放后的尺寸
    nsource: int           # 源节点数量
    nvert: int             # 垂直血管数量
    nsurf: int             # 表面血管数量
    ncapp: int             # 毛细血管数量
    nconn: int             # 连接数量
```

---

## 📁 文件结构

```
vasculature/
├── __init__.py              # 模块初始化
├── simulate_blood_vessels.py # 主函数
├── major_vessels.py         # 主要血管生成
├── capillaries.py           # 毛细血管生成
├── nodes_and_connections.py # 节点和连接处理
├── utils.py                 # 工具函数
├── test_vasculature.py      # 单元测试
└── README.md               # 此文档
```

---

## 🔧 关键算法实现

### **1. 主要血管生成算法**

#### **节点放置策略**
- **源节点**: 在体积边界随机放置
- **垂直血管**: 从源节点向下生长
- **表面血管**: 在顶部表面水平生长

#### **生长规则**
```python
def grow_major_vessels(nv: VascNetwork, np: NodeParams, vp: VascParams) -> Tuple[List[VascNode], VascNetwork]:
    nodes = []

    # 放置源节点
    source_nodes = place_source_nodes(nv, vp)

    # 生成垂直血管
    vert_nodes = grow_vertical_vessels(source_nodes, nv, vp)

    # 生成表面血管
    surf_nodes = grow_surface_vessels(source_nodes, nv, vp)

    return source_nodes + vert_nodes + surf_nodes, nv
```

### **2. 毛细血管生成算法**

#### **连接策略**
- 寻找最近的主要血管节点
- 使用距离权重进行连接
- 限制最大连接距离

#### **生长规则**
```python
def grow_capillaries(nodes: List[VascNode], conn: List[VascConnection],
                    neur_ves: np.ndarray, nv: VascNetwork, vp: VascParams, vres: float):
    # 为每个主要血管节点找到毛细血管连接
    for node_idx in major_vessel_indices:
        nearest_nodes = find_nearest_nodes(nodes[node_idx], nodes, max_dist=vp.maxcappdist)
        for target_idx in nearest_nodes:
            if should_connect_capillary(nodes[node_idx], nodes[target_idx], vp):
                conn.append(create_capillary_connection(node_idx, target_idx, vp))
```

### **3. 体积转换算法**

#### **连接到体积的转换**
```python
def conn_to_vol(nodes: List[VascNode], conn: List[VascConnection], nv: VascNetwork,
               conn_indices: Optional[List[int]] = None, existing_vol: Optional[np.ndarray] = None) -> np.ndarray:
    """
    将节点连接转换为3D体积
    """
    if existing_vol is None:
        vol = np.zeros(nv.size, dtype=bool)
    else:
        vol = existing_vol.copy()

    # 处理指定的连接或所有连接
    connections = conn if conn_indices is None else [conn[i] for i in conn_indices]

    for connection in connections:
        # 在两个节点之间绘制血管
        vol = draw_vessel_between_nodes(vol, nodes[connection.start], nodes[connection.ends],
                                      connection.weight, nv)

    return vol
```

---

## 🧪 测试策略

### **1. 单元测试**
- 测试参数缩放是否正确
- 测试节点放置算法
- 测试连接生成逻辑

### **2. 集成测试**
- 测试完整血管生成流程
- 验证输出体积的几何属性

### **3. 与MATLAB对比测试**
- 使用相同的随机种子
- 比较输出体积的统计特性
- 验证血管密度和分布

---

## 📊 性能考虑

### **内存优化**
- 使用布尔数组而不是浮点数
- 分批处理大型体积
- 及时清理中间结果

### **速度优化**
- 向量化距离计算
- 使用KDTree进行最近邻搜索
- 并行化独立计算

---

## 🎯 实现计划

### **阶段1: 基础框架**
- [ ] 创建数据结构类
- [ ] 实现参数处理函数
- [ ] 基础测试框架

### **阶段2: 主要血管**
- [ ] 实现源节点放置
- [ ] 实现垂直血管生长
- [ ] 实现表面血管生长

### **阶段3: 毛细血管**
- [ ] 实现毛细血管连接算法
- [ ] 优化距离计算

### **阶段4: 体积转换**
- [ ] 实现连接到体积的转换
- [ ] 优化内存使用

### **阶段5: 集成和测试**
- [ ] 整合所有组件
- [ ] 与MATLAB版本对比测试
- [ ] 性能优化
