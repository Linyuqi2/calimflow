# NAOMi Simulation - Python Implementation

这是NAOMi双光子显微镜模拟系统的Python实现框架。

## 框架设计概述

### 1. 整体架构

```
naomi_sim_py/
├── core/
│   ├── __init__.py
│   ├── parameters.py          # 参数检查和设置模块（已实现）
│   ├── volume.py              # 神经体积模拟（待实现）
│   ├── optics.py              # 光学传播模拟（待实现）
│   ├── activity.py            # 时间活动生成（待实现）
│   ├── scanning.py            # 扫描模拟（待实现）
│   └── utils.py               # 工具函数（待实现）
└── README.md
```

### 2. 参数管理模块（parameters.py）

#### 核心设计理念

参数检查函数遵循与MATLAB版本相同的逻辑：

1. **输入处理**：接受用户参数（可为None、空字典或部分字段）
2. **默认值设置**：定义完整的默认参数字典
3. **参数合并**：使用`set_params()`函数合并默认值和用户值（用户值优先）
4. **验证和计算**：进行额外的验证和计算（如神经元密度计算）
5. **返回完整参数**：返回包含所有必需字段的完整参数字典

#### 已实现的check函数

- `check_vol_params()`: 体积参数检查
  - 处理体积大小、分辨率、神经元数量/密度等
  - 自动计算神经元密度或数量（如果只提供其中一个）
  - 确保体积深度是10的倍数

- `check_psf_params()`: PSF（点扩散函数）参数检查
  - 处理数值孔径、波长、折射率等光学参数
  - 支持fastmask子参数设置

- `check_scan_params()`: 扫描参数检查
  - 处理扫描平均、运动模拟、缓冲区等参数

- `check_spike_opts()`: 时间活动参数检查
  - 处理时间步数、采样率、动态类型等参数

- `check_noise_params()`: 噪声模型参数检查
  - 处理光子测量、电子噪声、像素串扰等参数

- `check_tpm_params()`: TPM参数检查
  - 处理激光功率、量子效率、荧光团浓度等参数

#### 工具函数

- `set_params()`: 参数合并函数
  - 递归合并嵌套字典
  - 用户提供的值优先于默认值
  - 支持添加新字段

### 3. 使用示例

```python
from naomi_sim_py.core.parameters import (
    check_vol_params,
    check_psf_params,
    check_scan_params,
    check_spike_opts,
    check_noise_params,
    check_tpm_params,
)

# 只提供部分参数，其他使用默认值
vol_params = check_vol_params({
    'vol_sz': [150, 150, 100],
    'vol_depth': 200,
})

# 完全不提供参数，全部使用默认值
psf_params = check_psf_params()

# 提供部分参数
psf_params = check_psf_params({
    'NA': 0.6,
    'objNA': 0.8,
    'pavg': 40,
})

# 检查所有参数
vol_params = check_vol_params(vol_params)
psf_params = check_psf_params(psf_params)
scan_params = check_scan_params({})
spike_opts = check_spike_opts({'nt': 20000, 'dt': 1/30})
noise_params = check_noise_params({})
tpm_params = check_tpm_params({'pavg': 40})
```

### 4. 与MATLAB版本的对应关系

| MATLAB函数 | Python函数 | 状态 |
|-----------|-----------|------|
| `check_vol_params.m` | `check_vol_params()` | ✅ 已实现 |
| `check_psf_params.m` | `check_psf_params()` | ✅ 已实现 |
| `check_scan_params.m` | `check_scan_params()` | ✅ 已实现 |
| `check_spike_opts.m` | `check_spike_opts()` | ✅ 已实现 |
| `check_noise_params.m` | `check_noise_params()` | ✅ 已实现 |
| `check_tpm_params.m` | `check_tpm_params()` | ✅ 已实现 |
| `setParams.m` | `set_params()` | ✅ 已实现 |

### 5. 后续开发计划

#### 待实现的核心模块

1. **体积模拟模块** (`volume.py`)
   - `simulate_neural_volume()`: 生成神经体积
   - 需要实现神经元、树突、轴突、血管、背景等组件的生成

2. **光学模块** (`optics.py`)
   - `simulate_optical_propagation()`: 模拟光学传播
   - 需要实现PSF计算、光学掩膜生成等

3. **活动生成模块** (`activity.py`)
   - `generate_time_traces()`: 生成时间活动轨迹
   - 需要实现AR模型、钙动力学模型等

4. **扫描模块** (`scanning.py`)
   - `scan_volume()`: 执行扫描模拟
   - 需要实现扫描过程、噪声添加等

5. **工具模块** (`utils.py`)
   - 文件I/O、可视化、数学工具等

### 6. 设计原则

1. **兼容性**：尽量保持与MATLAB版本的参数命名和逻辑一致
2. **类型安全**：使用类型提示提高代码可读性和可维护性
3. **灵活性**：支持部分参数输入，自动补全默认值
4. **可扩展性**：模块化设计，便于后续添加新功能

### 7. 依赖项

- `numpy`: 数值计算
- `typing`: 类型提示（Python标准库）

### 8. 测试建议

建议为每个check函数编写单元测试，验证：
- 默认值设置是否正确
- 用户参数是否正确覆盖默认值
- 参数验证逻辑是否正确
- 边界情况处理是否正确

