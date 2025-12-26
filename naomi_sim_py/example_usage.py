"""
示例：如何使用NAOMi Python框架的参数检查函数

这个示例展示了如何设置和检查各种模拟参数，与MATLAB版本的用法类似。
"""

from core.parameters import (
    check_vol_params,
    check_psf_params,
    check_scan_params,
    check_spike_opts,
    check_noise_params,
    check_tpm_params,
)


def main():
    """
    主函数：演示参数设置和检查的完整流程
    """
    print("=" * 60)
    print("NAOMi Simulation - Python参数检查示例")
    print("=" * 60)
    
    # 1. 设置体积参数（部分参数，其他使用默认值）
    print("\n1. 设置体积参数...")
    vol_params = check_vol_params({
        'vol_sz': [150, 150, 100],  # 体积大小（微米）
        'vol_depth': 200,            # 深度（微米）
    })
    print(f"   体积大小: {vol_params['vol_sz']}")
    print(f"   神经元数量: {vol_params['N_neur']}")
    print(f"   神经元密度: {vol_params['neur_density']:.2e} neurons/mm³")
    print(f"   分辨率: {vol_params['vres']} voxels/um")
    
    # 2. 设置PSF参数
    print("\n2. 设置PSF参数...")
    psf_params = check_psf_params({
        'NA': 0.6,      # 激发数值孔径
        'objNA': 0.8,   # 物镜数值孔径
        'lambda': 0.92, # 波长（微米）
    })
    print(f"   激发NA: {psf_params['NA']}")
    print(f"   物镜NA: {psf_params['objNA']}")
    print(f"   波长: {psf_params['lambda']} um")
    print(f"   PSF类型: {psf_params['type']}")
    
    # 3. 设置扫描参数
    print("\n3. 设置扫描参数...")
    scan_params = check_scan_params({
        'motion': True,    # 启用运动模拟
        'scan_avg': 2,     # 扫描平均
    })
    print(f"   运动模拟: {scan_params['motion']}")
    print(f"   扫描平均: {scan_params['scan_avg']}")
    print(f"   缓冲区: {scan_params['scan_buff']}")
    
    # 4. 设置时间活动参数
    print("\n4. 设置时间活动参数...")
    spike_opts = check_spike_opts({
        'nt': 20000,       # 时间步数
        'dt': 1/30,        # 采样间隔（30 Hz）
        'rate': 0.25,      # 平均发放率
    })
    print(f"   时间步数: {spike_opts['nt']}")
    print(f"   采样间隔: {spike_opts['dt']:.4f} s")
    print(f"   帧率: {1/spike_opts['dt']:.1f} Hz")
    print(f"   动态类型: {spike_opts['dyn_type']}")
    print(f"   蛋白质类型: {spike_opts['prot']}")
    
    # 5. 设置噪声参数（使用默认值）
    print("\n5. 设置噪声参数（全部使用默认值）...")
    noise_params = check_noise_params()
    print(f"   光子测量均值: {noise_params['mu']}")
    print(f"   光子测量方差: {noise_params['sigma']}")
    print(f"   电子噪声: {noise_params['sigma0']}")
    print(f"   像素串扰概率: {noise_params['bleedp']}")
    
    # 6. 设置TPM参数
    print("\n6. 设置TPM参数...")
    tpm_params = check_tpm_params({
        'pavg': 40,  # 激光功率（mW）
    })
    print(f"   激光功率: {tpm_params['pavg']} mW")
    print(f"   量子效率: {tpm_params['eta']}")
    print(f"   荧光团浓度: {tpm_params['conc']} uM")
    print(f"   激光重复频率: {tpm_params['f']} MHz")
    
    # 7. 展示完整参数结构
    print("\n" + "=" * 60)
    print("所有参数已设置完成，可以用于模拟")
    print("=" * 60)
    
    return {
        'vol_params': vol_params,
        'psf_params': psf_params,
        'scan_params': scan_params,
        'spike_opts': spike_opts,
        'noise_params': noise_params,
        'tpm_params': tpm_params,
    }


if __name__ == '__main__':
    params = main()

