%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% test_component_separation.m
%
% 测试各个组件是否可以独立调用和分离
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

fprintf('========================================\n');
fprintf('测试组件分离能力\n');
fprintf('========================================\n\n');

% 添加路径
addpath(genpath('code'));
installNAOMi;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 测试1: 单独调用血管生成函数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('测试1: 单独调用血管生成函数\n');
fprintf('--------------------------------\n');

try
    % 设置参数
    vol_params.vol_sz = [100, 100, 50];
    vol_params.vol_depth = 150;
    vol_params = check_vol_params(vol_params);

    vasc_params = check_vasc_params([]);

    % 调用血管生成函数
    tic;
    [neur_ves, vasc_params_out, neur_ves_all] = simulatebloodvessels(vol_params, vasc_params);
    toc_time = toc;

    fprintf('✓ 血管生成成功，耗时: %.2f 秒\n', toc_time);
    fprintf('  输出尺寸: neur_ves = %s, neur_ves_all = %s\n', ...
        mat2str(size(neur_ves)), mat2str(size(neur_ves_all)));
    fprintf('  非零体素数: neur_ves = %d, neur_ves_all = %d\n\n', ...
        nnz(neur_ves), nnz(neur_ves_all));

    % 可视化血管（如果需要）
    % figure; slice(neur_ves_all, [], [], 50); title('血管结构');

catch ME
    fprintf('✗ 血管生成失败: %s\n\n', ME.message);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 测试2: 单独调用神经元采样函数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('测试2: 单独调用神经元采样函数\n');
fprintf('--------------------------------\n');

try
    % 设置参数
    neur_params = check_neur_params([]);
    vol_params.vol_sz = [50, 50, 25];  % 更小的体积用于测试
    vol_params.vol_depth = 100;
    vol_params = check_vol_params(vol_params);

    % 先需要血管（作为障碍物）
    vasc_params = check_vasc_params(struct('flag', false));  % 关闭血管以简化
    [neur_ves, ~, ~] = simulatebloodvessels(vol_params, vasc_params);

    % 调用神经元采样
    tic;
    [neur_locs, Vcell, Vnuc, Tri, rotAng] = sampleDenseNeurons(neur_params, vol_params, neur_ves);
    toc_time = toc;

    fprintf('✓ 神经元采样成功，耗时: %.2f 秒\n', toc_time);
    fprintf('  神经元数量: %d\n', size(Vcell, 3));
    fprintf('  位置范围: X[%.1f, %.1f], Y[%.1f, %.1f], Z[%.1f, %.1f]\n', ...
        min(neur_locs(:,1)), max(neur_locs(:,1)), ...
        min(neur_locs(:,2)), max(neur_locs(:,2)), ...
        min(neur_locs(:,3)), max(neur_locs(:,3)));
    fprintf('\n');

catch ME
    fprintf('✗ 神经元采样失败: %s\n\n', ME.message);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 测试3: 单独调用神经体积生成函数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('测试3: 单独调用神经体积生成函数\n');
fprintf('--------------------------------\n');

try
    % 设置参数
    neur_params = check_neur_params([]);
    vol_params.vol_sz = [30, 30, 15];  % 很小的体积用于快速测试
    vol_params.vol_depth = 50;
    vol_params = check_vol_params(vol_params);

    vasc_params = check_vasc_params(struct('flag', false));
    [neur_ves, ~, ~] = simulatebloodvessels(vol_params, vasc_params);

    % 采样神经元
    [neur_locs, Vcell, Vnuc, Tri, rotAng] = sampleDenseNeurons(neur_params, vol_params, neur_ves);

    % 调用体积生成
    tic;
    [neur_soma, neur_vol, gp_nuc, gp_soma] = generateNeuralVolume(neur_params, vol_params, neur_locs, Vcell, Vnuc, neur_ves);
    toc_time = toc;

    fprintf('✓ 神经体积生成成功，耗时: %.2f 秒\n', toc_time);
    fprintf('  体积尺寸: %s\n', mat2str(size(neur_vol)));
    fprintf('  神经元体素数: %d\n', nnz(neur_soma));
    fprintf('  总荧光体素数: %d\n', nnz(neur_vol));
    fprintf('\n');

catch ME
    fprintf('✗ 神经体积生成失败: %s\n\n', ME.message);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 测试4: 单独调用树突生长函数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('测试4: 单独调用树突生长函数\n');
fprintf('--------------------------------\n');

try
    % 复用上面的神经元数据
    if exist('neur_soma', 'var') && exist('neur_vol', 'var')
        dend_params = check_dend_params([]);

        tic;
        [neur_num, cellVolumeAD, dend_params_out, gp_soma_out] = growNeuronDendrites(...
            vol_params, dend_params, neur_soma, neur_ves, neur_locs, gp_nuc, gp_soma, rotAng);
        toc_time = toc;

        fprintf('✓ 树突生长成功，耗时: %.2f 秒\n', toc_time);
        fprintf('  树突体素数: %d\n', nnz(neur_num) - nnz(neur_soma));
        fprintf('\n');
    else
        fprintf('⚠ 跳过树突生长测试（需要前面的神经元数据）\n\n');
    end

catch ME
    fprintf('✗ 树突生长失败: %s\n\n', ME.message);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 测试5: 单独调用背景生成函数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('测试5: 单独调用背景生成函数\n');
fprintf('--------------------------------\n');

try
    % 设置背景参数
    bg_params = check_bg_params(struct('flag', true));
    dend_params = check_dend_params([]);

    % 需要一些基本的神经元数据
    if ~exist('neur_num', 'var')
        neur_num = neur_soma;  % 如果没有树突，就用神经元体
        gp_vals = gp_soma;
    end

    tic;
    [neur_num_bg, neur_vol_bg, vol_params_bg, gp_vals_bg, neur_locs_bg] = ...
        generate_bgdendrites(vol_params, bg_params, dend_params, neur_vol, ...
                           neur_num, gp_vals, gp_nuc, neur_locs);
    toc_time = toc;

    fprintf('✓ 背景生成成功，耗时: %.2f 秒\n', toc_time);
    fprintf('  背景体素数: %d\n', nnz(neur_num_bg) - nnz(neur_num));
    fprintf('\n');

catch ME
    fprintf('✗ 背景生成失败: %s\n\n', ME.message);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 测试6: 单独调用轴突生成函数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('测试6: 单独调用轴突生成函数\n');
fprintf('--------------------------------\n');

try
    % 设置轴突参数
    axon_params = check_axon_params(struct('flag', true));

    if exist('neur_vol_bg', 'var')
        neur_vol_test = neur_vol_bg;
        neur_num_test = neur_num_bg;
        gp_vals_test = gp_vals_bg;
    else
        neur_vol_test = neur_vol;
        neur_num_test = neur_num;
        gp_vals_test = gp_vals;
    end

    tic;
    [neur_vol_ax, gp_bgvals, axon_params_out] = ...
        generate_axons(vol_params, axon_params, neur_vol_test, neur_num_test, ...
                     gp_vals_test, gp_nuc);
    toc_time = toc;

    fprintf('✓ 轴突生成成功，耗时: %.2f 秒\n', toc_time);
    fprintf('  轴突体素数: %d\n', nnz(neur_vol_ax) - nnz(neur_vol_test));
    fprintf('\n');

catch ME
    fprintf('✗ 轴突生成失败: %s\n\n', ME.message);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 总结
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('========================================\n');
fprintf('组件分离测试总结\n');
fprintf('========================================\n');

tests = {
    '血管生成', '神经元采样', '神经体积生成', '树突生长', '背景生成', '轴突生成'
};

results = [
    exist('neur_ves', 'var'), ...
    exist('neur_locs', 'var'), ...
    exist('neur_vol', 'var'), ...
    exist('neur_num', 'var'), ...
    exist('neur_num_bg', 'var'), ...
    exist('neur_vol_ax', 'var')
];

for i = 1:length(tests)
    status = '✓' if results(i) else '✗';
    fprintf('%s %s\n', status, tests{i});
end

fprintf('\n结论: %d/%d 个组件可以独立测试\n', sum(results), length(tests));

if sum(results) == length(tests)
    fprintf('🎉 所有组件都可以独立调用和测试！\n');
else
    fprintf('⚠️ 部分组件测试失败，建议检查依赖关系\n');
end

fprintf('\n这为分步开发Python版本提供了很好的参考。\n');

