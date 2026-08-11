% =========================================================================
% VL53L8CX 8x8 点阵测距实时分析与三维重建
% =========================================================================
clear; clc; close all;

%% 1. 参数配置
COM_PORT = 'COM8';          % 替换为你的 STM32 对应串口号
BAUD_RATE = 460800;         % 与 STM32 代码保持一致
FOV_DEG = 45;               % 传感器视场角 (水平/垂直约 45 度)

% 预计算角度网格 (8x8)
% 将 1~8 的索引映射到 -FOV/2 到 FOV/2 的弧度范围
angles = linspace(-FOV_DEG/2, FOV_DEG/2, 8) * pi / 180;
[ThetaX, ThetaY] = meshgrid(angles, angles);
TanX = tan(ThetaX);
TanY = tan(ThetaY);

%% 2. 初始化串口与独立可视化窗口
s = serialport(COM_PORT, BAUD_RATE);
configureTerminator(s, "LF"); % 设定换行符作为数据帧结束标志
flush(s);                     % 清空初始缓冲区

% 创建独立的可视化窗口
fig = figure('Name', '机器人前方空间形态实时监测', 'Position', [100, 100, 1000, 500]);

% 左侧面板：3D点云与拟合平面图
ax1 = subplot(1, 2, 1);
view(ax1, 3); grid on; hold on;
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('深度 Z (mm)');
title('实时三维点云与平面拟合');
zlim([0, 2000]); % 根据你的实际测距范围调整

% 右侧面板：文本结论输出
ax2 = subplot(1, 2, 2);
axis off;
title('实时空间状态分析结论');
txt_status = text(0.1, 0.8, '等待数据...', 'FontSize', 14, 'FontWeight', 'bold');
txt_details = text(0.1, 0.5, '', 'FontSize', 12);

%% 3. 实时读取与分析循环
disp('开始实时读取数据... 请按 Ctrl+C 或关闭窗口停止。');

% 当图形窗口有效时保持循环（关闭独立窗口即可退出程序）
while isvalid(fig)
    try
        % 1. 读取一行串口数据
        rawDataStr = readline(s);
        if isempty(rawDataStr); continue; end
        
        % 解析逗号分隔的 64 个数据
        dataArray = str2double(split(rawDataStr, ','));
        if length(dataArray) ~= 64
            continue; % 数据帧不完整则跳过
        end
        
        % 重塑为 8x8 矩阵 (注意转置，视 STM32 发送顺序而定)
        Z = reshape(dataArray, 8, 8)';
        
        % 滤除无效数据 (假设 0 或过大值为无效)
        valid_idx = (Z > 10 & Z < 4000); 
        
        % 2. 坐标映射 (2D -> 3D)
        X = Z .* TanX;
        Y = Z .* TanY;
        
        X_val = X(valid_idx);
        Y_val = Y(valid_idx);
        Z_val = Z(valid_idx);
        
        if length(Z_val) < 10
            set(txt_status, 'String', '状态：测距点过少，无法分析', 'Color', 'r');
            pause(0.05); continue;
        end
        
        % 3. 最小二乘法平面拟合: Z = a*X + b*Y + c
        % 构建设计矩阵 A = [X, Y, ones]
        A = [X_val, Y_val, ones(length(Z_val), 1)];
        coeffs = A \ Z_val; 
        
        a = coeffs(1); 
        b = coeffs(2); 
        c = coeffs(3);
        
        % 4. 计算拟合误差 (RMSE) 用于判断折面/杂乱
        Z_fit = a * X_val + b * Y_val + c;
        rmse = sqrt(mean((Z_val - Z_fit).^2));
        
        % 5. 解算姿态角 (度)
        yaw_deg = atan(a) * 180 / pi;
        pitch_deg = atan(b) * 180 / pi;
        min_dist = min(Z_val);
        
        % 6. 状态分类逻辑
        if rmse > 50  % 拟合误差过大，说明不是一个单一平整的平面
            status_str = '状态：检测到折角或杂乱障碍物';
            status_color = [0.8, 0.4, 0]; % 橙色
        else
            if abs(yaw_deg) < 5 && abs(pitch_deg) < 5
                status_str = '状态：与目标平面绝对正对';
                status_color = 'g';
            else
                status_str = '状态：与目标平面存在倾斜夹角';
                status_color = 'b';
            end
        end
        
        % 7. 实时更新 UI 界面
        % 更新结论文本
        set(txt_status, 'String', status_str, 'Color', status_color);
        detail_str = sprintf('最短距离: %.1f mm\n偏航角 (Yaw): %.1f°\n俯仰角 (Pitch): %.1f°\n平面粗糙度 (RMSE): %.1f', ...
                             min_dist, yaw_deg, pitch_deg, rmse);
        set(txt_details, 'String', detail_str);
        
        % 更新左侧 3D 图像
        cla(ax1); % 清除上一帧
        scatter3(ax1, X_val, Y_val, Z_val, 40, Z_val, 'filled'); % 绘制实际点云
        hold(ax1, 'on');
        
        % 绘制拟合出的半透明平面
        [Xq, Yq] = meshgrid(linspace(min(X_val), max(X_val), 10), ...
                            linspace(min(Y_val), max(Y_val), 10));
        Zq = a * Xq + b * Yq + c;
        surf(ax1, Xq, Yq, Zq, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', [0.5 0.5 0.5]);
        
        drawnow limitrate; % 限制刷新率以保证程序流畅不卡死
        
    catch ME
        % 捕获可能的串口断开等异常
        disp(ME.message);
    end
end