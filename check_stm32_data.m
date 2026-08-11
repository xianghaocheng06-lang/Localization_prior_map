% 清理可能残留的串口对象
clear s;

% 1. 创建串口对象 (请将 'COM8' 替换为你实际的串口号)
s = serialport("COM8", 115200);
s.Timeout = 20;
while true
    % 尝试直接读取 10 个字节看看是不是全零或有数据
    data = read(s, 10, "uint8");
    disp(data);
end
% 二进制传输不需要设置任何 Terminator
figure('Name', 'VL53L8CX 8x8 Real-time Heatmap', 'NumberTitle', 'off');
h = imagesc(zeros(8, 8)); 
colormap(jet);
colorbar;
caxis([0 2000]); % 假设测距范围 0 - 2000 mm，可根据实际调整
title('Real-time 8x8 Depth Matrix');
xlabel('X (Pixels)');
ylabel('Y (Pixels)');

fprintf('Waiting for STM32 binary stream...\n');

while ishandle(h)
    % 2. 寻找包头 0xAA (170)
    b1 = read(s, 1, "uint8");
    if b1 == 170
        % 3. 寻找第二个包头 0x55 (85)
        b2 = read(s, 1, "uint8");
        if b2 == 85
            % 4. 一次性读取随后的 128 个字节 (64个 int16_t)
            raw_data = read(s, 128, "uint8");
            
            % 5. 字节流转为 8x8 矩阵
            matrix_data = reshape(typecast(raw_data, 'int16'), [8, 8]);
            
            % 6. 实时刷新热力图
            set(h, 'CData', matrix_data);
            drawnow limitrate;
        end
    end
end