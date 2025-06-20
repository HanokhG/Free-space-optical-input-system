import numpy as np
import cv2
from PIL import Image
import time

# 读取并展示初始图片 image0
image_0 = Image.open(r'D:\app\pycharm\learn\key\image0.bmp')  # 使用绝对路径
image_0_cv = np.array(image_0)  # 转换为 OpenCV 格式
image_0_cv_resized = cv2.resize(image_0_cv, (1920, 1080))  # 调整初始图像大小

# 显示初始图片 image0
cv2.imshow('Displayed Image', image_0_cv_resized)

# 初始化摄像头输入
cap = cv2.VideoCapture(1)  # 使用摄像头编号 1 (根据你的摄像头配置)
ret, first_frame = cap.read()

# 检查摄像头是否成功读取
if not ret:
    print("无法读取摄像头输入")
    exit()

# 记录开始时间，用于10秒后保存图像
start_time = time.time()
image_saved = False  # 标记图像是否保存

# 灰度化初始图像
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# 定义要裁剪的区域
useful_row_start = 100  # 定义要裁剪的起始行
useful_row_end = 350    # 定义要裁剪的结束行
useful_col_start = 260  # 定义要裁剪的起始列
useful_col_end =460 # 定义要裁剪的结束列

# 只处理这个区域的图像
first_gray_cropped = first_gray[useful_row_start:useful_row_end, useful_col_start:useful_col_end]

# 检查裁剪后的区域是否为空
if first_gray_cropped.size == 0:
    print("裁剪后的区域为空，无法处理")
    exit()

# 获取裁剪后的图像尺寸
cropped_rows, cropped_cols = first_gray_cropped.shape

# 将裁剪区域划分为3x3区域
row_step = cropped_rows // 3
col_step = cropped_cols // 3

# 定义一个函数用于获取图像的子区域
def get_zones(image_cropped, row_step, col_step, cropped_rows, cropped_cols):
    return [
        image_cropped[0:row_step, 0:col_step],
        image_cropped[0:row_step, col_step:2 * col_step],
        image_cropped[0:row_step, 2 * col_step:cropped_cols],
        image_cropped[row_step:2 * row_step, 0:col_step],
        image_cropped[row_step:2 * row_step, col_step:2 * col_step],
        image_cropped[row_step:2 * row_step, 2 * col_step:cropped_cols],
        image_cropped[2 * row_step:cropped_rows, 0:col_step],
        image_cropped[2 * row_step:cropped_rows, col_step:2 * col_step],
        image_cropped[2 * row_step:cropped_rows, 2 * col_step:cropped_cols]
    ]

# 当前显示的图片路径
current_image = r'D:\app\pycharm\learn\key\image0.bmp'
displayed_image = image_0_cv_resized  # 初始显示的图片是 image0
last_change_time = time.time()  # 用于记录最后一次更改图片的时间
change_detected = False  # 检测到变化的标志

# 定义一个函数用于计算傅里叶变换的峰值强度
def compute_fourier_peak_intensity(image, r_high=20):
    # 计算傅里叶变换
    f_transform = np.fft.fft2(image)
    f_transform_shift = np.fft.fftshift(f_transform)

    # 创建掩膜，保留三阶及以上分量
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # 找到频域中心

    mask = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if dist > r_high:  # 仅保留距离中心大于r_high的频率分量
                mask[i, j] = 1

    # 应用掩膜
    f_transform_shift_masked = f_transform_shift * mask
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shift_masked) + 1)

    # 返回傅里叶频域图像的峰值强度
    return np.max(magnitude_spectrum)

# 计算每个子区域的初始傅里叶峰值强度
initial_zones = get_zones(first_gray_cropped, row_step, col_step, cropped_rows, cropped_cols)
initial_peak_intensities = [compute_fourier_peak_intensity(zone) for zone in initial_zones]

# 持续测量差异检测
while True:
    ret, frame = cap.read()

    if not ret:
        print("无法读取视频帧")
        break

    # 实时展示摄像头捕获的帧并裁剪到指定区域
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_cropped = gray[useful_row_start:useful_row_end, useful_col_start:useful_col_end]

    # 检查裁剪后的区域是否为空
    if gray_cropped.size == 0:
        print("裁剪后的摄像头图像为空，无法继续")
        break

    # 显示摄像头捕获的裁剪图像
    cv2.imshow('Camera Feed', gray_cropped)

    # 计算当前帧的每个子区域的傅里叶峰值强度
    zones = get_zones(gray_cropped, row_step, col_step, cropped_rows, cropped_cols)

    # 初始化一个标志变量，表示是否有变化
    detected_change = False

    # 遍历所有区域，比较当前帧与初始帧的傅里叶峰值强度差异
    for zone_index, zone in enumerate(zones):
        current_peak_intensity = compute_fourier_peak_intensity(zone)
        initial_peak_intensity = initial_peak_intensities[zone_index]

        # 如果当前区域的傅里叶峰值强度与初始值的差异超过阈值，则切换图片
        if abs(current_peak_intensity - initial_peak_intensity) > 50:  # 阈值可以根据需要调整
            # 如果有变化，并且不是当前显示的图片，则更换图片
            image_path = rf'D:\app\pycharm\learn\key\image{zone_index+1}.bmp'
            if current_image != image_path:
                img = Image.open(image_path)
                img_cv = np.array(img)
                img_cv_resized = cv2.resize(img_cv, (1920, 1080))  # 全屏显示
                displayed_image = img_cv_resized  # 更新显示的图片
                current_image = image_path  # 更新当前图片
                last_change_time = time.time()  # 记录更改时间
                detected_change = True
                change_detected = True  # 设置变化标志
            break  # 找到一个变化区域后，不需要继续检查其他区域

    # 如果2秒内没有检测到变化，则恢复显示初始图片 image0
    if not detected_change and change_detected and time.time() - last_change_time > 2:
        displayed_image = image_0_cv_resized  # 恢复初始图片
        current_image = r'D:\app\pycharm\learn\key\image0.bmp'
        change_detected = False  # 重置变化标志

    # **显示当前选定的图片（无论是否变化，始终显示一个窗口）**
    cv2.imshow('Displayed Image', displayed_image)

    # 在摄像头启动后 10 秒保存当前帧（仅保存裁剪区域）
    if not image_saved and time.time() - start_time > 10:
        save_path = r'D:\app\pycharm\learn\key\saved_cropped_image.png'  # 设置保存路径
        cv2.imwrite(save_path, gray_cropped)  # 保存裁剪后的区域
        print(f"裁剪区域图像已保存到: {save_path}")
        image_saved = True  # 标记图像已保存

    # 等待 1 毫秒，以便图像刷新
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
