import os
import cv2
import numpy as np

# 設定資料夾和檔案類型
data_folder = 'C:\\train'
file_type = '.JPG'

# 設定翻轉的模式和裁切的範圍
flip_mode = -1  # 1表示左右翻轉，0表示上下翻轉，-1表示左右上下翻轉

# 設定旋轉的角度範圍和比例
angle_range = 2  # 旋轉角度的範圍
scale_range = 0.4  # 旋轉比例的範圍

# 設定亮度下降的比例
brightness_scale = 0.95

# 設定對比度的增加值
contrast_increase = 5

# 設定腐蝕操作的核大小
kernel_size = 1

# 遍歷資料夾內的所有圖片檔案
count = 1
for file in os.listdir(data_folder):
    if file.endswith(file_type):
        # 讀取圖片
        img = cv2.imread(os.path.join(data_folder, file))

        # 減少反光值
        alpha = brightness_scale  # 調整亮度的比例
        beta = -contrast_increase  # 調整對比度的值
        brightness_and_contrast_adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # 去除反光
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        erosion = cv2.erode(brightness_and_contrast_adjusted, kernel, iterations=1)

        # 翻轉圖片
        flipped_img = cv2.flip(erosion, flip_mode)

        # 旋轉圖片
        rotated_img = cv2.rotate(flipped_img, cv2.ROTATE_90_CLOCKWISE)

        # 取得檔案副檔名
        file_ext = os.path.splitext(file)[1]

        # 構造新檔名
        new_file_name = '{:04d}{}'.format(count, file_ext)

        # 如果新檔名已經存在，就更改新檔名直到不存在為止
        while os.path.exists(os.path.join(data_folder, new_file_name)):
            count += 1
            new_file_name = '{:04d}{}'.format(count, file_ext)

        # 判斷是否存在舊檔案
        old_file_path = os.path.join(data_folder, file)
        if os.path.exists(old_file_path):
            # 儲存處理後的圖片
            cv2.imwrite(os.path.join(data_folder, new_file_name), rotated_img)

            # 刪除原始檔案
            os.remove(old_file_path)

