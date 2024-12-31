import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # 读取视差预测图和真实图
    disp_pred = cv2.imread('disp_pred.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    disp_gt = cv2.imread('disp_gt.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # 创建误差图,初始化为真实图
    error_map = cv2.cvtColor(disp_gt, cv2.COLOR_GRAY2BGR)

    # 计算误差
    mask = disp_gt > 0  # 真实图中有效视差的掩码
    error = np.abs(disp_pred - disp_gt)
    
    # 标记误差大于3px的位置
    bad_pixels = (error > 3) & mask
    
    # 在误差图上用红色标记错误估计的像素
    error_map[bad_pixels] = [0, 0, 255]  # BGR格式,红色

    # 显示误差图
    plt.figure(figsize=(10,5))
    plt.imshow(cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB))
    plt.title('Disparity Error Map')
    plt.colorbar()
    plt.show()

    # 保存误差图
    cv2.imwrite('error_map.png', error_map)

