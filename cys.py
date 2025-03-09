import numpy as np
import matplotlib.pyplot as plt

# 定义系数矩阵和常数项
A = np.array([[20, 7], [7, 20]])
B = np.array([144000, 174000])

# 解线性方程组
solution = np.linalg.solve(A, B)
x, y = solution[0], solution[1]

print(f"19英寸生产量: {round(x)} 台")
print(f"21英寸生产量: {round(y)} 台")


#显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 定义利润函数
def profit(x, y):
    return 144*x + 174*y - 0.007*x*y - 0.01*x**2 - 0.01*y**2 - 400000

# 生成数据点
x = np.linspace(3000, 6000, 100)
y = np.linspace(5000, 9000, 100)
X, Y = np.meshgrid(x, y)
Z = profit(X, Y)

# 绘制等高线图
plt.figure(figsize=(10, 6))
contour = plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)
plt.scatter(4735, 7043, c='red', s=50, label='最大利润点 (4735, 7043)')  # 标记极值点
plt.xlabel('19英寸生产量 (台)', fontsize=12)
plt.ylabel('21英寸生产量 (台)', fontsize=12)
plt.title('利润函数等高线图', fontsize=14)
plt.grid(alpha=0.3)
plt.legend()
plt.show()