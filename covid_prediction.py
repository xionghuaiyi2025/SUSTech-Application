# 南方科技大学申请材料 - 疫情数据建模
# 作者：张三（常德市一中高三X班）

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 数据加载（示例数据）
data = {
    '日期': ['1/1', '1/2', '1/3', '1/4', '1/5'],
    '新增病例': [120, 135, 148, 165, 182]
}
df = pd.DataFrame(data)

# 简单线性回归
X = [[i] for i in range(len(df))]
y = df['新增病例']
model = LinearRegression().fit(X, y)
pred = model.predict([[5], [6]])

# 可视化
plt.scatter(X, y, color='blue')
plt.plot([5, 6], pred, color='red')
plt.title('常德市疫情趋势预测')
plt.savefig('result.png')  # 保存结果图