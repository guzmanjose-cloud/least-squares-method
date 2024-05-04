import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import pandas as pd


file_path = "income.csv"
df = pd.read_csv(file_path)
df = df.dropna()

x1 = df["age"].values
x2 = df["experience"].values
y = df["income"].values

def least_squares_method(x1, x2, y):
    total_x1 = sum(x1)
    total_x2 = sum(x2)
    total_y = sum(y)
    total_x1y = sum(x1 * y)
    total_x2y = sum(x2 * y)
    total_x1_sq = sum(x1 ** 2)
    total_x2_sq = sum(x2 ** 2)

    m1 = (len(y) * total_x1y - total_x1 * total_y) / (len(y) * total_x1_sq - total_x1 ** 2)
    m2 = (len(y) * total_x2y - total_x2 * total_y) / (len(y) * total_x2_sq - total_x2 ** 2)
    b = (total_y - m1 * total_x1 - m2 * total_x2) / len(y)
    
    return m1, m2, b

def calculate_predictions(m1, m2, b, x1, x2):
    return m1 * np.array(x1) + m2 * np.array(x2) + b

def gradient_descent(x1, x2, y, learning_rate=0.0000001, iterations=100):
    m1, m2, b = least_squares_method(x1, x2, y)
    
    for _ in range(iterations):
        predictions = (m1 * x1 + m2 * x2 + b)
        
        d_m1 = (1/len(y)) * np.sum(x1 * (y - predictions))
        d_m2 = (1/len(y)) * np.sum(x2 * (y - predictions))
        d_b = (1/len(y)) * np.sum(y - predictions)

        m1 -= learning_rate * d_m1
        m2 -= learning_rate * d_m2
        b -= learning_rate * d_b

    return m1, m2, b

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=.40)

m1_ls, m2_ls, b_ls = least_squares_method(x1_train, x2_train, y_train)
print("Least Squares Method Coefficients:", m1_ls, m2_ls, b_ls)

m1_gd, m2_gd, b_gd = gradient_descent(x1_train, x2_train, y_train)
print("Gradient Descent Coefficients:", m1_gd, m2_gd, b_gd)

y_predict = calculate_predictions(m1_gd, m2_gd, b_gd, x1_test, x2_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1_range = np.linspace(min(x1_test), max(x1_test), 100)
x2_range = np.linspace(min(x2_test), max(x2_test), 100)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
y_predict_grid = calculate_predictions(m1_gd, m2_gd, b_gd, x1_grid, x2_grid)
ax.plot_surface(x1_grid, x2_grid, y_predict_grid, color='black', alpha=0.5, label='Predicted Surface')

ax.scatter(x1_train, x2_train, y_train, c='b', label='Training Data')

ax.scatter(x1_test, x2_test, y_test, c='r', label='Test Data')

ax.set_xlabel('Age')
ax.set_ylabel('Experience')
ax.set_zlabel('Income')

plt.title('3D Scatter Plot with Predicted Surface')

plt.show()


data = {
    'income': y_test,
    'income prediced': y_predict
}

df = pd.DataFrame(data)

print(df)
