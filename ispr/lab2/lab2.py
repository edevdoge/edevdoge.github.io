from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

class1 = np.array([[4.12, 52, 132, 8.8], [2.27, 53, 115, 3.9], [4.33, 47, 137, 4.7], [3.45, 56, 112, 1.7], [1.6, 41, 144, 6.6], [3.84, 63, 148, 3.5], [3.19, 46, 110, 5.3]])

# example class1
# class1 = np.array([[22.4, 17.1, 22], [224.2, 17.1, 23], [151.8, 14.9, 21.5], [147.3, 13.6, 28.7], [152.3, 10.5, 10.2]])

print("Елементи першої вибірки:")
print(class1);

class2 = np.array([[1.69, 28, 116, 2.34], [2.31, 31, 114, 4.43], [1.72, 42, 115, 6.43], [1.83, 37, 105, 6.30], [2.14, 29, 112, 8.47], [1.18, 25, 108, 9.50], [3.06, 22, 122, 7.49]])

# example class2
# class2 = np.array([[46.8, 4.4, 11.1], [29, 5.5, 6.1], [52.1, 4.2, 11.8], [37.1, 5.5, 11.9], [64, 4.2, 12.9]])

print("\nЕлементи другої вибірки:")
print(class2);

combined_classes = np.vstack((class1, class2))

y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

# example fit array
# y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

lda = LinearDiscriminantAnalysis()

lda.fit(combined_classes, y)

# example search predictions
# z_values = np.array([[75, 9.6, 18.5], [95, 12.5, 16.1]])

z1 = np.array([[3.5, 62, 121, 5.45]])
z2 = np.array([[1.71, 49, 128, 6.41]])

print("\nПерший масив для пошуку серед вибірок:")
print(z1);

print("\nДругий масив для пошуку серед вибірок:")
print(z2);

combined_z = np.vstack((z1, z2))

predictions = lda.predict(combined_z)

print("\nПерший масив належить до вибірки: " + str(predictions[0]))
print("\nДругий масив належить до вибірки: " + str(predictions[1]))
