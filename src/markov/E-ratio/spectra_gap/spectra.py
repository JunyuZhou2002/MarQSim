import numpy as np
import pandas as pd
import xlsxwriter

# Load the matrix from the .npy file
m1 = np.load('P_0.npy')
m2 = np.load('P_1.npy')
m3 = np.load('P_2.npy')

# print(m1)
# print(m2)

P0 = m1
P1 = 0.4*m1+0.6*m2
P2 = 0.4*m1+0.3*m2+0.3*m3

print(P0)
print(P1)

# Calculate the eigenvalues
eigenvalues = np.linalg.eigvals(P1)

# Sort the eigenvalues by their absolute values in descending order
sorted_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

print(sorted_eigenvalues)


# Convert the array to a DataFrame
df = pd.DataFrame(sorted_eigenvalues, columns=["Spectra"])

# Save the DataFrame to an Excel file
file_path = "list.xlsx"
df.to_excel(file_path, index=False)

# sorted_eigenvalues = [x for x in sorted_eigenvalues if x <= 0.99999]

# print(np.mean(sorted_eigenvalues))

# # Identify the largest eigenvalue (considering its multiplicity algebraically but not geometrically)
# largest_eigenvalue = sorted_eigenvalues[0]
    
#     # Remove the largest eigenvalue and find the next distinct one
# for value in sorted_eigenvalues:
#     if value < largest_eigenvalue-0.00001:
#         second_largest_eigenvalue = value
#         break
    

# print(sorted_eigenvalues)

# # Calculate the spectral gap

# spectral_gap = sorted_eigenvalues[0] - second_largest_eigenvalue

# # Print the spectral gap
# print("Spectral gap of the matrix is:", spectral_gap)

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

# # Define the model function
# def model_func(x, alpha):
#     # return (x-0.0202) ** alpha
#     return x ** alpha + 0.0202

# # Sample data
# y_data = np.array([0.020276, 0.026839, 0.020209, 0.035804, 0.020993, 0.023566])
# x_data = np.array([0, 0.3, 0.23, 0.4, 0.1, 0.31])
# # Fit the model to the data
# popt, pcov = curve_fit(model_func, x_data, y_data)
# alpha = popt[0]

# # Print the result
# print(f"Fitted alpha: {alpha}")

# # Generate a smooth line for the fit
# x_fit = np.linspace(min(x_data), max(x_data), 500)
# y_fit = model_func(x_fit, alpha)

# # Plot the data and the fit
# plt.scatter(x_data, y_data, label='Data Point', color='blue', marker='x')
# plt.plot(x_fit, y_fit, label=f'Fit: $y = x^{{{alpha:.2f}}}+0.02$', color='red')
# plt.xlabel('Spectra')
# plt.ylabel('Standard Deviation')
# plt.legend()
# # plt.show()
# plt.savefig('std' + '.png')
