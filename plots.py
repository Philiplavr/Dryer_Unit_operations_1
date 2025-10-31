from matplotlib import pyplot as plt
import numpy as np

param_list = [(2.06097892, 0.01764726),(1.95213585, 0.01942927)]
x0_list = [2.931, 2.931]


# Define model function
def x_func(xo, t, xinf, k):
    return xinf + (xo - xinf) * np.exp(-k * t)


for i, param in enumerate(param_list):
    x_inf, k_fit = param
    t_new = np.linspace(0, 94, 200)
    plt.plot(t_new, x_func(x0_list[i],t_new, *param), '--', label=f'Μπανάνα {i+1}')

plt.xlabel("t [min]")
plt.ylabel("x(t)")
plt.title("Ξυραντήρας - Μπανάνες")
plt.legend()
plt.grid(True)
plt.show()

param_list = [(9.09700461, 8.44893103e-03), (6.99311386, 0.01120436)]
x0_list = [25, 25]

for i, param in enumerate(param_list):
    x_inf, k_fit = param
    t_new = np.linspace(0, 94, 200)
    plt.plot(t_new, x_func(x0_list[i],t_new, *param), '--', label=f'Αγγούρι {i+1}')

plt.xlabel("t [min]")
plt.ylabel("x(t)")
plt.title("Ξυραντήρας - Αγγούρια")
plt.legend()
plt.grid(True)
plt.show()

param_list = [(2.16046182, 0.00609026),(2.21482698, 0.00762149)]
x0_list = [2.931,2.931]

for i, param in enumerate(param_list):
    x_inf, k_fit = param
    t_new = np.linspace(0, 94, 200)
    plt.plot(t_new, x_func(x0_list[i],t_new, *param), '--', label=f'Μπανάνα {i+1}')

plt.xlabel("t [min]")
plt.ylabel("x(t)")
plt.title("Φούρνος- Μπανάνες")
plt.legend()
plt.grid(True)
plt.show()

param_list = [(1.58427002e+01, 3.90383051e-03), (1.40031536e+01, 5.28419262e-03)]
x0_list = [25, 25]

for i, param in enumerate(param_list):
    x_inf, k_fit = param
    t_new = np.linspace(0, 94, 200)
    plt.plot(t_new, x_func(x0_list[i], t_new, *param), '--', label=f'Αγγούρι {i+1}')

plt.xlabel("t [min]")
plt.ylabel("x(t)")
plt.title("Φούρνος - Αγγούρια")
plt.legend()
plt.grid(True)
plt.show()
