import numpy as np

# Constants
B = 50e-6  # Earth's magnetic field in Tesla (50 microT)
k_B = 1.380649e-23  # Boltzmann constant in J/K
T = 298  # Temperature in Kelvin (room temperature)
eta_water = 8.9e-4  # Dynamic viscosity of water in Pa.s
eta = 5.5 * eta_water  # Given medium is 5.5 times as viscous as water
mu_magnetite = 4.46e-23  # Magnetic moment of magnetite in A·m²
Ms =  4.46 * 10e-5  # Saturation magnetization of magnetite in A/m
diameters = [35e-9, 92.6e-9, 620.4e-9, 250e-9]  # Diameters in meters

# Function to calculate the volume of a sphere
def volume_of_sphere(diameter):
    return (4/3) * np.pi * (diameter / 2)**3

# Function to calculate tau (relaxation time)
def calculate_tau(eta, V, T):
    return (2* 3 * eta * V) / (k_B * T)

# Function to calculate sensitivity
def calculate_sensitivity(B, T, tau, mu):
    return np.sqrt(B) * np.sqrt((2 * k_B * T * tau) / mu)

# Function to calculate sensitivity
def calculate_sensitivity2(B, eta, V, mu):
    return np.sqrt(B) * np.sqrt((3 * eta * V) / mu)

# Function to calculate sensitivity
def calculate_sensitivity3(B, eta, Ms):
    return np.sqrt(B) * np.sqrt((2* 3 * eta) / Ms)

# Calculate sensitivities for each diameter
sensitivities = []
sensitivities2 = []
sensitivities3 = []

for d in diameters:
    V = volume_of_sphere(d)  # Volume of the magnetite grain
    tau = calculate_tau(eta, V, T)  # Relaxation time tau
    sensitivity = calculate_sensitivity(B, T, tau, mu_magnetite)  # Sensitivity
    sensitivities.append(sensitivity)
    
    sensitivity2 = calculate_sensitivity2(B, eta, V, mu_magnetite)  # Sensitivity
    sensitivities2.append(sensitivity2)
    
    sensitivity3 = calculate_sensitivity3(B, eta, Ms)  # Sensitivity
    sensitivities3.append(sensitivity3)

# Display the results
for i, d in enumerate(diameters):
    print(f"Diameter: {d*1e9:.1f} nm, Sensitivity: {sensitivities[i]:.4e} T/√Hz")
    print(f"Diameter: {d*1e9:.1f} nm, 2nd Sensitivity: {sensitivities2[i]:.4e} T/√Hz")
    print(f"Diameter: {d*1e9:.1f} nm, 3rd Sensitivity: {sensitivities3[i]:.4e} T/√Hz\n\n\n")
