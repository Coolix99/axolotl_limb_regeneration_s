import matplotlib.pyplot as plt

# Set default font settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['font.serif'] = ['Nimbus Roman']  # Use Times New Roman on Windows, Nimbus Roman on Linux
plt.rcParams['figure.dpi'] = 300
plt.rcParams['lines.linewidth'] = 0.5

# Use the "Set2" color palette
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.get_cmap("Dark2").colors)

# Set axis label font size and weight
plt.rcParams['axes.labelsize'] = 12
#plt.rcParams['axes.labelweight'] = 'bold'

# Set axis ticks
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Set legend font size and location
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.loc'] = 'upper left'
plt.rcParams['legend.frameon'] = False  # Disable legend border

# Define default figure sizes in inches
default_figsize = {
    "small": (9 * 0.3937, 6 * 0.3937),  # 9 cm x 6 cm
    "medium": (11 * 0.3937, 11 * 0.3937),  # 11 cm x 11 cm
    "large": (18 * 0.3937, 22 * 0.3937)   # 18 cm x 22 cm
}