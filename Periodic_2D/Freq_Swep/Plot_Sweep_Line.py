import matplotlib.pyplot as plt
import pandas as pd

# Load the data
file_path = 'TE1_DS4_Line.xlsx'
df = pd.read_excel(file_path)

# Clean data: keep rows that have at least one of Alpha or Beta present
df = df.dropna(subset=['Alpha_Mode_1', 'Beta_Mode_1'], how='all')

# Convert Frequency from Hz to GHz
freq_GHz = df['Frequency (Hz)'] / 1e9
beta = df['Beta_Mode_1']
alpha = df['Alpha_Mode_1']

# Create the plot
fig, ax1 = plt.subplots(figsize=(5, 4), dpi=300)
ax1.plot(freq_GHz, beta, 'b.-', markersize=4)
ax1.set_xlabel('Frequency (GHz)')
ax1.set_ylabel(r'Normalized Beta ($\beta/k_0)$', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid()
ax1.set_ylim(-1, 1)
ax1.set_xlim(20, 30)

# Create a second y-axis for Alpha
ax2 = ax1.twinx()
ax2.plot(freq_GHz, abs(alpha), 'r.-', markersize=4)
ax2.set_ylabel(r'Normalized Alpha ($\alpha/k_0)$', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(-0.1, 0.1)

plt.title("Dispersion of TE1 Surface Wave Based LWA \n"
          r"$\varepsilon_r=10.2$, $h=2.5$ mm, $p=5$ mm, $ws1=0.5$ mm, $ws2=0.4$ mm ")
plt.show()
