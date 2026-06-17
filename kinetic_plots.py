import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# DATA DEFINITIONS (Tier 3 Michaelis-Menten & Microenvironment Metrics)
# ----------------------------------------------------------------
# Km for human PHD2 is 105 uM. Vmax is normalized to 100%
Km = 105
Vmax = 100

# Oxygen range simulating local accumulation (0 to 1600 uM)
oxygen_concentrations = np.linspace(0, 1600, 500)
enzyme_velocities = (Vmax * oxygen_concentrations) / (Km + oxygen_concentrations)

# Specific data point from our predictive pipeline (Oxygen = 1560 uM)
target_O2 = 1560
target_v = (Vmax * target_O2) / (Km + target_O2)

# Pie chart segments for tissue oxygenation state after bacterial deployment
pie_labels = ['Reactivated PHD2\n(Normoxic Matrix)', 'Residual Hypoxia\n(Deep Microscopic Cores)']
pie_sizes = [target_v, (100 - target_v)]
pie_colors = ['#2ca02c', '#d62728'] # Deep green for safe/active, crimson for hypoxia

# ----------------------------------------------------------------
# PLOT SETUP: Side-by-Side Panels (Publication standard)
# ----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plt.rcParams['font.family'] = 'sans-serif'

# PANEL 1: Michaelis-Menten Kinetics Line Plot
ax1.plot(oxygen_concentrations, enzyme_velocities, color='#1f77b4', linewidth=2.5, label='PHD2 Reaction Velocity')
ax1.scatter(target_O2, target_v, color='#d62728', s=100, zorder=5, label=f'S. elongatus Delivery ({target_v:.1f}% Vmax)')
ax1.axhline(y=target_v, color='#d62728', linestyle='--', alpha=0.6)
ax1.axvline(x=target_O2, color='#d62728', linestyle='--', alpha=0.6)

ax1.set_title('Human PHD2 Enzyme Reaction Kinetics', fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel('Microenvironmental Oxygen Concentration ($[O_2]$, $\mu$M)', fontsize=11)
ax1.set_ylabel('Fractional Enzyme Velocity ($v$, % of $V_{max}$)', fontsize=11)
ax1.set_xlim(0, 1600)
ax1.set_ylim(0, 110)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right', fontsize=10)

# PANEL 2: Microenvironmental State Allocation Pie Chart
wedges, texts, autotexts = ax2.pie(
    pie_sizes, 
    labels=pie_labels, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=pie_colors,
    textprops=dict(fontsize=11),
    wedgeprops=dict(width=0.6, edgecolor='w', linewidth=2) # Modern donut style
)
plt.setp(autotexts, size=11, weight="bold", color="white")
ax2.set_title('Tumor Microenvironment State Post-Treatment', fontsize=13, fontweight='bold', pad=15)

# Final clean layout adjustments
plt.tight_layout()
plt.savefig('figure2_kinetic_analysis.png', dpi=300)
print("SUCCESS: figure2_kinetic_analysis.png has been generated in your folder!")