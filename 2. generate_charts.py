import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# BACA DATA
def baca_csv(path='results_final.csv'):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

data = baca_csv()

# Pisahkan data sintetis (untuk chart tren) dan semua data (untuk chart profit)
sintetis = [d for d in data if d['label'].startswith('Sintetis')]
semua    = data  

label_sintetis = [d['label'] for d in sintetis]
n_items        = [int(d['n_items']) for d in sintetis]
label_semua    = [d['label'] for d in semua]
x_semua        = np.arange(len(semua))


# PENGATURAN UMUM
WARNA_BB = '#2563EB'   
WARNA_DP = '#DC2626'   
WARNA_SA = '#16A34A'   

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

TIMEOUT_DETIK = 30.0  # batas waktu eksperimen


# CHART 1: PERBANDINGAN WAKTU EKSEKUSI (dataset sintetis)
fig1, ax1 = plt.subplots(figsize=(8, 5))

# Waktu eksekusi dan status timeout untuk setiap dataset
bb_waktu   = [float(d['bb_time']) for d in sintetis]
dp_waktu   = [float(d['dp_time']) for d in sintetis]
sa_waktu   = [float(d['sa_time_mean']) for d in sintetis]
bb_timeout = [d['bb_timeout'] == 'True' for d in sintetis]
dp_timeout = [d['dp_timeout'] == 'True' for d in sintetis]

ax1.plot(n_items, bb_waktu, 'o-', color=WARNA_BB, linewidth=2,
         markersize=7, label='Branch and Bound')
ax1.plot(n_items, dp_waktu, 's-', color=WARNA_DP, linewidth=2,
         markersize=7, label='Dynamic Programming')
ax1.plot(n_items, sa_waktu, '^-', color=WARNA_SA, linewidth=2,
         markersize=7, label='Simulated Annealing')

# Tandai titik yang timeout dengan marker berbeda
for i, (t, w) in enumerate(zip(bb_timeout, bb_waktu)):
    if t:
        ax1.plot(n_items[i], w, 'x', color=WARNA_BB,
                 markersize=12, markeredgewidth=2.5)
for i, (t, w) in enumerate(zip(dp_timeout, dp_waktu)):
    if t:
        ax1.plot(n_items[i], w, 'x', color=WARNA_DP,
                 markersize=12, markeredgewidth=2.5)

# Garis putus-putus batas timeout
ax1.axhline(y=TIMEOUT_DETIK, color='gray', linestyle='--',
            linewidth=1.2, alpha=0.7)
ax1.text(n_items[-1] + 0.1, TIMEOUT_DETIK + 0.4,
         'Batas timeout (30s)', color='gray', fontsize=9, va='bottom')

# Tanda ✕ sebagai keterangan timeout
timeout_patch = mpatches.Patch(color='gray', label='✕ = timeout tercapai')

ax1.set_xlabel('Jumlah Item', fontsize=11)
ax1.set_ylabel('Waktu Eksekusi (detik)', fontsize=11)
ax1.set_title('Perbandingan Waktu Eksekusi Ketiga Algoritma\n(Dataset Sintetis, Grid 7×5)')
ax1.set_xticks(n_items)
ax1.legend(handles=[
    *ax1.get_lines()[:3],
    timeout_patch
], fontsize=9)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(bottom=-1)

plt.tight_layout()
plt.savefig('chart_waktu.png', bbox_inches='tight')
print("Tersimpan: chart_waktu.png")
plt.close()


# CHART 2: PERBANDINGAN KUALITAS SOLUSI (semua dataset)
fig2, ax2 = plt.subplots(figsize=(10, 5.5))

lebar  = 0.25
x_pos  = np.arange(len(semua))

bb_profit  = [float(d['bb_profit']) for d in semua]
dp_profit  = [float(d['dp_profit']) for d in semua]
sa_mean    = [float(d['sa_profit_mean']) for d in semua]
sa_std     = [float(d['sa_profit_std']) for d in semua]
bb_timeout = [d['bb_timeout'] == 'True' for d in semua]
dp_timeout = [d['dp_timeout'] == 'True' for d in semua]

bar_bb = ax2.bar(x_pos - lebar, bb_profit, lebar,
                  color=WARNA_BB, alpha=0.85, label='Branch and Bound')
bar_dp = ax2.bar(x_pos,          dp_profit, lebar,
                  color=WARNA_DP, alpha=0.85, label='Dynamic Programming')
bar_sa = ax2.bar(x_pos + lebar, sa_mean,   lebar,
                  color=WARNA_SA, alpha=0.85, label='Simulated Annealing (rata-rata)')

# Error bar untuk SA
ax2.errorbar(x_pos + lebar, sa_mean, yerr=sa_std,
             fmt='none', color='black', capsize=4, linewidth=1.5)

# Penanda bar yang timeout (arsiran)
for i, t in enumerate(bb_timeout):
    if t:
        bar_bb[i].set_hatch('///')
        bar_bb[i].set_edgecolor('white')
for i, t in enumerate(dp_timeout):
    if t:
        bar_dp[i].set_hatch('///')
        bar_dp[i].set_edgecolor('white')

# Keterangan arsiran
timeout_patch = mpatches.Patch(facecolor='lightgray', hatch='///',
                                edgecolor='gray', label='⚠ timeout (hasil tidak optimal)')

ax2.set_xlabel('Dataset', fontsize=11)
ax2.set_ylabel('Profit (gold)', fontsize=11)
ax2.set_title('Perbandingan Kualitas Solusi Ketiga Algoritma\n(Grid 7×5, SA = rata-rata 20 run, error bar = std dev)')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(label_semua, rotation=15, ha='right', fontsize=10)
ax2.legend(handles=[bar_bb, bar_dp, bar_sa, timeout_patch], fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Nilai profit di atas bar B&B (yang optimal)
for i, (p, t) in enumerate(zip(bb_profit, bb_timeout)):
    if not t:
        ax2.text(x_pos[i] - lebar, p + max(bb_profit) * 0.01,
                 f'{int(p):,}', ha='center', va='bottom', fontsize=8, color=WARNA_BB)

plt.tight_layout()
plt.savefig('chart_profit.png', bbox_inches='tight')
print("Tersimpan: chart_profit.png")
plt.close()

print("\nSelesai. Dua file PNG sudah siap untuk makalah.")
