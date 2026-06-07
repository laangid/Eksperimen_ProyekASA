import time
import random
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict

# 1. REPRESENTASI DATA
@dataclass
class Item:
    #Representasi satu item dalam inventory Gold & Glory
    id: int
    name: str
    rows: int    # jumlah baris yang ditempati item
    cols: int    # jumlah kolom yang ditempati item
    profit: int  # nilai gold item

    @property
    def size(self) -> int:
        return self.rows * self.cols

    @property
    def efficiency(self) -> float:
        return self.profit / self.size


# Item nyata yang digunakan sebagai dataset eksperimen
GAME_ITEMS = [
    Item(0,  "Black Pearl",            1, 1,  6014),
    Item(1,  "Forest Boots",           2, 1,  2040),
    Item(2,  "Abyssal Skullcrusher",   3, 2, 34420),
    Item(3,  "Rapier",                 3, 1,  8600),
    Item(4,  "Spellbook",              2, 2,  7890),
    Item(5,  "Alchemist Helmet",       2, 1,  2040),
    Item(6,  "Mystic Armor Essence",   1, 1,  7200),
    Item(7,  "Protector Chest",        2, 2,  3750),
    Item(8,  "Mystic Realm Cloak",     2, 2,  2580),
    Item(9,  "Guardian Pants",         2, 2,  3750),
    Item(10, "Golden Requiem Bell",    2, 1,   730),
    Item(11, "Noble Candelabra",       2, 2,   811),
    Item(12, "Gold-inlaid Compass",    1, 1,   315),
    Item(13, "Courtier's Pitcher",     3, 2,  1183),
    Item(14, "Giant Halberd",          4, 1,  7890),
]


def generate_items(n: int, seed: int = 42) -> List[Item]:
    random.seed(seed)

    # Pool ukuran
    size_pool = [
        (1, 1), (1, 1), (1, 1),   
        (2, 1), (2, 1),            
        (3, 1),                    
        (4, 1),                    
        (2, 2), (2, 2), (2, 2),   
        (3, 2), (3, 2),            
    ]

    items = []
    for i in range(n):
        r, c = random.choice(size_pool)
        # Profit berkorelasi kasar dengan ukuran, ditambah variasi acak
        profit = r * c * random.randint(300, 3000)
        # 15% kemungkinan item langka dengan nilai tinggi
        if random.random() < 0.15:
            profit = random.randint(5000, 35000)
        items.append(Item(i, f"Item_{i+1:02d}", r, c, profit))

    return items


# 2. OPERASI GRID
Grid = List[List[bool]]  # True = sel terisi, False = sel kosong


def create_grid(rows: int, cols: int) -> Grid:
    return [[False] * cols for _ in range(rows)]

# Memeriksa apakah item bisa ditempatkan mulai dari posisi (r, c)
def can_place(grid: Grid, item: Item, r: int, c: int) -> bool:
    if r + item.rows > len(grid) or c + item.cols > len(grid[0]):
        return False
    return all(
        not grid[r + dr][c + dc]
        for dr in range(item.rows)
        for dc in range(item.cols)
    )

def place_item(grid: Grid, item: Item, r: int, c: int):
    for dr in range(item.rows):
        for dc in range(item.cols):
            grid[r + dr][c + dc] = True

def remove_item(grid: Grid, item: Item, r: int, c: int):
    for dr in range(item.rows):
        for dc in range(item.cols):
            grid[r + dr][c + dc] = False

# Mengembalikan semua posisi (r, c) yang valid untuk menempatkan item
def get_valid_positions(grid: Grid, item: Item) -> List[Tuple[int, int]]:
    return [
        (r, c)
        for r in range(len(grid))
        for c in range(len(grid[0]))
        if can_place(grid, item, r, c)
    ]

# Mengubah grid 2D menjadi tuple hashable
# dipakai sebagai kunci memoization DP
def grid_to_state(grid: Grid) -> tuple:
    return tuple(cell for row in grid for cell in row)

# state tuple kembali menjadi grid 2D
def state_to_grid(state: tuple, rows: int, cols: int) -> Grid:
    grid = create_grid(rows, cols)
    for i, val in enumerate(state):
        grid[i // cols][i % cols] = bool(val)
    return grid


# 3. BRANCH AND BOUND
# - Branch : untuk setiap item, eksplorasi semua posisi valid atau lewati item
# - Bound  : upper bound = profit saat ini + relaksasi fraksional sisa item
#            yang masih memiliki posisi valid di grid saat ini.
#            Filter spasial dilakukan sebelum relaksasi fraksional sehingga
#            item yang tidak bisa ditempatkan sama sekali tidak dihitung.
# - Prune  : pangkas cabang jika upper bound <= solusi terbaik yang sudah ditemukan

def branch_and_bound(
    items: List[Item],
    grid_rows: int,
    grid_cols: int,
    time_limit: float = 60.0
) -> dict:
    
    best = {'profit': 0, 'placements': [], 'nodes': 0}
    start_time = time.time()
    timed_out = [False]

    def compute_upper_bound(idx: int, current_profit: int, free_cells: int,
                             grid: Grid) -> float:
        ub = float(current_profit)
        remaining = free_cells

        # Filter spasial
        placeable = [
            item for item in items[idx:]
            if get_valid_positions(grid, item)
        ]

        # Relaksasi fraksional pada item yang bisa ditempatkan, diurutkan berdasarkan efisiensi tertinggi
        for item in sorted(placeable, key=lambda x: x.efficiency, reverse=True):
            if remaining <= 0:
                break
            if item.size <= remaining:
                ub += item.profit
                remaining -= item.size
            else:
                ub += item.profit * (remaining / item.size)
                remaining = 0
        return ub

    def bb(idx: int, grid: Grid, current_profit: int, placements: list, free_cells: int):
        if timed_out[0]:
            return
        if time.time() - start_time > time_limit:
            timed_out[0] = True
            return

        best['nodes'] += 1

        # Base case
        if idx == len(items):
            if current_profit > best['profit']:
                best['profit'] = current_profit
                best['placements'] = placements.copy()
            return

        # Pruning
        if compute_upper_bound(idx, current_profit, free_cells, grid) <= best['profit']:
            return

        item = items[idx]

        # Branch 1: tempatkan item di setiap posisi yang valid
        for r, c in get_valid_positions(grid, item):
            place_item(grid, item, r, c)
            placements.append((item.id, r, c))
            bb(idx + 1, grid, current_profit + item.profit,
               placements, free_cells - item.size)
            remove_item(grid, item, r, c)
            placements.pop()

        # Branch 2: lewati item
        bb(idx + 1, grid, current_profit, placements, free_cells)

    grid = create_grid(grid_rows, grid_cols)
    bb(0, grid, 0, [], grid_rows * grid_cols)
    elapsed = time.time() - start_time

    return {
        'algorithm': 'Branch and Bound',
        'profit': best['profit'],
        'nodes_explored': best['nodes'],
        'time': elapsed,
        'timed_out': timed_out[0],
        'optimal': not timed_out[0],
    }


# BAGIAN 4: DYNAMIC PROGRAMMING (MEMOIZED RECURSION)
# State  : (indeks item saat ini, kondisi grid sebagai tuple biner)
#   Rekurens:
#      dp(idx, state) = max(
#         dp(idx+1, state),                              # lewati item
#        profit[idx] + dp(idx+1, state_setelah_taruh)  # tempatkan item
#   )

def dynamic_programming(
    items: List[Item],
    grid_rows: int,
    grid_cols: int,
    time_limit: float = 60.0,
    state_limit: int = 200000
) -> dict:
    
    memo: Dict[tuple, int] = {}
    nodes = [0]
    start_time = time.time()
    timed_out = [False]

    def dp(idx: int, state: tuple) -> int:
        if timed_out[0]:
            return 0
        if time.time() - start_time > time_limit:
            timed_out[0] = True
            return 0

        nodes[0] += 1

        # Base case
        if idx == len(items):
            return 0

        if (idx, state) in memo:
            return memo[(idx, state)]

        # Batas jumlah state untuk mencegah kehabisan memori
        if len(memo) >= state_limit:
            timed_out[0] = True
            return 0

        item = items[idx]
        grid = state_to_grid(state, grid_rows, grid_cols)

        # Pilihan 1: lewati item
        best = dp(idx + 1, state)

        # Pilihan 2: tempatkan item di setiap posisi valid
        for r, c in get_valid_positions(grid, item):
            new_grid = [row[:] for row in grid]
            place_item(new_grid, item, r, c)
            new_state = grid_to_state(new_grid)
            val = item.profit + dp(idx + 1, new_state)
            if val > best:
                best = val

        memo[(idx, state)] = best
        return best

    initial_state = grid_to_state(create_grid(grid_rows, grid_cols))
    result = dp(0, initial_state)
    elapsed = time.time() - start_time

    return {
        'algorithm': 'Dynamic Programming',
        'profit': result,
        'states_explored': nodes[0],
        'memo_size': len(memo),
        'time': elapsed,
        'timed_out': timed_out[0],
        'optimal': not timed_out[0],
    }


# 5. SIMULATED ANNEALING
# Representasi solusi: list of (item_idx, r, c)
# Operasi neighbor   : ADD item baru, REMOVE item, MOVE item, atau SWAP item
# Kriteria penerimaan: delta > 0  → selalu diterima
#                      delta <= 0 → diterima dengan probabilitas exp(delta / T)

def simulated_annealing(
    items: List[Item],
    grid_rows: int,
    grid_cols: int,
    initial_temp: float = 5000.0,
    cooling_rate: float = 0.998,
    min_temp: float = 0.1,
    max_iter: int = 50000,
    seed: int = 0
) -> dict:
    
    random.seed(seed)

    def total_profit(placements: list) -> int:
        return sum(items[i].profit for i, r, c in placements)

    def build_grid(placements: list) -> Grid:
        grid = create_grid(grid_rows, grid_cols)
        for item_idx, r, c in placements:
            place_item(grid, items[item_idx], r, c)
        return grid

    def get_neighbor(placements: list) -> list:
        placed_ids = {i for i, r, c in placements}
        unplaced_ids = [i for i in range(len(items)) if i not in placed_ids]

        ops = []
        if unplaced_ids:
            ops.append('add')
        if placements:
            ops.append('remove')
            ops.append('move')
        if placements and unplaced_ids:
            ops.append('swap')
        if not ops:
            return placements.copy()

        op = random.choice(ops)
        new_placements = placements.copy()

        if op == 'remove':
            new_placements.pop(random.randrange(len(new_placements)))

        elif op == 'add':
            item_idx = random.choice(unplaced_ids)
            item = items[item_idx]
            grid = build_grid(new_placements)
            positions = get_valid_positions(grid, item)
            if positions:
                r, c = random.choice(positions)
                new_placements.append((item_idx, r, c))

        elif op == 'move':
            idx = random.randrange(len(new_placements))
            item_idx, _, _ = new_placements[idx]
            item = items[item_idx]
            temp = new_placements.copy()
            temp.pop(idx)
            grid = build_grid(temp)
            positions = get_valid_positions(grid, item)
            if positions:
                r, c = random.choice(positions)
                new_placements[idx] = (item_idx, r, c)

        elif op == 'swap':
            remove_idx = random.randrange(len(new_placements))
            removed = new_placements.pop(remove_idx) 
            new_item_idx = random.choice(unplaced_ids)
            new_item = items[new_item_idx]
            grid = build_grid(new_placements)
            positions = get_valid_positions(grid, new_item)
            if positions:
                r, c = random.choice(positions)
                new_placements.append((new_item_idx, r, c))
            else:
                new_placements.insert(remove_idx, removed)

        return new_placements

    # Solusi awal: greedy
    current = []
    grid = create_grid(grid_rows, grid_cols)
    for i in sorted(range(len(items)), key=lambda i: items[i].efficiency, reverse=True):
        item = items[i]
        positions = get_valid_positions(grid, item)
        if positions:
            r, c = positions[0]
            place_item(grid, item, r, c)
            current.append((i, r, c))

    best = current.copy()
    best_profit = total_profit(best)
    current_profit = best_profit

    temp = initial_temp
    iterations = 0
    start_time = time.time()

    while temp > min_temp and iterations < max_iter:
        neighbor = get_neighbor(current)
        neighbor_profit = total_profit(neighbor)
        delta = neighbor_profit - current_profit

        # Terima neighbor jika lebih baik, atau secara probabilistik jika lebih buruk
        if delta > 0 or random.random() < math.exp(delta / temp):
            current = neighbor
            current_profit = neighbor_profit
            if current_profit > best_profit:
                best_profit = current_profit
                best = current.copy()

        temp *= cooling_rate
        iterations += 1

    elapsed = time.time() - start_time

    return {
        'algorithm': 'Simulated Annealing',
        'profit': best_profit,
        'iterations': iterations,
        'final_temp': round(temp, 4),
        'time': elapsed,
        'timed_out': False,
        'optimal': False,
    }


# 6. RUNNER EKSPERIMEN
SA_RUNS = 20  # Jumlah run independen untuk Simulated Annealing


def run_sa_multirun(items, grid_rows, grid_cols, n_runs=SA_RUNS):
    profits = []
    times = []
    for seed in range(n_runs):
        r = simulated_annealing(items, grid_rows, grid_cols, seed=seed)
        profits.append(r['profit'])
        times.append(r['time'])

    mean_p = sum(profits) / n_runs
    std_p = (sum((p - mean_p) ** 2 for p in profits) / n_runs) ** 0.5

    return {
        'sa_profit_mean': round(mean_p, 2),
        'sa_profit_max': max(profits),
        'sa_profit_min': min(profits),
        'sa_profit_std': round(std_p, 2),
        'sa_time_mean': round(sum(times) / n_runs, 4),
        'sa_runs': n_runs,
    }


def run_one(label, items, grid_rows, grid_cols, bb_limit=30.0, dp_limit=30.0):
    print(f"\n{'─'*65}")
    print(f"  {label}  |  {len(items)} item  |  Grid {grid_rows}×{grid_cols}")
    print(f"{'─'*65}")

    row = {'label': label, 'n_items': len(items), 'grid': f'{grid_rows}x{grid_cols}'}

    # Branch and Bound
    print("  [B&B] Menjalankan...", end='', flush=True)
    r = branch_and_bound(items, grid_rows, grid_cols, time_limit=bb_limit)
    row.update({
        'bb_profit': r['profit'], 'bb_time': round(r['time'], 4),
        'bb_nodes': r['nodes_explored'], 'bb_timeout': r['timed_out'],
    })
    flag = " ← TIMEOUT" if r['timed_out'] else (" ← OPTIMAL" if r['optimal'] else "")
    print(f" profit={r['profit']:>8,}  waktu={r['time']:>7.4f}s  nodes={r['nodes_explored']:>8,}{flag}")

    # Dynamic Programming
    print("  [DP]  Menjalankan...", end='', flush=True)
    r = dynamic_programming(items, grid_rows, grid_cols, time_limit=dp_limit)
    row.update({
        'dp_profit': r['profit'], 'dp_time': round(r['time'], 4),
        'dp_states': r['states_explored'], 'dp_timeout': r['timed_out'],
    })
    flag = " ← TIMEOUT" if r['timed_out'] else (" ← OPTIMAL" if r['optimal'] else "")
    print(f" profit={r['profit']:>8,}  waktu={r['time']:>7.4f}s  states={r['states_explored']:>7,}{flag}")

    # Simulated Annealing (multi-run)
    print(f"  [SA]  Menjalankan {SA_RUNS}x...", end='', flush=True)
    sa = run_sa_multirun(items, grid_rows, grid_cols)
    row.update(sa)
    print(f" mean={sa['sa_profit_mean']:>8,.1f}  max={sa['sa_profit_max']:>8,}"
          f"  min={sa['sa_profit_min']:>8,}  std={sa['sa_profit_std']:>7,.1f}"
          f"  waktu≈{sa['sa_time_mean']:.4f}s")

    return row


def run_experiments():
    GRID_ROWS, GRID_COLS = 7, 5

    print("\n" + "="*65)
    print("  EKSPERIMEN: Optimasi Profit Inventory Grid — Gold & Glory")
    print("="*65)

    results = []

    # Setiap ukuran dataset menggunakan seed berbeda
    dataset_seeds = {5: 10, 8: 20, 10: 30, 12: 40, 15: 50}

    for n, seed in dataset_seeds.items():
        items = generate_items(n, seed=seed)
        results.append(run_one(f"Sintetis-{n}", items, GRID_ROWS, GRID_COLS))

    # Eksperimen dengan item asli dari game
    results.append(run_one("Game-Items (15)", GAME_ITEMS, GRID_ROWS, GRID_COLS))

    # Simpan hasil ke CSV
    fieldnames = list(results[0].keys())
    with open('results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*65}")
    print("  Hasil tersimpan di: results.csv")
    print("="*65)

    return results


# ENTRY POINT
if __name__ == "__main__":
    run_experiments()
