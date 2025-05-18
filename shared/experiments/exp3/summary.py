import re
from collections import defaultdict

def parse_hpl_log(log_file):
    """
    解析 HPL 日志文件，提取性能数据并按进程数分组。
    """
    # 正则表达式匹配HPL性能数据行
    pattern = re.compile(
        r'WR\S+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+e[+-]\d+)'
    )
    
    results = defaultdict(list)
    
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                N, NB, P_str, Q_str, Time, Gflops = match.groups()
                P = int(P_str)
                Q = int(Q_str)
                procs = P * Q

                # 转换 GFLOPS 字段为浮点数
                try:
                    gflops_value = float(Gflops)
                except ValueError:
                    gflops_value = 0.0

                theoretical_peak = 79.2 # Ryzen 9 5900X 双精度理论峰值 (GFLOPS per core)
                efficiency = (gflops_value / (theoretical_peak * procs)) * 100

                results[procs].append({
                    'N': N,
                    'NB': NB,
                    'P': P,
                    'Q': Q,
                    'Time': Time,
                    'Gflops': f"{gflops_value:.2f}",
                    'Efficiency': f"{efficiency:.2f}%",
                    'Peak': f"{theoretical_peak * procs:.2f}"
                })
    
    return results

def filter_top_gflops(results):
    """
    对每个进程数（procs），仅保留实测 GFLOPS 最高的记录。
    """
    top_results = {}

    for procs, entries in results.items():
        # 按照 GFLOPS 降序排序，并取第一个
        best_entry = max(entries, key=lambda x: float(x['Gflops']))
        top_results[procs] = best_entry

    return top_results

def generate_summary_table(top_results):
    """
    打印汇总表格，仅展示每个进程数下最高 GFLOPS 的配置。
    """
    print("| 进程数 | 矩阵大小(N) | 分块大小(NB) | P | Q | 时间(s) | 实测GFLOPS | 效率(%) | 理论峰值(GFLOPS) |")
    print("|--------|-------------|--------------|---|---|---------|------------|----------|------------------|")

    for procs in sorted(top_results.keys()):
        entry = top_results[procs]
        print(f"| {procs} "
              f"| {entry['N']} "
              f"| {entry['NB']} "
              f"| {entry['P']} "
              f"| {entry['Q']} "
              f"| {entry['Time']} "
              f"| {entry['Gflops']} "
              f"| {entry['Efficiency']} "
              f"| {entry['Peak']} |")

if __name__ == "__main__":
    log_file = "/media/drew/data1/SHU-Computer-Arch/shared/experiments/exp3/log/np-24/hpl_naive_20250429_185253.log"  # 替换为你自己的日志路径
    results = parse_hpl_log(log_file)
    top_results = filter_top_gflops(results)
    generate_summary_table(top_results)