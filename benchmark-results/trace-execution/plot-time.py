import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import sys
import re
import pandas as pd
import numpy

v = []
for fname in os.listdir('data'):
    f = open('data/' + fname)
    lines = [l.rstrip() for l in f.readlines()]

    r = r'benchmark-([a-zA-Z0-9_-]*)-(native|remote)-(remote|local)-.*'
    m = re.search(r, fname)
    benchmark_name = m.group(1)
    benchmark_type = m.group(2)
    benchmark_flag = m.group(3)

    if benchmark_type == 'native':
        for l in lines:
            v.append([benchmark_name, 'Native', float(l)])
    elif benchmark_type == 'remote' and benchmark_flag == 'local':
        for l in lines:
            v.append([benchmark_name, 'Local TCP', float(l)])
    elif benchmark_type == 'remote' and benchmark_flag == 'remote':
        for l in lines:
            v.append([benchmark_name, 'Remote TCP', float(l)])

    f.close()

cols = ['benchmark', 'type', 'time']
bench_data = pd.DataFrame(data=v, columns=cols)

sns.set(rc={"figure.figsize": (8.5, 6.5)})
sns.set_theme(style='whitegrid')

hue_order=['Native', 'Local TCP', 'Remote TCP']
# order = ['latency', 'hotspot', 'srad_v1', 'resnet50']
order = ['latency', 'hotspot', 'srad_v1', 'resnet50', '3d-unet-kits19']

ax = sns.barplot(x='time',
                 y='benchmark',
                 hue='type',
                 hue_order=hue_order,
                 order=order,
                 orient='h',
                 data=bench_data,
                 ci=95)
ax.set_ylabel('Benchmark', rotation='horizontal')
ax.set_xlabel('Time [s]')
# ax.set_xscale('log')
ax.yaxis.set_label_coords(-0.06, 1.02)
ax.legend(loc='upper right')
ax.set_title('Traced remote execution time vs. median native performance\nNVIDIA A100, n=100, 95% confidence')
ax.figure.savefig('trace-time.pdf')