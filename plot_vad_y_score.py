from matplotlib import pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import os
import numpy as np
import matplotlib.ticker as mt
from pyecharts.charts import ThemeRiver
import pyecharts.options as opts
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats.mstats import zscore
from plot_func import many_densities_plot

time_dimen_v = ['v-'+str(y) for y in range(2010,2022)]
time_dimen_a = ['a-'+str(y) for y in range(2010,2022)]
time_dimen_d = ['d-'+str(y) for y in range(2010,2022)]
fig, glossary = many_densities_plot(time_dimen_d, percentilize=True, single_column=True)
plt.show()
print('done')
