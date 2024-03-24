from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.render import make_snapshot

from snapshot_selenium import snapshot

import random
from pyecharts import options as opts
from pyecharts.charts import HeatMap
from pyecharts.faker import Faker

value = [[i, j, random.randint(0, 50)] for i in range(24) for j in range(7)]
c = (
    HeatMap()
    .add_xaxis(Faker.clock)
    .add_yaxis(
        "series0",
        Faker.week,
        value,
        label_opts=opts.LabelOpts(is_show=True, position="inside"),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="HeatMap-Label 显示"),
        # visualmap_opts=opts.VisualMapOpts(),
    )
    .render("heatmap_with_label_show.html")
)
#--------------------------------------------
bar = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.PURPLE_PASSION))
    .add_xaxis(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    .add_yaxis("降雨量", [2.0, 4.9, 7.0, 23.2, 25.6, 76.7, 135.6, 162.2, 32.6, 20.0, 6.4, 3.3])
    .add_yaxis("蒸发量", [2.6, 5.9, 9.0, 26.4, 28.7, 70.7, 175.6, 182.2, 48.7, 18.8, 6.0, 2.3])
    .set_global_opts(title_opts=opts.TitleOpts(title="柱状图", subtitle="一年的降雨量与蒸发量"))
    # 或者直接使用字典参数
    # .set_global_opts(title_opts={"text": "主标题", "subtext": "副标题"})
)
bar.render('one_tset.html')

make_snapshot(snapshot, bar.render(), 'bar.png')