# -*- coding: utf-8 -*-
"""
generate_areas.py
=================
入口脚本：基于 areas_code.json 生成全量地址 CSV 数据文件。

输出列：address, province, city, district, code6, aa, bb, cc
其中 address = 省+市+区 + 随机详细地址；aa/bb/cc 为 6 位行政码 2+2+2 拆分。

支持三种模式（--mode）：
  full        完整地址（默认）
  no_province 文本不含省份（标签保留）
  no_city     文本不含市名（标签保留）
  no_district 文本不含区县（区县标签+三级编码cc置空，aa/bb保留）

用法
----
    python3 generate_areas.py [--json PATH] [--output PATH]
                              [--per-district N] [--mode MODE] [--seed SEED]

示例（需求中的样子）：
    北京市海淀区知春路知春里小区1号   ->  address 列
     对应 aa=11, bb=01, cc=08
"""
import argparse
import os
import sys

# 允许直接运行：把当前目录加入 path 以引用 areas_util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from areas_utils import AreasUtil  # noqa: E402


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_json = os.path.join(here, "../areas_code.json")
    default_out = os.path.join(here, "../areas_full.csv")

    util = AreasUtil(default_json)
    print("加载完成:", util)
    print("统计:", util.stats())

    n = util.to_csv(default_out, per_district=1)
    print("已写入 %d 条记录 (mode=%s) -> %s" % (n, 'full', default_out))

    # 预览前 5 行
    import csv as _csv
    with open(default_out, "r", encoding="utf-8-sig") as f:
        rows = list(_csv.DictReader(f))[:5]
    print("\n预览:")
    for r in rows:
        print("  %(address)s  |  %(aa)s,%(bb)s,%(cc)s  (%(code6)s)" % r)


if __name__ == "__main__":
    main()
