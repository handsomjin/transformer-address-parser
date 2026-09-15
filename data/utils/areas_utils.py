# -*- coding: utf-8 -*-
"""
行政区划工具类 (areas_util)
===========================
基于民政部《三级行政区地址》JSON 文件 (areas_code.json) 提供：

* 加载 / 查询行政区划树
* 按 6 位行政码 (省/市/区) 或名称检索节点
* 把 12 位 code 解析成 省-市-区 三段 (aa,bb,cc)
* 生成随机详细地址文本 (省/市/区 + 随机街道/路/门牌/小区)
* 导出全量 (省, 市, 区, 6位码, aa,bb,cc) 数据文件

编码约定
--------
JSON 中 code 为 12 位字符串，例如 ``110108000000``，
有效 6 位行政码取其前 6 位 ``110108``，按 2+2+2 切分：

    aa = 11   (省/直辖市)
    bb = 01   (地级市/直辖市区)
    cc = 08   (区/县/县级市)

用法示例
--------
# >>> from areas_util import AreasUtil
# >>> u = AreasUtil("areas_code.json")
# >>> u.get_name("110108")          # '海淀区'
# >>> u.parse_code("110108000000")  # ('11','01','08')
# >>> u.random_full_address("110108")
# >>> # '北京市海淀区知春路1号'
# >>> u.to_csv("areas_full.csv")    # 导出全量数据
"""

import csv
import json
import os
from typing import List, Dict, Optional, Tuple
import random
import string
from pypinyin import lazy_pinyin, Style
import Levenshtein
import pandas as pd
from fuzzychinese import FuzzyChineseMatch

# ----------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------
VALID_LEVELS = {1, 2, 3}  # 省 / 市 / 区(县)
# 常用汉字表（3500字，来源：《通用规范汉字表》一级字表）

# 常用字池（仅作为“候选全集”，不是错别字映射）
COMMON = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞"
char_list = list(set(COMMON))  # set 去重


# 极简部首/笔画（生产可换成完整2500字数据；这里示例用规则近似）
def _stroke_diff(a, b):
    # 没有完整笔画表时用 codepoint 差近似不合适；建议外接数据
    return abs(len(a) - len(b))


def _pinyin_char(ch):
    try:
        return "".join(lazy_pinyin(ch, style=Style.NORMAL))
    except Exception:
        return ""


def _initial(ch):
    try:
        return lazy_pinyin(ch, style=Style.FIRST_LETTER)[0]
    except Exception:
        return ""


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def split_six_code(six_code: str) -> Tuple[str, str, str]:
    """
    将 6 位行政码按 2+2+2 拆分为 (省, 市, 区)。

    例: ``110108`` -> ``('11','01','08')``
    """
    s = str(six_code).strip()
    if len(s) < 6:
        raise ValueError("行政码不足 6 位: %s" % six_code)
    return s[0:2], s[2:4], s[4:6]


def to_six_code(code) -> str:
    """兼容 6 位 / 12 位 code，统一返回前 6 位。"""
    s = str(code).strip()
    if len(s) >= 12:
        return s[:6]
    if len(s) == 6:
        return s
    raise ValueError("无法解析的行政码: %s" % code)


# ----------------------------------------------------------------------
# 主工具类
# ----------------------------------------------------------------------
class AreasUtil:
    """行政区划工具类。"""

    #: 直辖市 code 前两位（直辖市省级节点 level=1，但其子节点直接是区）
    MUNICIPALITY_PREFIXES = {"11", "12", "31", "50"}

    def __init__(self, json_path: str):
        """
        Parameters
        ----------
        json_path : str
            指向 areas_code.json 的文件路径。
        """
        if not os.path.isfile(json_path):
            raise FileNotFoundError("找不到行政区划文件: %s" % json_path)

        self.json_path = json_path
        with open(json_path, "r", encoding="utf-8") as f:
            self.root = json.load(f)

        # 省/直辖市 (level=1) 列表
        self.provinces: List[Dict] = self.root.get("children", [])

        # code(6位) -> 节点 dict 的索引，便于快速查询
        self._index: Dict[str, Dict] = {}
        # 区县级节点 (level=3) 列表
        self.districts: List[Dict] = []
        self.parent_child_map = {}
        self.code_name_map = {}
        self._build_index(self.root, _is_root=True)

        # 随机组件（可外部修改以定制地址风格）
        self.road_names = list(_DEFAULT_ROADS)
        self.community_suffixes = list(_DEFAULT_COMMUNITIES)

    # ------------------------------------------------------------------
    # 构建索引
    # ------------------------------------------------------------------
    def _build_index(self, node: Dict, _is_root: bool = False):
        if not _is_root:  # 根节点 code="00"，不索引
            code = node.get("code", "")
            try:
                six = to_six_code(code)
            except ValueError:
                six = None
            if six and six not in self._index:
                self._index[six] = node
            if node.get("level") == 3:
                self.districts.append(node)
            parent_name = node["name"]
            self.code_name_map[node.get("code", "")] = node.get("name", "")
        else:
            parent_name = "00"

        for child in node.get("children", []):
            self._build_index(child)
            self.parent_child_map.setdefault(parent_name, set()).add(child["code"])

    # ------------------------------------------------------------------
    # 基本查询
    # ------------------------------------------------------------------
    def get_node(self, six_code: str) -> Optional[Dict]:
        """按 6 位行政码获取节点 dict。"""
        return self._index.get(to_six_code(six_code))

    def get_name(self, six_code: str) -> Optional[str]:
        """按 6 位行政码获取名称。"""
        node = self.get_node(six_code)
        return node["name"] if node else None

    def get_level(self, six_code: str) -> Optional[int]:
        """按 6 位行政码获取层级。"""
        node = self.get_node(six_code)
        return node["level"] if node else None

    def is_municipality(self, province_six: str) -> bool:
        """判断某省级 6 位码是否为直辖市。"""
        return to_six_code(province_six)[:2] in self.MUNICIPALITY_PREFIXES

    # ------------------------------------------------------------------
    # 解析 code -> (省,市,区) 名称 / 编码
    # ------------------------------------------------------------------
    def parse_code(self, code) -> Tuple[str, str, str]:
        """
        把 6 位或 12 位 code 解析为 (aa, bb, cc)。

        例: ``110108000000`` -> ``('11','01','08')``
        """
        return split_six_code(to_six_code(code))

    def resolve(self, district_six: str) -> Dict:
        """
        由区县级 6 位码向上回溯，返回 ``{province, city, district}`` 三节点。

        直辖市特殊处理：city 节点与 province 同为直辖市本身。
        """
        district = self.get_node(district_six)
        if district is None:
            raise KeyError("未知的区县 code: %s" % district_six)

        # 向上找父节点（省/市）
        parents = self._parents_of(district_six)
        # parents 从根到父依次排列
        province, city = None, None
        for p in parents:
            lvl = p.get("level")
            if lvl == 1:
                province = p
            elif lvl == 2:
                city = p

        # 直辖市：没有 level=2 的市，city 用 province 自身
        if province and province["code"][:2] in self.MUNICIPALITY_PREFIXES:
            city = province

        return {
            "province": province or {},
            "city": city or {},
            "district": district,
        }

    def _parents_of(self, six_code: str) -> List[Dict]:
        """从根节点到目标节点的父链（不含目标自身）。"""
        target = to_six_code(six_code)
        stack = []

        def dfs(node):
            # 根节点 code="00" 不参与比较
            code = node.get("code", "")
            try:
                six = to_six_code(code)
            except ValueError:
                six = None
            if six == target:
                return True
            for c in node.get("children", []):
                stack.append(c)
                if dfs(c):
                    return True
                stack.pop()
            return False

        dfs(self.root)
        # stack 在找到时，末尾元素即「目标自身」，需剔除；
        # 其余元素组成 根 -> 父 的链。
        result = []
        for n in stack:
            six = to_six_code(n.get("code", ""))
            if six != target:
                result.append(n)
        return result

    # ------------------------------------------------------------------
    # 名称 -> code
    # ------------------------------------------------------------------
    def search(self, name: str, level: Optional[int] = None) -> List[Dict]:
        """按名称模糊搜索节点，可限定层级。"""
        results = []
        for six, node in self._index.items():
            if name in node.get("name", ""):
                if level is None or node.get("level") == level:
                    results.append(node)
        return results

    # ------------------------------------------------------------------
    # 地址生成
    # ------------------------------------------------------------------
    def random_detail(self) -> str:
        """随机生成一段详细地址（街道/路 + 门牌/小区，不含省市区）。"""
        road = random.choice(self.road_names)
        number = random.randint(1, 200)
        if random.random() < 0.5:
            return "%s%d号" % (road, number)
        community = random.choice(self.community_suffixes)
        unit = random.choice(string.ascii_uppercase + string.digits)
        floor = random.randint(1, 30)
        return "%s%s%s%d单元%d室" % (road, number, community, number, floor)

    def random_full_address(self, district_six: Optional[str] = None) -> str:
        """
        生成一条完整地址文本：``省 + 市 + 区 + 随机详细地址``。

        若指定 ``district_six`` 则锁定该区县，否则随机选取一个区县。
        """
        if district_six is not None:
            info = self.resolve(district_six)
        else:
            info = self.resolve(random.choice(self.districts)["code"])

        parts = [info["province"].get("name", ""),
                 info["city"].get("name", ""),
                 info["district"].get("name", "")]
        return "".join(parts) + self.random_detail()

    def format_record(self, district_six: str) -> Dict:
        """
        生成单条 CSV 记录 dict，对应需求中的一行：

            address, aa, bb, cc
        """
        info = self.resolve(district_six)
        aa, bb, cc = self.parse_code(district_six)
        province = info["province"].get("name", "")
        city = info["city"].get("name", "")
        district = info["district"].get("name", "")
        # 直辖市：province 与 city 同名，避免重复拼接，如 "北京市北京市"
        if province == city:
            addr = province + district + self.random_detail()
        else:
            addr = province + city + district + self.random_detail()
        return {
            "address": addr,
            "province": province,
            "city": city,
            "district": district,
            "code6": to_six_code(district_six),
            "aa": aa, "bb": bb, "cc": cc,
        }

    def _sound_candidates(self, ch, pool, max_py_dist=1):
        py = _pinyin_char(ch)
        if not py:
            return []
        out = []
        for c in pool:
            if c == ch:
                continue
            cpy = _pinyin_char(c)
            if not cpy:
                continue
            d = Levenshtein.distance(py, cpy)  # 拼音编辑距离[11](@ref)
            if d == 0:
                out.append((c, 0.0))  # 同音最优
            elif d <= max_py_dist:
                # 首字母相同额外加权（z/zh 这类不算首字母同但拼音距离1）
                score = d + (0 if _initial(ch) == _initial(c) else 0.5)
                out.append((c, score))
        out.sort(key=lambda x: x[1])
        return [c for c, _ in out[:8]]

    def near_shape_fuzzy(self, ch, topn=5):
        fcm = FuzzyChineseMatch(ngram_range=(1, 1), analyzer='stroke')
        fcm.fit(pd.Series(char_list))
        res = fcm.transform(pd.Series([ch]), n=topn)
        return res[:, 1].tolist()

    # ------------------------------------------------------------------
    # 错别字替换
    # ------------------------------------------------------------------
    def format_record_near_typo(self, district_six, max_typos=3,
                                sound_ratio=0.6, pool=None):
        """相近字替换：音近+形近，标签不变。"""
        base = self.format_record(district_six)
        addr = base["address"]
        pool = pool or COMMON
        chars = list(addr)

        for _ in range(max_typos):
            han_idx = [i for i, c in enumerate(chars) if '一' <= c <= '鿿']
            if not han_idx:
                break
            i = random.choice(han_idx)
            ch = chars[i]
            if random.random() < sound_ratio:
                cands = self._sound_candidates(ch, pool, max_py_dist=1)
            else:
                cands = self.near_shape_fuzzy(ch)
            if cands:
                chars[i] = random.choice(cands)

        return {
            "address": "".join(chars),
            "province": base["province"],
            "city": base["city"],
            "district": base["district"],
            "aa": base["aa"], "bb": base["bb"], "cc": base["cc"], "code6": base["code6"],
        }

    # ------------------------------------------------------------------
    # 缺少省份的地址
    # ------------------------------------------------------------------
    def format_record_no_province(self, district_six: str) -> Dict:
        """
        生成「文本地址中**不含省份**」的记录，但省市区标签 (aa/bb/cc) 保持不变。

        与 :meth:`format_record` 的唯一区别：
        地址文本从「市」开始 (市 + 区 + 随机详细地址)，不再包含省名；
        而 ``province / city / district / aa / bb / cc`` 字段与
        :meth:`format_record` **完全一致**，仍可用于按行政区编码训练/对齐。

        示例（海淀区 110108）::

             address  = "海淀区知春路96号"          # 注意：不含"北京市"
             province = "北京市"                     # 标签保留
             aa,bb,cc = 11,01,08                    # 编码不变
        """
        info = self.resolve(district_six)
        aa, bb, cc = self.parse_code(district_six)
        province = info["province"].get("name", "")
        city = info["city"].get("name", "")
        district = info["district"].get("name", "")

        # 地址文本：直辖市/省均从「市或区」起，刻意省略省名
        # 直辖市：province 与 city 同名，直接用区名起步，避免"北京市"
        if province == city:
            addr = district + self.random_detail()
        else:
            # 普通省：省名省略，保留 市 + 区
            addr = city + district + self.random_detail()

        return {
            "address": addr,
            "province": province,
            "city": city,
            "district": district,
            "code6": to_six_code(district_six),
            "aa": aa, "bb": bb, "cc": cc,
        }

    # ------------------------------------------------------------------
    # 缺少市的地址
    # ------------------------------------------------------------------
    def format_record_no_city(self, district_six: str) -> Dict:
        """
        生成「文本地址中**不含市名**」的记录，但省市区标签 (aa/bb/cc) 保持不变。

        与 :meth:`format_record` 的区别：
        地址文本为「省 + 区 + 随机详细地址」，跳过市级名称；
        ``province / city / district / aa / bb / cc`` 字段 **完全一致**。

        处理逻辑：
        - 普通省（陕西省 + 西安市 + 雁塔区）::

               address  = "陕西省雁塔区长安路96号"     # 不含"西安市"
               city     = "西安市"                    # 标签保留
               aa,bb,cc = 61,01,13                   # 编码不变

        - 直辖市（北京市 == 市级，海淀区）::

               address  = "海淀区知春路96号"           # 无省也无市
               city     = "北京市"                    # 标签保留

          直辖市没有独立的「地级市」层级，province 与 city 同名，
          去掉市名后等同于只剩区名，此行为符合直觉，属预期结果。

        该方法与 :meth:`format_record_no_province` 可组合使用，
        若同时需要「省+市都不出现」，可再自行裁剪 ``address``。
        """
        info = self.resolve(district_six)
        aa, bb, cc = self.parse_code(district_six)
        province = info["province"].get("name", "")
        city = info["city"].get("name", "")
        district = info["district"].get("name", "")

        # 直辖市：市名 == 省名，去市即只剩区
        if province == city:
            addr = district + self.random_detail()
        else:
            # 普通省：省 + 区，刻意跳过市名
            addr = province + district + self.random_detail()

        return {
            "address": addr,
            "province": province,
            "city": city,
            "district": district,
            "code6": to_six_code(district_six),
            "aa": aa, "bb": bb, "cc": cc,
        }

    # ------------------------------------------------------------------
    # 省市区乱序的地址
    # ------------------------------------------------------------------
    def format_record_shuffled(self, district_six: str) -> dict:
        """
        生成省市区名称随机排序的地址记录。
        - 省、市、区三级名称随机排列（仅文本顺序改变）
        - 详细地址始终跟随区名（district）之后
        - 标签字段（province/city/district/aa/bb/cc/code6）保持不变
        - 直辖市自动去重（省==市时只保留一个）
        """
        rec = self.resolve(district_six)
        if not rec:
            raise ValueError(f"无效的区县编码: {district_six}")

        info = self.resolve(district_six)
        aa, bb, cc = self.parse_code(district_six)
        province_name = info["province"].get("name", "")
        city_name = info["city"].get("name", "")
        district_name = info["district"].get("name", "")
        detail = self.random_detail()
        # 1. 打乱省市区顺序
        parts = [province_name, city_name, district_name]
        random.shuffle(parts)

        # 2. 去重（保留首次出现顺序），解决直辖市省==市重复问题
        unique_parts = list(dict.fromkeys(parts))

        # 3. 找到区名的位置，在其后插入详细地址
        dist_idx = unique_parts.index(district_name)
        unique_parts.insert(dist_idx + 1, detail)

        # 4. 拼接最终地址
        address = "".join(unique_parts)

        return {
            "address": address,
            "province": province_name,
            "city": city_name,
            "district": district_name,
            "aa": aa,
            "bb": bb,
            "cc": cc,
            "code6": to_six_code(district_six)
        }

    # ------------------------------------------------------------------
    # 省市区后缀去除
    # ------------------------------------------------------------------
    def format_record_stripped(self, district_six: str) -> dict:
        """
        生成去除行政级别后缀（省/市/区/县）的地址记录。
        - 若名称长度 ≤ 2，则保留后缀（如“北京”不去“市”）
        - 否则去掉末尾的“省”“市”“区”“县”
        - 直辖市自动去重（省==市时只保留一个）
        - 详细地址始终跟随区名之后
        - 标签字段（province/city/district/aa/bb/cc/code6）保持原始值
        """
        rec = self.resolve(district_six)
        if not rec:
            raise ValueError(f"无效的区县编码: {district_six}")

        info = self.resolve(district_six)
        aa, bb, cc = self.parse_code(district_six)
        province_name = info["province"].get("name", "")
        city_name = info["city"].get("name", "")
        district_name = info["district"].get("name", "")
        detail = self.random_detail()

        def strip_suffix(name: str) -> str:
            if len(name) <= 2:
                return name
            for suffix in ("省", "市", "区", "县"):
                if name.endswith(suffix):
                    return name[:-len(suffix)]
            return name

        province_stripped = strip_suffix(province_name)
        city_stripped = strip_suffix(city_name)
        district_stripped = strip_suffix(district_name)

        # 构建有序列表并去重（解决直辖市省==市问题）
        parts = [province_stripped, city_stripped, district_stripped]
        unique_parts = list(dict.fromkeys(parts))

        # 找到区名的位置，在其后插入详细地址
        dist_idx = unique_parts.index(district_stripped)
        unique_parts.insert(dist_idx + 1, detail)

        address = "".join(unique_parts)

        return {
            "address": address,
            "province": province_name,
            "city": city_name,
            "district": district_name,
            "aa": aa,
            "bb": bb,
            "cc": cc,
            "code6": to_six_code(district_six)
        }

    # ------------------------------------------------------------------
    # 缺少区/县的地址
    # ------------------------------------------------------------------
    def format_record_no_district(self, district_six: str) -> Dict:
        """
        生成「文本地址中**不含区/县**」的记录。

        与前两个 ``no_*`` 方法的**关键区别**在于：
        区/县是三级行政码 (cc) 的直接载体，一旦文本中缺失区县名，
        **三级编码 (cc) 也就无法对应到具体区县**，因此按需求约定：

        - ``district`` 标签 **置空** (``""``)
        - 三级编码 **cc 置空** (``""``)；code6 末两位同样无法确认，故 ``code6`` 也置空
        - ``aa / bb`` (**省、市**两级编码) **保持不变**

        即「缺失哪一级的信息，就清空哪一级的标签与编码」，上级信息仍可用于对齐。

        地址文本为「省 + 市 + 随机详细地址」，不再出现区/县名。
        直辖市因 province == city，文本退化为「市 + 详细地址」。

        示例（雁塔区 610113，三级码 61-01-13）::

             address  = "陕西省西安市和平路96号"          # 不含"雁塔区"
             district = ""                               # 区县标签清空
             aa,bb,cc = 61,01,""                        # 仅 cc 置空
             code6    = ""                               # 末两位无法确认

        与 :meth:`format_record_no_city` 的区别::

             no_city    : 标签全保留，只改文本        -> 适合「文本有噪、编码干净」
             no_district: 区县标签 + cc 编码一并清空   -> 适合「确实缺区县」的弱标注
        """
        info = self.resolve(district_six)
        aa, bb, _cc = self.parse_code(district_six)  # cc 将被置空，不再使用
        province = info["province"].get("name", "")
        city = info["city"].get("name", "")

        # 地址文本：省 + 市（+ 详细地址），刻意省略区/县
        if province == city:
            # 直辖市：省=市，去区后只剩「市 + 详细地址」
            addr = city + self.random_detail()
        else:
            addr = province + city + self.random_detail()

        return {
            "address": addr,
            "province": province,
            "city": city,
            "district": "",  # 缺失区县：标签置空
            "code6": "",  # 末两位无法确认 -> 置空
            "aa": aa, "bb": bb,  # 省、市两级保持不变
            "cc": "",  # 三级编码置空
        }

    def format_record_wrong_province(self, district_six: str, rng: random.Random = None) -> dict:
        """
        将地址中的省替换为其他任意省（不含自身）。
        文本仅保留“错误省 + 详细地址”；标签中 province/aa 更新为新省，city/district/bb/cc/code6 置空。
        """
        if rng is None:
            rng = random.Random()
        rec = self.resolve(district_six)
        if not rec:
            raise ValueError(f"无效的区县编码: {district_six}")

        province = rec["province"].get("name", "")

        candidates = [c for c in self.parent_child_map['00'] if c != rec["province"].get("code", "")]
        new_prov_code = rng.choice(candidates)
        new_prov_name = self.code_name_map[new_prov_code]
        new_aa, _, _ = self.parse_code(new_prov_code)

        city = rec["city"].get("name", "")
        district = rec["district"].get("name", "")
        # 直辖市：province 与 city 同名，避免重复拼接，如 "北京市北京市"
        if province == city:
            addr = new_prov_name + district + self.random_detail()
        else:
            addr = new_prov_name + city + district + self.random_detail()
        return {
            "address": addr,
            "province": new_prov_name,
            "city": city,
            "district": district,
            "code6": to_six_code(district_six),
            "aa": new_aa, "bb": "", "cc": ""
        }

    def format_record_wrong_city(self, district_six: str, rng: random.Random = None) -> dict:
        """
        将地址中的市替换为同一省内其他市（不含自身）。
        文本保留“原省 + 错误市 + 详细地址”；标签中 city/bb 更新，district/cc/code6 置空。
        若省==市（直辖市）或省内仅有一个市，则抛出 RuntimeError。
        """
        if rng is None:
            rng = random.Random()
        rec = self.resolve(district_six)
        if not rec:
            raise ValueError(f"无效的区县编码: {district_six}")

        province = rec["province"].get("name", "")
        city = rec["city"].get("name", "")

        if province == city:
            return

        candidates = [c for c in self.parent_child_map[province] if c != rec["city"].get("code", "")]
        new_city_code = rng.choice(candidates)
        new_city_name = self.code_name_map[new_city_code]
        aa, bb, cc = self.parse_code(district_six)

        district = rec["district"].get("name", "")
        # 直辖市：province 与 city 同名，避免重复拼接，如 "北京市北京市"
        if province == city:
            addr = province + district + self.random_detail()
        else:
            addr = province + new_city_name + district + self.random_detail()
        return {
            "address": addr,
            "province": province,
            "city": new_city_name,
            "district": district,
            "code6": to_six_code(district_six),
            "aa": aa, "bb": bb, "cc": ""
        }

    def format_record_with_junk(self, standard: dict, min_insertions: int = 1, max_insertions: int = 3) -> dict:
        """
        在标准地址中随机插入多余空格或特殊符号。
        - 符号池：空格、制表符、逗号、句号、分号、冒号、横线、下划线、斜杠
        - 插入位置：地址文本中任意两个字符之间（包括开头和结尾）
        - 插入次数：min_insertions ~ max_insertions 之间的随机整数
        - 标签字段（province/city/district/aa/bb/cc/code6）保持不变
        """
        # 先生成标准地址（含直辖市去重）
        address = standard["address"]

        junk_pool = [' ', '\t', '，', '.', ';', ':', '-', '_', '/']
        chars = list(address)

        insertion_count = random.randint(min_insertions, max_insertions)
        for _ in range(insertion_count):
            if not chars:
                break
            # 插入位置：0 到 len(chars) 之间（包括末尾）
            pos = random.randint(0, len(chars))
            junk = random.choice(junk_pool)
            chars.insert(pos, junk)

        new_address = ''.join(chars)

        return {
            "address": new_address,
            "province": standard["province"],
            "city": standard["city"],
            "district": standard["district"],
            "aa": standard["aa"],
            "bb": standard["bb"],
            "cc": standard["cc"],
            "code6": standard["code6"],
        }

    # ------------------------------------------------------------------
    # 批量生成
    # ------------------------------------------------------------------
    def generate_all(self, per_district: int = 1,
                     mode: str = "full") -> List[Dict]:
        """
        为**每一个**区县生成地址记录。

        Parameters
        ----------
        per_district : int
            每个区县生成的条数（默认 1）。
        mode : str
            地址文本的生成模式，决定文本中省略哪一级：

            - ``"full"``       : 完整地址，省 + 市 + 区（默认）
            - ``"no_province"``: 文本不含省份（见 :meth:`format_record_no_province`）
            - ``"no_city"``    : 文本不含市名（见 :meth:`format_record_no_city`）
            - ``"no_district"``: 文本不含区/县；此时区县标签与三级编码
              (district / cc / code6) 一并置空，仅 aa/bb 保留
              （见 :meth:`format_record_no_district`）

            前三种模式标签 (aa/bb/cc) 完全不变；
            ``"no_district"`` 会清空缺失层及其对应的编码。
        """
        fmt_map = {
            "full": self.format_record,
            "no_province": self.format_record_no_province,
            "no_city": self.format_record_no_city,
            "no_district": self.format_record_no_district,
            "disorder": self.format_record_shuffled,
            "strip": self.format_record_stripped,
            "near_type": self.format_record_near_typo,
            "wrong_prov": self.format_record_wrong_province,
            "wrong_city": self.format_record_wrong_city
        }

        records = []
        for fmt in fmt_map.values():
            for d in self.districts:
                six = to_six_code(d["code"])
                n = max(1, int(per_district))
                idx = random.randint(0, 10)
                for i in range(n):
                    data = fmt(six)
                    if data:
                        records.append(data)
                        if 3 == idx:
                            r = self.format_record_with_junk(data)
                            records.append(r)

        return records

    def to_csv(self, output_path: str, per_district: int = 1,
               encoding: str = "utf-8-sig") -> int:
        """
        导出全量数据到 CSV。

        Parameters
        ----------
        output_path : str
            输出文件路径。
        per_district : int
            每个区县生成的条数。
        mode : str
            地址生成模式，见 :meth:`generate_all`：

            - ``"full"`` 完整 / ``"no_province"`` 缺省 / ``"no_city"`` 缺市

            无论哪种，标签列 (province/city/district/aa/bb/cc) 均完整保留。
        encoding : str
            CSV 编码，默认 ``utf-8-sig`` (带 BOM，Excel 友好)。

        Columns
        -------
        address, province, city, district, code6, aa, bb, cc

        Returns
        -------
        int
            写入的记录条数。
        """
        records = self.generate_all(per_district=per_district)
        fields = ["address", "province", "city", "district",
                  "code6", "aa", "bb", "cc"]
        with open(output_path, "w", encoding=encoding, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        return len(records)

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def stats(self) -> Dict:
        """返回省市区三级数量统计。"""
        provinces = [p for p in self.provinces]
        cities = sum(len(p.get("children", [])) for p in provinces)
        return {
            "provinces": len(provinces),
            "cities": cities,
            "districts": len(self.districts),
        }

    def __len__(self) -> int:
        return len(self.districts)

    def __repr__(self) -> str:
        s = self.stats()
        return ("AreasUtil(provinces=%d, cities=%d, districts=%d, "
                "json=%r)") % (s["provinces"], s["cities"], s["districts"],
                               self.json_path)


# ----------------------------------------------------------------------
# 默认随机地址词库（可自由扩充）
# ----------------------------------------------------------------------
_DEFAULT_ROADS = (
    "知春路", "中关村大街", "长安街", "建国路", "朝阳路", "学院路",
    "人民路", "解放路", "建设路", "和平路", "新华路", "中山路",
    "青年路", "文化路", "科技路", "光华路", "花园路", "湖滨路",
    "南京路", "淮海路", "四川路", "北京路", "广州路", "西湖路",
    "东风路", "迎宾路", "环城路", "幸福路", "光明街", "胜利街",
    "幸福大街", "友谊路", "长江路", "黄河路", "泰山路", "华山路",
)

_DEFAULT_COMMUNITIES = (
    "小区", "家园", "公寓", "新村", "苑", "花园", "名城",
    "雅苑", "嘉园", "豪庭", "庭院", "名邸", "华庭", "怡景",
)
