import random
import pandas as pd
from typing import List, Dict, Optional

# 省份数据（与之前相同）
FULL_PROVINCES = [
    '北京市', '天津市', '上海市', '重庆市',
    '河北省', '山西省', '辽宁省', '吉林省', '黑龙江省',
    '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省',
    '河南省', '湖北省', '湖南省', '广东省', '海南省',
    '四川省', '贵州省', '云南省', '陕西省', '甘肃省', '青海省',
    '台湾省',
    '内蒙古自治区', '广西壮族自治区', '西藏自治区', '宁夏回族自治区', '新疆维吾尔自治区',
    '香港特别行政区', '澳门特别行政区'
]

SHORT_PROVINCES = []
for p in FULL_PROVINCES:
    short = p.replace('省', '').replace('市', '').replace('自治区', '').replace('特别行政区', '').replace('壮族',
                                                                                                          '').replace(
        '回族', '').replace('维吾尔', '')
    SHORT_PROVINCES.append(short)

SHORT_TO_FULL = dict(zip(SHORT_PROVINCES, FULL_PROVINCES))


def get_label(label_map, val):
    if pd.isna(val):
        return '<unk>'
    s = str(val).strip()
    return label_map.get(s, '<unk>')


def levenshtein_distance(s1: str, s2: str) -> int:
    """计算编辑距离"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)  # 注意此处有笔误，应修正为 dp[i-1][j], dp[i][j-1], dp[i-1][j-1]
    return dp[m][n]


def get_similar_province(original_full: str, top_k: int = 3, max_dist: int = 2) -> str:
    """获取与原始省份编辑距离相近的省份"""
    orig_short = original_full
    for suffix in ['省', '市', '自治区', '特别行政区', '壮族', '回族', '维吾尔']:
        orig_short = orig_short.replace(suffix, '')

    distances = []
    for idx, short in enumerate(SHORT_PROVINCES):
        if short == orig_short:
            continue
        dist = levenshtein_distance(orig_short, short)
        distances.append((dist, short))

    if not distances:
        return random.choice(FULL_PROVINCES)

    distances.sort(key=lambda x: x[0])
    candidates = [short for d, short in distances if d <= max_dist]
    if candidates:
        chosen_short = random.choice(candidates)
    else:
        top_candidates = [short for _, short in distances[:top_k]]
        chosen_short = random.choice(top_candidates)

    return SHORT_TO_FULL[chosen_short]


def generate_negative_samples(
        label_map,
        data: List[Dict[str, str]],
        seed: Optional[int] = None,
        rule2_prob: float = 0.5,
) -> List[Dict[str, str]]:
    """生成三种 negative 样本"""
    if seed is not None:
        random.seed(seed)

    results = []
    for row in data:
        text = row["text"]
        labels = row["labels"]
        prov = str(row.get('省', '')) if row.get('省') is not None else ''
        city = str(row.get('市', '')) if row.get('市') is not None else ''
        dist = str(row.get('区', '')) if row.get('区') is not None else ''

        # 规则1
        text_rule1 = text.replace('省', '').replace('市', '').replace('区', '')
        results.append({'text': text_rule1, 'province': prov, 'city': city, 'district': dist, 'labels': labels})

        # 规则2
        if city:
            if random.random() < rule2_prob:
                city_full = city
                city_short = city.rstrip('市')
                new_text = text
                if city_full and city_full in new_text:
                    new_text = new_text.replace(city_full, '', 1)
                elif city_short and city_short in new_text:
                    new_text = new_text.replace(city_short, '', 1)
                new_city = ''
            else:
                new_text = text
                new_city = city
            results.append({'text': new_text, 'province': prov, 'city': new_city, 'district': dist, 'labels': labels})

        # 规则3
        if prov:
            new_prov = get_similar_province(prov)
            prov_full = prov
            prov_short = prov.rstrip('省').rstrip('市').rstrip('自治区').rstrip('特别行政区')
            temp_text = text
            if prov_full and prov_full in temp_text:
                temp_text = temp_text.replace(prov_full, '', 1)
            elif prov_short and prov_short in temp_text:
                temp_text = temp_text.replace(prov_short, '', 1)
            new_text = new_prov + temp_text
            labels = f"{get_label(label_map, new_prov)},<unk>,<unk>"
            results.append({'text': new_text, 'province': new_prov, 'city': '', 'district': '', 'labels': labels})

    return results
