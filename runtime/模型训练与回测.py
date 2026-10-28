# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import logging
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get('TEDDY_DATA_DIR') or os.path.join(BASE_DIR, '数据集_清洗')
))
RESULT_DIR = os.path.join(BASE_DIR, '训练结果')
LOG_DIR = os.path.join(RESULT_DIR, '训练日志')

for d in [RESULT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

log_file = os.path.join(LOG_DIR, f'训练日志_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def safe_parse_date(series, col_name='unknown'):
    import pandas as pd
    if series.dtype in ['int64', 'float64', 'int32', 'float32']:
        return pd.to_datetime(series.astype(int).astype(str), format='%Y%m%d', errors='coerce')
    else:
        result = pd.to_datetime(series, format='%Y%m%d', errors='coerce')
        if result.isna().sum() > len(series) * 0.5:
            result = pd.to_datetime(series, errors='coerce')
        return result


def run_stage2():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from tqdm import tqdm

    logger.info("阶段2：事件识别与分类 开始")

    stage_dir = os.path.join(RESULT_DIR, '5.1_事件识别与分类')
    os.makedirs(stage_dir, exist_ok=True)

    # 加载事件数据
    event_path = os.path.join(DATA_DIR, '事件数据', '事件数据_清洗.csv')
    if not os.path.exists(event_path):
        for alt in ['事件数据/事件数据.csv', '事件数据_清洗.csv']:
            if os.path.exists(os.path.join(DATA_DIR, alt)):
                event_path = os.path.join(DATA_DIR, alt)
                break
    event_df = pd.read_csv(event_path)
    logger.info(f"  加载事件数据: {len(event_df)} 条")

    title_col = next((c for c in ['标题_清洗后', '标题', 'title', 'event_title'] if c in event_df.columns), None)
    summary_col = next((c for c in ['摘要_清洗后', '摘要', 'summary'] if c in event_df.columns), None)
    logger.info(f"  标题列: {title_col}, 摘要列: {summary_col}")

    # AHP层次分类
    logger.info("  模型一：AHP层次分类")

    AHP_RULES = {
        '宏观': {
            'keywords': ['央行', '利率', '降息', '加息', '货币', '通胀', 'CPI', 'GDP', 'PMI',
                          '财政', '税收', '关税', '汇率', '人民币', '美元', '外汇', '贸易战',
                          '经济', '宏观', '政策', '国务院', '发改委', '证监会', '银保监'],
            'subs': {
                '政策发布': ['政策', '法规', '条例', '意见', '通知', '规划', '方案', '改革', '监管', '审批'],
                '技术突破': ['技术突破', '创新', '研发', '专利', '首发', '首款', '突破', '颠覆', '芯片', 'AI', '人工智能'],
            }
        },
        '行业': {
            'keywords': ['行业', '产业', '板块', '领域', '市场', '需求', '供给', '产能', '价格',
                          '原油', '煤炭', '钢铁', '有色', '半导体', '新能源', '汽车', '医药'],
            'subs': {
                '政策发布': ['政策', '补贴', '扶持', '准入', '标准'],
                '技术突破': ['技术突破', '创新', '研发', '专利'],
            }
        },
        '公司': {
            'keywords': ['公司', '股份', '集团', '有限', '董事', '股东', '高管', '年报', '季报',
                          '财报', '业绩', '营收', '利润', '分红', '回购', '增持', '减持',
                          '并购', '重组', '收购', '合并', '合同', '中标', '签约'],
            'subs': {
                '业绩': ['业绩', '营收', '利润', '净利', '预增', '预减', '略增', '略减', '扭亏', '首亏', '续亏',
                          '超预期', '不及预期', '快报', '年报', '季报', '半年报', '财报'],
                '人事变动': ['辞职', '任命', '离职', '聘任', '解聘', '董事', '高管', '总经理', '董秘', '独董'],
                '股东变动': ['增持', '减持', '质押', '解质', '回购', '转让', '增发', '配股', '定增', '股权激励',
                              '大宗交易', '举牌', '要约收购'],
                '分红送转': ['分红', '送转', '派息', '送股', '转增', '预案', '除权', '除息', '股权登记',
                              '分红送转', '利润分配'],
                '并购': ['并购', '重组', '收购', '合并', '剥离', '分立', '借壳', '资产注入', '要约收购'],
                '融资': ['融资', '定增', '增发', '配股', '可转债', '发债', '募资', '非公开', 'IPO'],
                '利空公告': ['违规', '处罚', '立案', '退市', '风险', '警示', '诉讼', '冻结', '暴雷', '违约',
                              '亏损', '停产', '事故'],
                '合同': ['合同', '中标', '签约', '订单', '框架协议', '战略合作'],
            }
        },
        '其他': {
            'keywords': [],
            'subs': {}
        }
    }

    EXCLUSIVE_SUB_MAP = {
        '分红送转': ['分红', '送转', '派息', '送股', '转增', '预案', '除权', '除息', '股权登记',
                      '分红送转', '利润分配'],
        '业绩': ['业绩', '预增', '预减', '略增', '略减', '扭亏', '首亏', '续亏', '快报'],
        '利空公告': ['违规', '处罚', '立案', '退市', '警示', '诉讼', '冻结', '暴雷', '违约'],
        '并购': ['并购', '重组', '收购', '借壳', '资产注入'],
        '融资': ['融资', '定增', '增发', '可转债', '发债', '募资'],
        '合同': ['合同', '中标', '签约', '订单', '战略合作'],
        '技术突破': ['技术突破', '创新', '研发', '专利', '首发', '首款', 'AI', '人工智能'],
        '政策发布': ['政策', '法规', '条例', '意见', '规划', '改革'],
    }

    def classify_ahp(title, summary=''):
        text = str(title) + ' ' + str(summary)

        for sub_type, keywords in EXCLUSIVE_SUB_MAP.items():
            if any(kw in text for kw in keywords):
                for level1, rules in AHP_RULES.items():
                    if sub_type in rules.get('subs', {}):
                        return level1, sub_type
                return '公司', sub_type

        for level1, rules in AHP_RULES.items():
            if level1 == '其他':
                continue
            if any(kw in text for kw in rules['keywords']):
                for sub_type, sub_kws in rules.get('subs', {}).items():
                    if any(kw in text for kw in sub_kws):
                        return level1, sub_type
                return level1, level1

        return '其他', '其他'

    ahp_results = event_df.apply(
        lambda r: classify_ahp(r.get(title_col, ''), r.get(summary_col, '')), axis=1
    )
    event_df['AHP一级分类'] = [r[0] for r in ahp_results]
    event_df['event_sub'] = [r[1] for r in ahp_results]
    event_df['event_type'] = event_df['AHP一级分类']

    logger.info("  AHP分类分布:")
    for cat, cnt in event_df['AHP一级分类'].value_counts().items():
        logger.info(f"    {cat}: {cnt}")
    logger.info("  event_sub分布:")
    for sub, cnt in event_df['event_sub'].value_counts().items():
        logger.info(f"    {sub}: {cnt}")

    # TF-IDF + LogisticRegression
    logger.info("  模型二：TF-IDF + LogisticRegression")

    C1_col = next((c for c in ['事件大类', 'event_category', 'AHP一级分类'] if c in event_df.columns), 'AHP一级分类')
    logger.info(f"  C1标签列: {C1_col}, 类别数: {event_df[C1_col].nunique()}")

    texts = event_df[title_col].fillna('').astype(str) + ' ' + event_df.get(summary_col, pd.Series([''] * len(event_df))).fillna('').astype(str)

    import jieba

    def tokenize_zh(text):
        return ' '.join(jieba.cut(str(text)))

    logger.info("  中文分词中...")
    tokenized = texts.apply(tokenize_zh)

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_tfidf = tfidf.fit_transform(tokenized)
    logger.info(f"  TF-IDF矩阵: {X_tfidf.shape}")

    le_c1 = LabelEncoder()
    y_c1 = le_c1.fit_transform(event_df[C1_col].astype(str))

    np.random.seed(42)
    indices = np.arange(len(event_df))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    X_tr, X_va = X_tfidf[train_idx], X_tfidf[val_idx]
    y_tr, y_va = y_c1[train_idx], y_c1[val_idx]

    lr_c1 = LogisticRegression(C=1.0, max_iter=500, class_weight='balanced', random_state=42, solver='lbfgs')
    lr_c1.fit(X_tr, y_tr)

    c1_pred = lr_c1.predict(X_va)
    c1_acc = accuracy_score(y_va, c1_pred)
    c1_f1 = f1_score(y_va, c1_pred, average='weighted')
    logger.info(f"  C1(TF-IDF+LR): Accuracy={c1_acc:.4f}, Weighted-F1={c1_f1:.4f}")

    # C2/C3/C4 语义规则映射
    logger.info("  C2/C3/C4: 语义规则映射")

    SHORT_TERM_SUBS = ['人事变动', '股东变动', '利空公告', '其他']
    LONG_TERM_SUBS = ['并购', '融资', '政策发布', '技术突破']

    def map_period(sub):
        if sub in SHORT_TERM_SUBS:
            return '短期'
        elif sub in LONG_TERM_SUBS:
            return '长期'
        else:
            return '中期'

    HIGH_PRED_SUBS = ['业绩', '分红送转']
    MED_PRED_SUBS = ['并购', '政策发布', '融资']

    def map_predictability(sub):
        if sub in HIGH_PRED_SUBS:
            return '高'
        elif sub in MED_PRED_SUBS:
            return '中'
        else:
            return '低'

    event_df['周期标签'] = event_df['event_sub'].apply(map_period)
    event_df['可预测性标签'] = event_df['event_sub'].apply(map_predictability)

    industry_col = next((c for c in ['行业标签', 'industry', '行业'] if c in event_df.columns), None)
    if industry_col is None:
        INDUSTRY_KW = {
            '金融': ['银行', '证券', '保险', '基金', '信托'],
            '科技': ['芯片', 'AI', '人工智能', '软件', '半导体', '电子', '通信', '互联网'],
            '制造': ['制造', '机械', '设备', '汽车', '新能源', '光伏', '风电'],
            '医药': ['医药', '生物', '医疗', '药', '器械', 'CRO'],
            '消费': ['消费', '零售', '食品', '饮料', '服装', '家电', '旅游'],
            '地产': ['地产', '房产', '建筑', '建材', '基建'],
            '能源': ['能源', '石油', '煤炭', '化工', '有色', '钢铁'],
            '其他': [],
        }

        def map_industry(title):
            for ind, kws in INDUSTRY_KW.items():
                if any(kw in str(title) for kw in kws):
                    return ind
            return '其他'

        event_df['行业标签'] = event_df[title_col].fillna('').apply(map_industry)
        industry_col = '行业标签'

    for col, name in [('周期标签', 'C2'), ('行业标签', 'C3'), ('可预测性标签', 'C4')]:
        if col in event_df.columns:
            dist = event_df[col].value_counts()
            logger.info(f"  {name}({col})分布: {dist.to_dict()}")

    c1_all_pred = lr_c1.predict(X_tfidf)
    event_df['C1_pred'] = le_c1.inverse_transform(c1_all_pred)

    logger.info("  四维分类结果汇总")
    c2_le = LabelEncoder()
    c3_le = LabelEncoder()
    c4_le = LabelEncoder()

    c2_labels = c2_le.fit_transform(event_df['周期标签'].astype(str)) if '周期标签' in event_df.columns else np.zeros(len(event_df), dtype=int)
    c3_labels = c3_le.fit_transform(event_df[industry_col].astype(str)) if industry_col else np.zeros(len(event_df), dtype=int)
    c4_labels = c4_le.fit_transform(event_df['可预测性标签'].astype(str)) if '可预测性标签' in event_df.columns else np.zeros(len(event_df), dtype=int)

    logger.info(f"  C1(主体): TF-IDF+LR Weighted-F1={c1_f1:.4f}")
    logger.info(f"  C2(周期): 规则映射 n_class={len(c2_le.classes_)} -> {list(c2_le.classes_)}")
    logger.info(f"  C3(行业): 规则映射 n_class={len(c3_le.classes_)} -> {list(c3_le.classes_)}")
    logger.info(f"  C4(可预测性): 规则映射 n_class={len(c4_le.classes_)} -> {list(c4_le.classes_)}")

    event_df.to_csv(os.path.join(stage_dir, '事件分类结果.csv'), index=False, encoding='utf-8-sig')

    import joblib
    joblib.dump(tfidf, os.path.join(stage_dir, 'tfidf_vectorizer.pkl'))
    joblib.dump(lr_c1, os.path.join(stage_dir, 'c1_lr_model.pkl'))
    joblib.dump(le_c1, os.path.join(stage_dir, 'c1_label_encoder.pkl'))

    with open(os.path.join(stage_dir, 'stage2_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'c1_f1': float(c1_f1), 'c1_acc': float(c1_acc),
            'c1_classes': list(le_c1.classes_),
            'c2_classes': list(c2_le.classes_),
            'c3_classes': list(c3_le.classes_),
            'c4_classes': list(c4_le.classes_),
            'method': 'AHP_rules + TF-IDF+LR',
        }, f, ensure_ascii=False, indent=2)

    logger.info("阶段2完成")
    return True


def run_stage3():
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import jieba
    from tqdm import tqdm

    logger.info("阶段3：关联公司挖掘 开始")
    stage_dir = os.path.join(RESULT_DIR, '5.2_关联公司挖掘')
    os.makedirs(stage_dir, exist_ok=True)

    event_df = pd.read_csv(os.path.join(DATA_DIR, '事件数据', '事件数据_清洗.csv'))
    triplet_path = os.path.join(DATA_DIR, '建模核心表', '事件公司股价三元组_增强_v3清洗后.csv')
    if not os.path.exists(triplet_path):
        triplet_path = os.path.join(DATA_DIR, '建模核心表', '事件公司股价三元组_增强.csv')
    triplet_df = pd.read_csv(triplet_path) if os.path.exists(triplet_path) else pd.DataFrame()
    logger.info(f"  三元组: {os.path.basename(triplet_path)}: {len(triplet_df)} 条" if len(triplet_df) > 0 else "  无三元组数据")

    title_col = next((c for c in ['标题_清洗后', '标题', 'title'] if c in event_df.columns), event_df.columns[3])
    abstract_col = next((c for c in ['摘要_清洗后', '摘要', 'abstract'] if c in event_df.columns), None)

    logger.info("  TF-IDF文本相似度计算")
    texts = [(' '.join(jieba.cut(str(event_df.iloc[i][title_col]) + (' ' + str(event_df.iloc[i][abstract_col]) if abstract_col else '')))) for i in range(len(event_df))]
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    logger.info(f"  TF-IDF矩阵: {tfidf_matrix.shape}")

    logger.info("  典型事件关联展示")
    type_col = next((c for c in ['事件大类', 'AHP一级分类'] if c in event_df.columns), None)
    if type_col:
        macro_events = event_df[event_df[type_col] == '宏观']
        if len(macro_events) > 0:
            sample_idx = macro_events.index[0]
            sims = cosine_similarity(tfidf_matrix[sample_idx], tfidf_matrix).flatten()
            top_indices = np.argsort(sims)[-10:][::-1]
            logger.info(f"  典型宏观事件: {macro_events.iloc[0].get(title_col, 'N/A')[:50]}...")
            for rank, idx in enumerate(top_indices):
                if idx != sample_idx:
                    logger.info(f"    {rank + 1}. {event_df.iloc[idx].get(title_col, 'N/A')[:40]} (相似度={sims[idx]:.4f})")

    if 'ts_code' in triplet_df.columns and len(triplet_df) > 0:
        triplet_df[['ts_code']].drop_duplicates().to_csv(os.path.join(stage_dir, '事件公司关联结果.csv'), index=False, encoding='utf-8-sig')
    logger.info("  关联结果已保存")
    logger.info("阶段3完成")
    return True


def run_stage4():
    import numpy as np
    import pandas as pd
    from scipy import stats as scipy_stats
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from tqdm import tqdm
    import joblib

    logger.info("阶段4：影响预测 开始")

    stage_dir = os.path.join(RESULT_DIR, '5.3_事件影响预测')
    os.makedirs(stage_dir, exist_ok=True)

    # 1. 加载数据
    logger.info("加载增强版数据...")
    train_path = os.path.join(DATA_DIR, '建模核心表', '训练集_增强_v3清洗后.csv')
    val_path = os.path.join(DATA_DIR, '建模核心表', '验证集_增强_v3清洗后.csv')
    test_path = os.path.join(DATA_DIR, '建模核心表', '测试集_增强_v3清洗后.csv')
    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            alt = p.replace('_v3清洗后', '').replace('_v3', '')
            if os.path.exists(alt):
                pass

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    logger.info(f"  训练集: {len(train_df)} 条 | 验证集: {len(val_df)} 条 | 测试集: {len(test_df)} 条")

    car_col = 'CAR_post_new'
    if car_col not in train_df.columns:
        for c in ['CAR_post_rec', 'CAR_post_new', 'CAR_post', 'CAR_new', 'CAR', 'car']:
            if c in train_df.columns:
                car_col = c
                break
    logger.info(f"  使用CAR列: {car_col}")

    for df in [train_df, val_df, test_df]:
        df[car_col] = pd.to_numeric(df[car_col], errors='coerce')
        df['label_v12'] = df[car_col].apply(lambda x: 1 if x > 2 else (-1 if x < -2 else 0))
    logger.info(f"  Label分布(训练集): {train_df['label_v12'].value_counts().sort_index().to_dict()}")

    for df in [train_df, val_df, test_df]:
        if 'event_date' in df.columns:
            df['event_date_parsed'] = safe_parse_date(df['event_date'])

    bench_path = os.path.join(DATA_DIR, '股票行情数据', '沪深300指数_清洗.csv')
    bench_df = None
    if os.path.exists(bench_path):
        bench_df = pd.read_csv(bench_path)
        bench_df['trade_date'] = safe_parse_date(bench_df['trade_date'])
        bench_df = bench_df.dropna(subset=['trade_date']).sort_values('trade_date')
        if 'mkt_ret' in bench_df.columns:
            bench_df['mkt_ret'] = pd.to_numeric(bench_df['mkt_ret'], errors='coerce')

    # E1: 事件研究法统计模型（基线）
    logger.info("E1: 事件研究法统计模型（基线）")

    train_df[car_col] = pd.to_numeric(train_df[car_col], errors='coerce')

    def compute_sub_stats(data, car_col, label_col='label_v12'):
        data = data.copy()
        data[car_col] = pd.to_numeric(data[car_col], errors='coerce')
        stats = data.groupby('event_sub').agg(
            样本数=(car_col, 'count'), 平均CAR=(car_col, 'mean'), CAR标准差=(car_col, 'std'),
            上涨次数=(label_col, lambda x: (x == 1).sum()), 下跌次数=(label_col, lambda x: (x == -1).sum()),
            平盘次数=(label_col, lambda x: (x == 0).sum()), 总次数=(label_col, 'count'),
        ).reset_index()
        stats['胜率'] = stats['上涨次数'] / stats['总次数'].clip(lower=1)
        stats['t统计量'] = stats['平均CAR'] / (stats['CAR标准差'] / np.sqrt(stats['样本数']).clip(lower=1))
        stats['p值'] = 2 * (1 - scipy_stats.t.cdf(abs(stats['t统计量']).fillna(0), stats['样本数'].clip(lower=2) - 1))
        stats['是否显著'] = stats['p值'] < 0.05
        stats['方向'] = stats['平均CAR'].apply(lambda x: '涨' if x > 0 else ('跌' if x < 0 else '平'))
        return stats

    sub_stats = compute_sub_stats(train_df, car_col)
    logger.info("  事件子类统计:")
    for _, row in sub_stats.iterrows():
        sig_mark = '显著' if row['是否显著'] else '不显著'
        logger.info(f"    {row['event_sub']}: CAR={row['平均CAR']:+.4f}, 胜率={row['胜率']:.2%}, "
                    f"样本={row['样本数']}, t={row['t统计量']:.2f} ({sig_mark})")
    sub_stats.to_csv(os.path.join(stage_dir, '事件子类统计.csv'), index=False, encoding='utf-8-sig')

    # 贝叶斯事件先验（James-Stein收缩估计器）
    logger.info("  贝叶斯事件先验（James-Stein 收缩估计器）")

    global_mean_car = train_df[car_col].mean()
    global_std_car = train_df[car_col].std()
    global_win_rate = (train_df['label_v12'] == 1).mean()
    N_total = len(train_df)
    logger.info(f"  全局先验: CAR_mean={global_mean_car:+.4f}, win_rate={global_win_rate:.3f}, N={N_total}")

    bayesian_priors = {}
    for _, row in sub_stats.iterrows():
        sub = row['event_sub']
        n_obs = int(row['样本数'])
        obs_mean = float(row['平均CAR'])
        obs_win_rate = float(row['胜率'])
        obs_std = float(row['CAR标准差']) if pd.notna(row['CAR标准差']) else global_std_car

        shrinkage_k = 20
        shrinkage_factor = n_obs / (n_obs + shrinkage_k)

        post_mean_car = shrinkage_factor * obs_mean + (1 - shrinkage_factor) * global_mean_car
        post_win_rate = shrinkage_factor * obs_win_rate + (1 - shrinkage_factor) * global_win_rate

        post_std = obs_std / np.sqrt(max(n_obs, 1))

        info_ratio = abs(post_mean_car) / max(post_std, 0.001) if post_std > 0 else 0

        bayesian_priors[sub] = {
            'post_mean_car': round(post_mean_car, 4),
            'post_win_rate': round(post_win_rate, 4),
            'post_std': round(post_std, 4),
            'shrinkage_factor': round(shrinkage_factor, 3),
            'obs_n': n_obs,
            'info_ratio': round(info_ratio, 3),
            'is_positive': post_mean_car > 0,
            'significant': bool(row['是否显著']),
        }

        logger.info(f"    {sub}: 观测CAR={obs_mean:+.4f}(n={n_obs}) -> "
                    f"后验CAR={post_mean_car:+.4f}(收缩={shrinkage_factor:.2f}), "
                    f"信息比={info_ratio:.3f}")

    bayes_df = pd.DataFrame([
        {'event_sub': k, **v} for k, v in bayesian_priors.items()
    ])
    bayes_df.to_csv(os.path.join(stage_dir, '贝叶斯事件先验.csv'), index=False, encoding='utf-8-sig')

    # 构建LGBM特征矩阵
    logger.info("构建LGBM特征矩阵（技术因子 + 事件编码 + 交互特征 + 贝叶斯先验）")

    TECH_FACTOR_COLS = [
        'MA5', 'MA10', 'MA20', 'MA60',
        'dev_MA5', 'dev_MA10', 'dev_MA20', 'dev_MA60',
        'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
        'volatility_20d', 'RSI14',
        'MACD_dif', 'MACD_dea', 'MACD_hist',
        'BOLL_upper', 'BOLL_lower', 'BOLL_width', 'BOLL_pos',
        'vol_ma5', 'vol_ratio',
        'ATR14', 'ATR14_pct',
        'gap_up', 'gap_down', 'gap_pct',
        'alpha', 'beta', 'decline_days', 'close_t1',
    ]
    available_tech_cols = [c for c in TECH_FACTOR_COLS if c in train_df.columns]
    if len(available_tech_cols) < 5:
        available_tech_cols = [c for c in ['ret_5d', 'ret_20d', 'volatility_20d', 'vol_ratio', 'ATR14', 'BOLL_pos', 'decline_days', 'gap_pct', 'RSI14', 'alpha'] if c in train_df.columns]
    logger.info(f"  技术因子({len(available_tech_cols)}): {available_tech_cols[:8]}...")

    event_sub_encoder = LabelEncoder()
    train_df['_event_sub_code'] = event_sub_encoder.fit_transform(train_df['event_sub'].astype(str))
    val_df['_event_sub_code'] = event_sub_encoder.transform(val_df['event_sub'].astype(str))
    test_df['_event_sub_code'] = event_sub_encoder.transform(test_df['event_sub'].astype(str))
    n_event_subs = len(event_sub_encoder.classes_)
    logger.info(f"  事件子类别: {n_event_subs} 类 -> {event_sub_encoder.classes_.tolist()}")

    def add_market_regime(df, bench_df):
        if bench_df is None or 'mkt_ret' not in bench_df.columns:
            df['market_regime'] = '震荡'
            return df
        df = df.copy()
        bench_sorted = bench_df.sort_values('trade_date')
        bench_sorted['mkt_ret_5d'] = bench_sorted['mkt_ret'].rolling(5, min_periods=1).sum()
        dc = 'event_date_parsed' if 'event_date_parsed' in df.columns else 'event_date'

        def get_regime(ed):
            if pd.isna(ed):
                return '震荡'
            nearest = bench_sorted[bench_sorted['trade_date'] <= ed]
            if len(nearest) == 0:
                return '震荡'
            r5 = nearest.iloc[-1].get('mkt_ret_5d', 0)
            if pd.isna(r5):
                return '震荡'
            return '牛市' if r5 > 0.02 else ('熊市' if r5 < -0.02 else '震荡')

        df['market_regime'] = df[dc].apply(get_regime)
        return df

    def add_tech_state(df):
        df = df.copy()
        rsi_c = next((c for c in ['RSI14', 'rsi_14', 'RSI_14', 'rsi'] if c in df.columns), None)
        mom_c = next((c for c in ['ret_20d', 'mom_20d'] if c in df.columns), None)
        if rsi_c:
            df[rsi_c] = pd.to_numeric(df[rsi_c], errors='coerce')
            df['tech_state'] = df[rsi_c].apply(lambda x: '超卖' if x < 30 else ('超买' if x > 70 else '中性'))
        elif mom_c:
            df[mom_c] = pd.to_numeric(df[mom_c], errors='coerce')
            df['tech_state'] = df[mom_c].apply(lambda x: '超卖' if x < -5 else ('超买' if x > 5 else '中性'))
        else:
            df['tech_state'] = '中性'
        return df

    train_df = add_market_regime(train_df, bench_df)
    train_df = add_tech_state(train_df)
    val_df = add_market_regime(val_df, bench_df)
    val_df = add_tech_state(val_df)
    test_df = add_market_regime(test_df, bench_df)
    test_df = add_tech_state(test_df)

    regime_encoder = LabelEncoder()
    train_df['_regime_code'] = regime_encoder.fit_transform(train_df['market_regime'])
    val_df['_regime_code'] = regime_encoder.transform(val_df['market_regime'])
    test_df['_regime_code'] = regime_encoder.transform(test_df['market_regime'])

    state_encoder = LabelEncoder()
    train_df['_state_code'] = state_encoder.fit_transform(train_df['tech_state'])
    val_df['_state_code'] = state_encoder.transform(val_df['tech_state'])
    test_df['_state_code'] = state_encoder.transform(test_df['tech_state'])

    def build_lgbm_features(df, available_tech_cols, bayesian_priors, fit_scaler=None):
        from scipy.stats import rankdata
        features_list = []
        feature_names = []

        # Group A: 技术因子
        scaler = StandardScaler()
        tech_raw = df[available_tech_cols].fillna(0).values
        if fit_scaler is not None:
            tech_scaled = fit_scaler.transform(tech_raw)
        else:
            tech_scaled = scaler.fit_transform(tech_raw)
        features_list.append(tech_scaled)
        feature_names.extend([f'tech_{c}' for c in available_tech_cols])

        # Group B: event_sub one-hot
        event_onehot = np.zeros((len(df), n_event_subs), dtype=np.float32)
        for i, code in enumerate(df['_event_sub_code'].values):
            event_onehot[i, code] = 1.0
        features_list.append(event_onehot)
        feature_names.extend([f'ev_{event_sub_encoder.classes_[c]}'.replace('/', '_').replace(' ', '_') for c in range(n_event_subs)])

        # Group C: market_regime one-hot
        regime_onehot = np.zeros((len(df), len(regime_encoder.classes_)), dtype=np.float32)
        for i, code in enumerate(df['_regime_code'].values):
            regime_onehot[i, code] = 1.0
        features_list.append(regime_onehot)
        feature_names.extend([f'regime_{r}' for r in regime_encoder.classes_])

        # Group D: tech_state one-hot
        state_onehot = np.zeros((len(df), len(state_encoder.classes_)), dtype=np.float32)
        for i, code in enumerate(df['_state_code'].values):
            state_onehot[i, code] = 1.0
        features_list.append(state_onehot)
        feature_names.extend([f'state_{s}' for s in state_encoder.classes_])

        # Group E: 交互特征
        if 'RSI14' in df.columns:
            rsi = df['RSI14'].fillna(50).values.reshape(-1, 1) / 100.0
            interact_rsi = event_onehot * rsi
            features_list.append(interact_rsi)
            feature_names.extend([f'int_rsi_{event_sub_encoder.classes_[c][:8]}' for c in range(min(n_event_subs, 11))])

        if 'ret_20d' in df.columns:
            ret20 = df['ret_20d'].fillna(0).values.reshape(-1, 1) / 50.0
            interact_ret20 = event_onehot * ret20
            features_list.append(interact_ret20)
            feature_names.extend([f'int_ret20_{event_sub_encoder.classes_[c][:8]}' for c in range(min(n_event_subs, 11))])

        if 'volatility_20d' in df.columns:
            vol = df['volatility_20d'].fillna(0).values.reshape(-1, 1) / 30.0
            interact_vol = event_onehot * vol
            features_list.append(interact_vol)
            feature_names.extend([f'int_vol_{event_sub_encoder.classes_[c][:8]}' for c in range(min(n_event_subs, 11))])

        if 'MACD_hist' in df.columns:
            macdh = df['MACD_hist'].fillna(0).values.reshape(-1, 1) / 1.0
            interact_macd = event_onehot * macdh
            features_list.append(interact_macd)
            feature_names.extend([f'int_macd_{event_sub_encoder.classes_[c][:8]}' for c in range(min(n_event_subs, 11))])

        # Group F: 贝叶斯先验特征
        bayes_features = np.zeros((len(df), 5), dtype=np.float32)
        for i, row in df.iterrows():
            sub = row.get('event_sub', '其他')
            bp = bayesian_priors.get(sub, {})
            bayes_features[i, 0] = bp.get('post_mean_car', 0.0)
            bayes_features[i, 1] = bp.get('post_win_rate', global_win_rate)
            bayes_features[i, 2] = bp.get('shrinkage_factor', 0.5)
            bayes_features[i, 3] = bp.get('info_ratio', 0.0)
            bayes_features[i, 4] = 1.0 if bp.get('is_positive', False) else 0.0
        features_list.append(bayes_features)
        feature_names.extend(['bayes_car', 'bayes_winrate', 'bayes_shrink', 'bayes_inforatio', 'bayes_ispos'])

        # Group G: 横截面排名特征
        cross_features = np.zeros((len(df), 17), dtype=np.float32)

        def safe_percentile_rank(values):
            values = np.asarray(values, dtype=float)
            n = len(values)
            if n <= 1 or np.nanstd(values) < 1e-10:
                return np.full(n, 50.0)
            ranks = rankdata(values, method='average')
            return (ranks - 1) / (n - 1) * 100.0

        fi = 0

        if 'RSI14' in df.columns:
            cross_features[:, fi] = safe_percentile_rank(df['RSI14'].fillna(50).values)
            fi += 1
        if 'ret_20d' in df.columns:
            cross_features[:, fi] = safe_percentile_rank(df['ret_20d'].fillna(0).values)
            fi += 1
        if 'volatility_20d' in df.columns:
            cross_features[:, fi] = safe_percentile_rank(df['volatility_20d'].fillna(0).values)
            fi += 1
        if 'alpha' in df.columns:
            cross_features[:, fi] = safe_percentile_rank(df['alpha'].fillna(0).values)
            fi += 1

        if 'ret_5d' in df.columns:
            cross_features[:, fi] = safe_percentile_rank(df['ret_5d'].fillna(0).values)
            fi += 1
        if 'MA5' in df.columns and 'MA60' in df.columns:
            ma_ratio = df['MA5'] / df['MA60'].replace(0, np.nan)
            cross_features[:, fi] = safe_percentile_rank(ma_ratio.fillna(1.0).values)
            fi += 1
        if 'ATR14_pct' in df.columns:
            cross_features[:, fi] = safe_percentile_rank(df['ATR14_pct'].fillna(0).values)
            fi += 1

        if 'BOLL_pos' in df.columns:
            cross_features[:, fi] = safe_percentile_rank(df['BOLL_pos'].fillna(0.5).values)
            fi += 1
        if 'decline_days' in df.columns:
            cross_features[:, fi] = safe_percentile_rank(df['decline_days'].fillna(0).values)
            fi += 1

        ev_scores_raw = np.array([
            bayesian_priors.get(row.get('event_sub', '其他'), {}).get('post_mean_car', 0.0)
            for _, row in df.iterrows()
        ])
        cross_features[:, fi] = safe_percentile_rank(ev_scores_raw)
        fi += 1
        ev_info_ratios = np.array([
            bayesian_priors.get(row.get('event_sub', '其他'), {}).get('info_ratio', 0.0)
            for _, row in df.iterrows()
        ])
        cross_features[:, fi] = safe_percentile_rank(ev_info_ratios)
        fi += 1
        ev_win_rates = np.array([
            bayesian_priors.get(row.get('event_sub', '其他'), {}).get('post_win_rate', 0.333)
            for _, row in df.iterrows()
        ])
        cross_features[:, fi] = safe_percentile_rank(ev_win_rates)
        fi += 1

        if 'vol_ratio' in df.columns:
            cross_features[:, fi] = safe_percentile_rank(df['vol_ratio'].fillna(1.0).values)
            fi += 1
        if 'vol_ma5' in df.columns:
            cross_features[:, fi] = safe_percentile_rank(df['vol_ma5'].fillna(0).values)
            fi += 1

        if 'MACD_dif' in df.columns:
            cross_features[:, fi] = safe_percentile_rank(df['MACD_dif'].fillna(0).values)
            fi += 1
        if 'close_t1' in df.columns:
            cross_features[:, fi] = safe_percentile_rank(df['close_t1'].fillna(0).values)
            fi += 1

        tech_score_cols = ['RSI14', 'ret_20d', 'alpha']
        avail_score_cols = [c for c in tech_score_cols if c in df.columns]
        if len(avail_score_cols) >= 2:
            tech_combo = df[avail_score_cols].fillna(0).values
            tech_z = (tech_combo - tech_combo.mean(axis=0)) / (tech_combo.std(axis=0) + 1e-8)
            composite_tech = tech_z.mean(axis=1)
            cross_features[:, fi] = safe_percentile_rank(composite_tech)
            fi += 1
        elif 'RSI14' in df.columns:
            cross_features[:, fi] = safe_percentile_rank(df['RSI14'].fillna(50).values)
            fi += 1

        logger.info(f"  横截面排名特征(Group G): {fi}维")
        features_list.append(cross_features[:, :fi])
        feature_names.extend([f'xrank_RSI14', f'xrank_mom20d', f'xrank_vol20d', f'xrank_alpha',
                              f'xrank_mom5d', f'xrank_MA_ratio', f'xrank_ATRpct',
                              f'xrank_BOLLpos', f'xrank_declinedays',
                              f'xrank_evCAR', f'xrank_evInfoRatio', f'xrank_evWinRate',
                              f'xrank_volratio', f'xrank_volma5',
                              f'xrank_MACDdif', f'xrank_closeT1', f'xrank_compositeTech'][:fi])

        X = np.hstack(features_list)
        return X, feature_names, scaler

    logger.info("  构建训练集特征...")
    X_train, feature_names, scaler = build_lgbm_features(train_df, available_tech_cols, bayesian_priors, fit_scaler=None)
    X_val, _, _ = build_lgbm_features(val_df, available_tech_cols, bayesian_priors, fit_scaler=scaler)
    X_test, _, _ = build_lgbm_features(test_df, available_tech_cols, bayesian_priors, fit_scaler=scaler)

    y_train = train_df['label_v12'].values
    y_val = val_df['label_v12'].values
    y_test = test_df['label_v12'].values

    label_map = {-1: 0, 0: 1, 1: 2}
    y_train_enc = np.array([label_map[y] for y in y_train])
    y_val_enc = np.array([label_map[y] for y in y_val])
    y_test_enc = np.array([label_map[y] for y in y_test])

    logger.info(f"  特征矩阵维度: 训练={X_train.shape}, 验证={X_val.shape}, 测试={X_test.shape}")
    logger.info(f"  特征数量: {len(feature_names)} (技术:{len(available_tech_cols)} + 事件:{n_event_subs} "
                f"+ 状态:{len(regime_encoder.classes_) + len(state_encoder.classes_)} + 交互 + 贝叶斯)")

    joblib.dump(scaler, os.path.join(stage_dir, 'feature_scaler.pkl'))

    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    serializable_priors = make_json_serializable(bayesian_priors)

    with open(os.path.join(stage_dir, 'tech_factor_cols.json'), 'w', encoding='utf-8') as f:
        json.dump(available_tech_cols, f, ensure_ascii=False, indent=2)

    with open(os.path.join(stage_dir, 'feature_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'tech_cols': available_tech_cols,
            'feature_names': feature_names,
            'n_event_subs': n_event_subs,
            'event_sub_classes': list(event_sub_encoder.classes_),
            'regime_classes': list(regime_encoder.classes_),
            'state_classes': list(state_encoder.classes_),
            'bayesian_priors': serializable_priors,
            'label_map': label_map,
        }, f, ensure_ascii=False, indent=2)
    logger.info("  特征工程artifacts已保存")

    # LGBM训练与消融实验
    import lightgbm as lgb

    def train_lgbm(X_tr, y_tr, X_va, y_va, params_override=None, feature_subset=None,
                   mode='classification', car_target=None):
        from scipy.stats import spearmanr
        fs = feature_subset if feature_subset else list(range(X_tr.shape[1]))
        Xt, Xv = X_tr[:, fs], X_va[:, fs]

        use_gpu = False
        try:
            _ = lgb.Dataset(np.zeros((10, 5)), label=np.zeros(10, dtype=int))
            obj = 'regression' if mode == 'regression' else 'multiclass'
            p = {'device': 'gpu', 'verbose': -1, 'objective': obj, 'n_estimators': 2}
            if obj == 'multiclass':
                p['num_class'] = 3
            _ = lgb.train(p, lgb.Dataset(np.zeros((10, 5)), label=np.zeros(10, dtype=int)),
                          num_boost_round=2)
            use_gpu = True
        except Exception:
            pass

        if mode == 'regression':
            yt = car_target
            yv = y_va

            base_params = {
                'objective': 'regression', 'metric': 'mae',
                'boosting_type': 'gbdt', 'n_estimators': 500, 'learning_rate': 0.03,
                'num_leaves': 31, 'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.7,
                'reg_alpha': 0.1, 'reg_lambda': 1.0, 'verbose': -1, 'random_state': 42,
                'early_stopping_rounds': 30,
                'device': 'gpu' if use_gpu else 'cpu',
            }
            if params_override:
                base_params.update(params_override)

            logger.info(f"  LGBM设备: {'GPU' if use_gpu else 'CPU'} | 模式=REGRESSION | "
                        f"特征={Xt.shape[1]} | 样本={len(X_tr)}/{len(X_va)}")

            dtrain = lgb.Dataset(Xt, label=yt)
            dval = lgb.Dataset(Xv, label=yv)
            model = lgb.train(base_params, dtrain, valid_sets=[dval],
                              callbacks=[lgb.log_evaluation(period=0)])

            pred_car = model.predict(Xv)
            pred_labels_3cls = np.where(pred_car > 2, 2, np.where(pred_car < -2, 0, 1))

            rank_ic, ic_pval = spearmanr(pred_car, yv)
            mae_val = np.mean(np.abs(pred_car - yv))
            corr, _ = spearmanr(pred_car, yv)

            actual_dir = (yv > 0).astype(int)
            pred_dir = (pred_car > 0).astype(int)
            dir_acc = (actual_dir == pred_dir).mean()

            logger.info(f"  回归指标: Rank_IC={rank_ic:.4f}(p={ic_pval:.4f}), MAE={mae_val:.4f}, "
                        f"Spearman={corr:.4f}, 方向准确率={dir_acc:.4f}")

            return model, pred_labels_3cls, pred_car, float(rank_ic), float(dir_acc), fs

        else:
            base_params = {
                'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
                'boosting_type': 'gbdt', 'n_estimators': 500, 'learning_rate': 0.05,
                'num_leaves': 31, 'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.7,
                'reg_alpha': 0.1, 'reg_lambda': 1.0, 'verbose': -1, 'random_state': 42,
                'early_stopping_rounds': 30, 'class_weight': 'balanced',
                'device': 'gpu' if use_gpu else 'cpu',
            }
            if params_override:
                base_params.update(params_override)

            logger.info(f"  LGBM设备: {'GPU' if use_gpu else 'CPU'} | 模式=CLASSIFICATION | "
                        f"特征={Xt.shape[1]} | 样本={len(X_tr)}/{len(X_va)}")

            dtrain = lgb.Dataset(Xt, label=y_tr)
            dval = lgb.Dataset(Xv, label=y_va, reference=dtrain)
            model = lgb.train(base_params, dtrain, valid_sets=[dval],
                              callbacks=[lgb.log_evaluation(period=0)])

            preds = model.predict(Xv)
            pred_labels = preds.argmax(axis=1)
            f1 = f1_score(y_va, pred_labels, average='weighted')
            acc = accuracy_score(y_va, pred_labels)
            return model, pred_labels, preds, f1, acc, fs

    # E2: LGBM纯技术因子
    logger.info("E2: LGBM纯技术因子（分类 + 回归）")
    tech_feature_idx = list(range(len(available_tech_cols)))

    e2_model_cls, e2_preds_cls, e2_probs_cls, e2_f1, e2_acc, _ = train_lgbm(
        X_train, y_train_enc, X_val, y_val_enc, feature_subset=tech_feature_idx
    )

    car_train = train_df[car_col].fillna(0).values.astype(float)
    car_val = val_df[car_col].fillna(0).values.astype(float)
    car_test = test_df[car_col].fillna(0).values.astype(float)

    e2_model_reg, e2_pred_labels_reg, e2_pred_car, e2_rank_ic, e2_dir_acc, _ = train_lgbm(
        X_train, y_train_enc, X_val, car_val,
        feature_subset=tech_feature_idx,
        mode='regression', car_target=car_train
    )
    logger.info(f"  E2-分类: F1={e2_f1:.4f}, Acc={e2_acc:.4f}")
    logger.info(f"  E2-回归: Rank_IC={e2_rank_ic:.4f}, 方向准确率={e2_dir_acc:.4f}")

    # E3: LGBM技术 + 事件one-hot
    logger.info("E3: LGBM技术 + 事件one-hot（分类 + 回归）")
    e3_end_idx = len(available_tech_cols) + n_event_subs
    e3_feature_idx = list(range(e3_end_idx))

    e3_model_cls, e3_preds_cls, e3_probs_cls, e3_f1, e3_acc, _ = train_lgbm(
        X_train, y_train_enc, X_val, y_val_enc, feature_subset=e3_feature_idx
    )

    e3_model_reg, e3_pred_labels_reg, e3_pred_car, e3_rank_ic, e3_dir_acc, _ = train_lgbm(
        X_train, y_train_enc, X_val, car_val,
        feature_subset=e3_feature_idx,
        mode='regression', car_target=car_train
    )
    logger.info(f"  E3-分类: F1={e3_f1:.4f}, Acc={e3_acc:.4f}")
    logger.info(f"  E3-回归: Rank_IC={e3_rank_ic:.4f}, 方向准确率={e3_dir_acc:.4f}")

    # E4: LGBM全特征
    logger.info("E4: LGBM全特征(+交互+横截面排名)（分类 + 回归）")
    e4_feature_idx = list(range(X_train.shape[1]))
    e4_params = {
        'n_estimators': 800, 'learning_rate': 0.03,
        'num_leaves': 45, 'min_child_samples': 15,
        'colsample_bytree': 0.75, 'subsample': 0.85,
        'reg_alpha': 0.08, 'reg_lambda': 0.8,
    }

    e4_model_cls, e4_preds_cls, e4_probs_cls, e4_f1, e4_acc, _ = train_lgbm(
        X_train, y_train_enc, X_val, y_val_enc, params_override=e4_params, feature_subset=e4_feature_idx
    )

    e4_params_reg = {
        'n_estimators': 1000, 'learning_rate': 0.02,
        'num_leaves': 45, 'min_child_samples': 15,
        'colsample_bytree': 0.75, 'subsample': 0.85,
        'reg_alpha': 0.05, 'reg_lambda': 0.5,
    }
    e4_model_reg, e4_pred_labels_reg, e4_pred_car, e4_rank_ic, e4_dir_acc, _ = train_lgbm(
        X_train, y_train_enc, X_val, car_val,
        params_override=e4_params_reg, feature_subset=e4_feature_idx,
        mode='regression', car_target=car_train
    )
    logger.info(f"  E4-分类: F1={e4_f1:.4f}, Acc={e4_acc:.4f}")
    logger.info(f"  E4-回归: Rank_IC={e4_rank_ic:.4f}, 方向准确率={e4_dir_acc:.4f}")

    e4_model = e4_model_reg

    # E4特征重要性Top-20
    try:
        e4_importance = sorted(zip(e4_model.feature_importance(importance_type='gain'),
                                   [feature_names[i] for i in range(len(feature_names))]), reverse=True)[:20]
        logger.info("  Top-20重要特征(回归模型gain):")
        for imp, name in e4_importance:
            logger.info(f"    {name:>35s}: gain={imp:.1f}")
    except Exception:
        logger.warning("  特征重要性提取失败")

    # E5: 分类 vs 回归对比
    logger.info("E5: 分类 vs 回归对比 + 组件增量分析")
    logger.info(f"  E2(纯技术):   分类F1={e2_f1:.4f} | 回归IC={e2_rank_ic:.4f}, 方向Acc={e2_dir_acc:.4f}")
    logger.info(f"  E3(+事件):    分类F1={e3_f1:.4f} | 回归IC={e3_rank_ic:.4f}, 方向Acc={e3_dir_acc:.4f}")
    logger.info(f"  E4(+交互+排名): 分类F1={e4_f1:.4f} | 回归IC={e4_rank_ic:.4f}, 方向Acc={e4_dir_acc:.4f}")

    e5_delta_ic_e4vse3 = e4_rank_ic - e3_rank_ic
    e5_delta_ic_e3vse2 = e3_rank_ic - e2_rank_ic
    logger.info(f"  回归增量: E3->E4 IC delta={e5_delta_ic_e4vse3:+.4f}, E2->E3 IC delta={e5_delta_ic_e3vse2:+.4f}")

    # E6: 市场状态条件化
    logger.info("E6: 市场状态条件化（回归版）")
    e6_pred_car = np.zeros(len(X_val))
    for regime_name, regime_code in zip(regime_encoder.classes_, range(len(regime_encoder.classes_))):
        mask_train = train_df['_regime_code'] == regime_code
        mask_val = val_df['_regime_code'] == regime_code
        if mask_train.sum() < 30 or mask_val.sum() < 10:
            continue
        try:
            m_e6, p_e6_l, pr_e6_car, ic_e6, da_e6, _ = train_lgbm(
                X_train[mask_train], y_train_enc[mask_train],
                X_val[mask_val], car_val[mask_val],
                feature_subset=e4_feature_idx,
                mode='regression', car_target=car_train[mask_train],
                params_override={'n_estimators': 300}
            )
            e6_pred_car[mask_val] = pr_e6_car
            logger.info(f"    {regime_name}: n_train={mask_train.sum()}, IC={ic_e6:.4f}")
        except Exception as ex:
            logger.warning(f"    {regime_name}: 训练失败({ex})，回退到E4全局")
            e6_pred_car[mask_val] = e4_pred_car[mask_val]

    from scipy.stats import spearmanr as sp_spearmanr
    e6_ic, _ = sp_spearmanr(e6_pred_car, car_val)
    e6_dir_acc = ((e6_pred_car > 0).astype(int) == (car_val > 0).astype(int)).mean()
    logger.info(f"  E6-条件化回归整体: Rank_IC={e6_ic:.4f}, 方向准确率={e6_dir_acc:.4f}")

    # E7: 综合融合
    logger.info("E7: 综合融合（LR方向+回归排序+事件过滤+追涨防御）")

    test_pred_car = e4_model.predict(X_test[:, e4_feature_idx])
    logger.info(f"  测试集回归预测CAR: mean={test_pred_car.mean():+.4f}, "
                f"std={test_pred_car.std():.4f}, min={test_pred_car.min():+.4f}, max={test_pred_car.max():+.4f}")

    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.preprocessing import StandardScaler as LRScaler

    # LR纯技术因子（不加排名/交互特征，避免共线性）
    tech_idx = list(range(len(available_tech_cols)))
    lr_feature_idx = tech_idx

    logger.info("  LR回退32维(纯技术因子), 排名特征仅用于LGBM")

    lr_scaler = LRScaler()
    X_train_lr = lr_scaler.fit_transform(X_train[:, lr_feature_idx])
    X_test_lr = lr_scaler.transform(X_test[:, lr_feature_idx])

    lr_direction = LR(C=0.5, class_weight='balanced', random_state=42, max_iter=500, solver='lbfgs')
    lr_direction.fit(X_train_lr, y_train_enc)
    lr_test_proba = lr_direction.predict_proba(X_test_lr)
    lr_up_prob = lr_test_proba[:, 2]
    lr_pred_dir = np.argmax(lr_test_proba, axis=1)

    lr_f1_enhanced = f1_score(y_test_enc, lr_pred_dir, average='weighted')
    lr_acc_enhanced = accuracy_score(y_test_enc, lr_pred_dir)
    logger.info(f"  LR(32维纯技术) F1={lr_f1_enhanced:.4f}, Acc={lr_acc_enhanced:.4f}")

    joblib.dump(lr_direction, os.path.join(stage_dir, 'e4_lr_model.pkl'))
    joblib.dump(lr_scaler, os.path.join(stage_dir, 'e4_lr_scaler.pkl'))
    logger.info("  LR模型和Scaler已保存")

    test_preds_enc = np.where(test_pred_car > 2, 2, np.where(test_pred_car < -2, 0, 1))
    test_up_prob = 1 / (1 + np.exp(-test_pred_car * 2))

    sub_stats_dict = {}
    for _, row in sub_stats.iterrows():
        sub_stats_dict[row['event_sub']] = {
            'avg_car': row['平均CAR'], 'win_rate': row['胜率'],
            't_stat': row['t统计量'], 'p_value': row['p值'],
            'significant': row['是否显著'], 'direction': row['方向'],
            'sample_count': row['样本数'],
        }

    ev_scores = np.array([
        compute_event_score(row, bayesian_priors) for _, row in test_df.iterrows()
    ])
    logger.info(f"  贝叶斯事件得分: mean={ev_scores.mean():.3f}, std={ev_scores.std():.3f}")

    # E7融合逻辑
    rsi_values = test_df['RSI14'].fillna(50).values if 'RSI14' in test_df.columns else np.full(len(test_df), 50.0)

    direction = np.array(lr_pred_dir.copy())

    car_negative_mask = (direction == 2) & (test_pred_car < -0.5)
    direction[car_negative_mask] = 1
    logger.info(f"  LR-CAR矛盾过滤: {car_negative_mask.sum()}条(LR看涨但CAR预测负)")

    ev_mean = ev_scores.mean()
    ev_std = ev_scores.std()
    veto_threshold = max(0.10, ev_mean - 0.5 * ev_std)
    logger.info(f"  否决动态阈值: {veto_threshold:.4f} (ev_mean={ev_mean:.3f}, ev_std={ev_std:.3f})")

    veto_mask = (direction == 2) & (ev_scores < veto_threshold)
    direction[veto_mask] = 1

    ev_q25 = np.percentile(ev_scores, 25)
    chase_veto_mask = (direction == 2) & (rsi_values > 70) & (ev_scores < ev_q25)
    direction[chase_veto_mask] = 1

    rank_score = lr_up_prob * 0.60 + ev_scores * 0.20 + test_up_prob * 0.20

    composite_scores = np.zeros(len(test_df))
    composite_scores[direction == 2] = rank_score[direction == 2]
    composite_scores[direction == 0] = -rank_score[direction == 0]

    fusion_preds_decoded = [{0: -1, 1: 0, 2: 1}[d] for d in direction]

    e7_f1 = f1_score(y_test, fusion_preds_decoded, average='weighted')
    e7_acc = accuracy_score(y_test, fusion_preds_decoded)
    logger.info(f"  E7综合融合: Accuracy={e7_acc:.4f}, Weighted-F1={e7_f1:.4f}")
    logger.info(f"  看涨信号: {(np.array(fusion_preds_decoded) == 1).sum()}, "
                f"看跌信号: {(np.array(fusion_preds_decoded) == -1).sum()}, "
                f"看平信号: {(np.array(fusion_preds_decoded) == 0).sum()}")

    # LightGBM's native writer does not accept non-ASCII Windows paths.
    with open(os.path.join(stage_dir, 'e4_lgbm_model.txt'), 'w', encoding='utf-8') as f:
        f.write(e4_model.model_to_string())

    test_df['事件研究法_预测标签'] = 0
    test_df['事件研究法_置信度'] = 0.3
    test_df['预测CAR'] = test_pred_car
    test_df['双塔_预测标签'] = fusion_preds_decoded
    test_df['双塔_预测置信度'] = test_up_prob
    test_df['事件驱动得分'] = ev_scores
    test_df['贝叶斯_信息比'] = [
        bayesian_priors.get(row.get('event_sub', '其他'), {}).get('info_ratio', 0.0)
        for _, row in test_df.iterrows()
    ]
    test_df['技术看涨概率'] = test_up_prob
    test_df['双塔看涨得分'] = test_up_prob
    test_df['融合_预测标签'] = fusion_preds_decoded
    test_df['融合_预测置信度'] = composite_scores
    test_df['综合评分'] = composite_scores

    fusion_reasons = []
    for i in range(len(test_df)):
        d = direction[i]
        ev = ev_scores[i]
        rsi = rsi_values[i]
        if d == 2 and ev < veto_threshold:
            fusion_reasons.append('看涨但贝叶斯事件否决')
        elif d == 2 and rsi > 70 and ev < ev_q25:
            fusion_reasons.append('看涨但追涨防御否决')
        elif d == 2 and ev >= 0.50:
            fusion_reasons.append('看涨+贝叶斯事件利好确认')
        elif d == 2 and ev >= ev_q25:
            fusion_reasons.append('看涨+事件中性')
        elif d == 0 and ev < ev_q25:
            fusion_reasons.append('看跌+贝叶斯利空共振')
        elif d == 0:
            fusion_reasons.append('看跌')
        else:
            fusion_reasons.append('看平')
    test_df['融合决策原因'] = fusion_reasons

    test_df.to_csv(os.path.join(stage_dir, '测试集预测结果_完整.csv'), index=False, encoding='utf-8-sig')

    # 消融实验结果汇总
    logger.info("  消融实验结果汇总（分类 + 回归双轨对比）")
    ablation_results = pd.DataFrame({
        '实验': [
            'E1-事件研究法统计(基线)',
            'E2-LGBM纯技术因子',
            'E3-LGBM技术+事件one-hot',
            'E4-LGBM全特征(+交互+排名)',
            'E5-分类vs回归增量分析',
            'E6-市场状态条件化(回归)',
            'E7-综合融合(回归驱动)',
        ],
        '分类F1': [
            '-',
            f'{e2_f1:.4f}',
            f'{e3_f1:.4f}',
            f'{e4_f1:.4f}',
            '-',
            '-',
            f'{e7_f1:.4f}',
        ],
        '回归Rank_IC': [
            '-',
            f'{e2_rank_ic:.4f}',
            f'{e3_rank_ic:.4f}',
            f'{e4_rank_ic:.4f}',
            f'E3->E4: IC增量={e5_delta_ic_e4vse3:+.4f}',
            f'{e6_ic:.4f}',
            '-',
        ],
        '方向准确率': [
            '-',
            f'{e2_dir_acc:.4f}',
            f'{e3_dir_acc:.4f}',
            f'{e4_dir_acc:.4f}',
            f'E2->E3: IC增量={e5_delta_ic_e3vse2:+.4f}',
            f'{e6_dir_acc:.4f}',
            '-',
        ],
        '方法说明': [
            '规则系统（胜率/t检验）',
            f'{len(available_tech_cols)}个技术因子',
            f'+{n_event_subs}维事件one-hot',
            '+交互+横截面排名+贝叶斯先验',
            '量化各组件增量贡献',
            '按牛/熊/震分别回归建模',
            '回归CAR+动态否决+追涨防御',
        ],
    })

    es_preds = [1 if s.get('win_rate', 0.33) > 0.55 else (-1 if s.get('win_rate', 0.33) < 0.45 else 0)
                for s in [sub_stats_dict.get(sub, {}) for sub in test_df['event_sub']]]
    es_f1 = f1_score(y_test, es_preds, average='weighted')
    ablation_results.loc[0, '分类F1'] = f'{es_f1:.4f}'

    ablation_results.to_csv(os.path.join(stage_dir, '消融实验结果.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"\n{ablation_results.to_string(index=False)}")

    try:
        importance_df = pd.DataFrame({
            'feature_name': [feature_names[i] for i in range(len(feature_names))],
            'gain': e4_model.feature_importance(importance_type='gain'),
            'split': e4_model.feature_importance(importance_type='split'),
        }).sort_values('gain', ascending=False)
        importance_df.to_csv(os.path.join(stage_dir, '特征重要性.csv'), index=False, encoding='utf-8-sig')
    except Exception as e_imp:
        logger.warning(f"  特征重要性保存失败: {e_imp}")

    logger.info("阶段4完成")
    return True


def compute_event_score(row, bayesian_priors):
    sub = row.get('event_sub', '其他')
    bp = bayesian_priors.get(sub, {})

    post_wr = bp.get('post_win_rate', 0.333)
    post_car = bp.get('post_mean_car', 0.0)
    info_ratio = bp.get('info_ratio', 0.0)
    is_sig = bp.get('significant', False)

    if post_wr >= 0.42:
        score = 0.52 + min(0.25, (post_wr - 0.42) * 3.5)
    elif post_wr >= 0.36:
        score = 0.38 + (post_wr - 0.36) * 2.3
    elif post_wr >= 0.30:
        score = 0.28 + (post_wr - 0.30) * 1.7
    else:
        score = max(0.08, 0.18 + (post_wr - 0.30) * 1.5)

    if post_car > 0.3:
        score = min(0.95, score + 0.06)
    elif post_car > 0.1:
        score = min(0.90, score + 0.03)
    elif post_car < -0.3:
        score = max(0.05, score - 0.06)
    elif post_car < -0.1:
        score = max(0.10, score - 0.03)

    if is_sig and post_wr >= 0.36:
        score = min(0.95, score + 0.04)

    if info_ratio > 0.5:
        score = min(0.95, score + min(0.05, info_ratio * 0.04))

    sentiment = str(row.get('情绪标签', ''))
    if sentiment in ['利好', '微偏正面']:
        score = min(0.95, score + 0.05)
    elif sentiment in ['利空', '微偏负面']:
        score = max(0.05, score - 0.05)

    title = str(row.get('title', ''))
    neg_kw = ['续亏', '首亏', '略减', '预减', '减持', '质押', '冻结', '违规', '处罚', '诉讼', '退市', '暴雷']
    pos_kw = ['增持', '回购', '分红', '预增', '略增', '扭亏', '中标', '创新高', '突破']
    has_neg = any(kw in title for kw in neg_kw)
    has_pos = any(kw in title for kw in pos_kw)
    if has_pos and not has_neg:
        score = min(0.95, score + 0.08)
    elif has_neg and not has_pos:
        score = max(0.05, score - 0.08)

    return round(max(0.05, min(0.95, score)), 4)


def refine_event_name(title, event_sub=''):
    import re
    name = str(title).strip()
    if not name or name in ('nan', 'None', ''):
        return str(event_sub)[:40] if event_sub else '未知事件'

    company = ''
    m_colon = re.match(r'^([^:]{2,10}):', name)
    if m_colon:
        company = m_colon.group(1).strip()
        name = name[len(m_colon.group(0)):]
        if name.startswith(company):
            name = name[len(company):]
        name = re.sub(r'^\s*\d{4,}-?\d*\s*', '', name)

    name = re.sub(
        r'[\u4e00-\u9fa5]*(?:国泰|中信|华泰|中金|招商|海通|广发|东方|兴业|国信|申万|银河|中信建投)[\w]*证券(?:股份有限公司|有限公司)?[^关]*关于[\u4e00-\u9fa5]{2,15}(?:股份有限公司|有限公司|集团股份有限公司|集团有限公司)',
        '', name)
    name = re.sub(r'^关于', '', name)
    name = re.sub(r'以[^，。、的]{2,15}方式', '', name)
    for suffix in ['的提示性公告', '的更正公告', '的补充公告', '的公告', '的实施方案']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    name = re.sub(r'暨(?:通知|完成|办理|披露|说明|补充)[^，。]*$', '', name)
    name = re.sub(r'(.{2,6})\1', r'\1', name)
    name = name.strip()

    if company and not name.startswith(company):
        result = company + name
    else:
        result = name if name else company
    result = result.strip()
    return (result[:39] + '...') if len(result) > 40 else (result if result else str(event_sub)[:40])


def run_stage5():
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    from datetime import timedelta

    logger.info("阶段5：回测 开始")

    stage_dir = os.path.join(RESULT_DIR, '5.4_投资策略构建')
    os.makedirs(stage_dir, exist_ok=True)

    pred_result_path = os.path.join(RESULT_DIR, '5.3_事件影响预测', '测试集预测结果_完整.csv')
    if os.path.exists(pred_result_path):
        test_df = pd.read_csv(pred_result_path)
        logger.info(f"  加载预测结果: {len(test_df)} 条, {len(test_df.columns)} 列")
    else:
        logger.error(f"  未找到预测结果: {pred_result_path}")
        return False

    price_df = pd.read_csv(os.path.join(DATA_DIR, '股票行情数据', '个股日频行情_清洗.csv'))
    logger.info(f"  行情数据: {len(price_df)} 条")

    if 'event_date' in test_df.columns:
        test_df['event_date_parsed'] = safe_parse_date(test_df['event_date'])
    price_df['trade_date_parsed'] = safe_parse_date(price_df['trade_date'])

    initial_capital = 100000
    max_stocks_per_week = 3
    transaction_cost = 0.003
    stop_loss_threshold = 0.03
    max_per_sub_per_week = 1

    logger.info(f"  初始资金: {initial_capital:,}")
    logger.info(f"  选股策略: LR32维纯技术+综合评分>0.5 + 贝叶斯事件否决 + 追涨防御 + ST过滤")
    logger.info(f"  回测期: 2025-12-08 ~ 2025-12-26 (3周)")
    logger.info(f"  买卖规则: 周二open买入 + 周五close卖出")

    logger.info("  生成交易信号")
    score_col = next((c for c in ['综合评分', '融合_预测置信度'] if c in test_df.columns), None)
    pred_col = next((c for c in ['融合_预测标签', '双塔_预测标签', '事件研究法_预测标签'] if c in test_df.columns), None)

    car_pred_col = '预测CAR' if '预测CAR' in test_df.columns else None

    signals = pd.DataFrame()
    if car_pred_col:
        test_df[car_pred_col] = pd.to_numeric(test_df[car_pred_col], errors='coerce')
        score_col = '融合_预测置信度' if '融合_预测置信度' in test_df.columns else None
        if score_col:
            test_df[score_col] = pd.to_numeric(test_df[score_col], errors='coerce')
            bullish = test_df[test_df[score_col] > 0.5].copy()
            signals = bullish
            logger.info(f"  综合评分>0.5看涨: {len(signals)} 条")
        else:
            bullish = test_df[test_df[car_pred_col] > 0].copy()
            signals = bullish
            logger.info(f"  fallback CAR>0: {len(signals)} 条")

    if len(signals) == 0 and not car_pred_col:
        if pred_col and score_col:
            test_df[pred_col] = pd.to_numeric(test_df[pred_col], errors='coerce')
            test_df[score_col] = pd.to_numeric(test_df[score_col], errors='coerce')
            signals = test_df[test_df[pred_col] == 1].copy()
            logger.info(f"  fallback: LGBM看涨: {len(signals)} 条")
        elif score_col:
            test_df[score_col] = pd.to_numeric(test_df[score_col], errors='coerce')
            signals = test_df[test_df[score_col] >= 0.40].copy()
            logger.info(f"  fallback: 综合评分>=0.40: {len(signals)} 条")
        else:
            logger.warning("  无预测标签")
            return False

    if len(signals) == 0:
        logger.warning("  无看涨信号")
        return False

    BACKTEST_START = pd.Timestamp('2025-12-08')
    BACKTEST_END = pd.Timestamp('2025-12-26')
    date_col = 'event_date_parsed' if 'event_date_parsed' in signals.columns else 'event_date'
    signals[date_col] = pd.to_datetime(signals[date_col], errors='coerce')
    signals = signals.dropna(subset=[date_col])
    signals = signals[(signals[date_col] >= BACKTEST_START) & (signals[date_col] <= BACKTEST_END)]
    logger.info(f"  回测期信号: {len(signals)} 条")

    logger.info("  回测执行")
    signals['iso_week'] = signals[date_col].dt.isocalendar().week.astype(int)
    signals['iso_year'] = signals[date_col].dt.isocalendar().year.astype(int)

    price_df_sorted = price_df.sort_values('trade_date_parsed')
    price_by_code = {}
    for code, group in price_df_sorted.groupby('ts_code'):
        price_by_code[code] = group.set_index('trade_date_parsed')

    capital = initial_capital
    weekly_results = []
    all_trades = []

    for year, week in sorted(signals.groupby(['iso_year', 'iso_week']).groups.keys()):
        week_signals = signals[(signals['iso_year'] == year) & (signals['iso_week'] == week)]
        if len(week_signals) == 0:
            continue
        logger.info(f"  {year}年第{week}周 ({len(week_signals)}条信号)")

        if car_pred_col and car_pred_col in week_signals.columns:
            if '融合_预测置信度' in week_signals.columns:
                week_signals = week_signals.sort_values('融合_预测置信度', ascending=False)
            else:
                week_signals = week_signals.sort_values(car_pred_col, ascending=False)
        elif score_col:
            week_signals = week_signals.sort_values(score_col, ascending=False)
            code_col_bt = 'ts_code' if 'ts_code' in week_signals.columns else '股票代码'
            week_signals = week_signals.drop_duplicates(subset=[code_col_bt], keep='first')

        # 追涨防御
        if 'ret_20d' in week_signals.columns:
            before_count = len(week_signals)
            week_signals = week_signals[week_signals['ret_20d'].fillna(0) < 0.30]
            filtered = before_count - len(week_signals)
            if filtered > 0:
                logger.info(f"    追涨防御: 过滤{filtered}只近期涨幅>30%股票")

        code_col = 'ts_code' if 'ts_code' in week_signals.columns else '股票代码'
        title_col_bt = next((c for c in ['title', '标题_清洗后', '标题', 'event_title'] if c in week_signals.columns), None)
        before_st = len(week_signals)
        if title_col_bt:
            st_mask = week_signals[title_col_bt].fillna('').str.contains(r'\*?\s*ST', case=False, regex=True, na=False)
            week_signals = week_signals[~st_mask]
        if code_col in week_signals.columns and 'name' in price_df.columns:
            st_codes = set(price_df[price_df['name'].str.contains(r'\*?\s*ST', case=False, na=False)]['ts_code'].unique())
            week_signals = week_signals[~week_signals[code_col].isin(st_codes)]
        st_filtered = before_st - len(week_signals)
        if st_filtered > 0:
            logger.info(f"    ST股过滤: 排除{st_filtered}只ST/*ST股票")

        selected = []
        seen_subs = set()
        seen_codes = set()
        for _, sig in week_signals.iterrows():
            if len(selected) >= max_stocks_per_week:
                break
            sub = sig.get('event_sub', '其他')
            code = sig.get('ts_code', sig.get('股票代码', ''))
            if sub in seen_subs and max_per_sub_per_week <= 1:
                continue
            if code in seen_codes:
                continue
            selected.append(sig)
            seen_subs.add(sub)
            seen_codes.add(code)

        if len(selected) == 0:
            logger.info("    无符合条件股票，本周空仓")
            weekly_results.append({'年': year, '周': week, '买入数': 0, '周收益': 0, '累计资金': capital})
            continue

        tradeable = []
        for sig in selected:
            code = sig.get('ts_code', sig.get('股票代码', ''))
            if not code or code not in price_by_code:
                continue
            stock_prices = price_by_code[code]
            iso_cal = stock_prices.index.isocalendar()
            wk_dates = stock_prices[(iso_cal.year == year) & (iso_cal.week == week)]
            if len(wk_dates) == 0:
                continue
            tues = wk_dates[wk_dates.index.dayofweek == 1]
            buy_price = tues.iloc[0].get('open', tues.iloc[0].get('close', np.nan)) if len(tues) > 0 else wk_dates.iloc[0].get('open', np.nan)
            fri = wk_dates[wk_dates.index.dayofweek == 4]
            sell_price = fri.iloc[-1].get('close', np.nan) if len(fri) > 0 else wk_dates.iloc[-1].get('close', np.nan)
            if pd.isna(buy_price) or pd.isna(sell_price) or buy_price <= 0:
                continue
            ret = (sell_price - buy_price) / buy_price
            ret_after_cost = ret - transaction_cost * 2
            tradeable.append((sig, buy_price, sell_price, ret, ret_after_cost))

        if len(tradeable) == 0:
            weekly_results.append({'年': year, '周': week, '买入数': 0, '周收益': 0, '累计资金': capital})
            continue

        # 资金分配：按预测CAR强度加权
        if car_pred_col:
            tradeable_scores = [max(0.01, float(sig.get(car_pred_col, 0.5))) for sig, _, _, _, _ in tradeable]
        else:
            tradeable_scores = [max(0.01, float(sig.get(score_col, 0.5))) for sig, _, _, _, _ in tradeable]
        score_sum = sum(tradeable_scores)
        raw_ratios = [s / score_sum for s in tradeable_scores]
        ratios = [max(0.20, min(0.40, r)) for r in raw_ratios]
        ratio_sum = sum(ratios)
        ratios = [r / ratio_sum for r in ratios]

        week_return = 0
        for idx, (sig, buy_price, sell_price, ret, ret_after_cost) in enumerate(tradeable):
            stock_capital = capital * ratios[idx]
            trade_profit = stock_capital * ret_after_cost
            week_return += ratios[idx] * ret_after_cost
            code = sig.get('ts_code', sig.get('股票代码', ''))
            all_trades.append({'年': year, '周': week, '股票代码': code,
                               '买入价': round(buy_price, 2), '卖出价': round(sell_price, 2),
                               '收益率': f"{ret * 100:.2f}%", '扣费收益': f"{ret_after_cost * 100:.2f}%",
                               '交易利润': round(trade_profit, 2),
                               '事件名称': refine_event_name(str(sig.get('标题_清洗后', sig.get('title', sig.get('event_sub', '')))), sig.get('event_sub', '')),
                               '事件子类': sig.get('event_sub', ''),
                               '预测CAR': round(float(sig.get(car_pred_col, sig.get(score_col, 0))), 4) if car_pred_col else round(sig.get(score_col, 0), 3),
                               '综合评分': round(float(sig.get('综合评分', sig.get('融合_预测置信度', 0))), 4),
                               })
            logger.info(f"    {code}: 买{buy_price:.2f}->卖{sell_price:.2f}, "
                        f"收益{ret * 100:.2f}%, 扣费{ret_after_cost * 100:.2f}%, 比例{ratios[idx] * 100:.1f}%")

        if week_return < -stop_loss_threshold:
            week_return = -stop_loss_threshold
        capital *= (1 + week_return)
        weekly_results.append({'年': year, '周': week, '买入数': len(tradeable),
                               '周收益': f"{week_return * 100:.2f}%", '累计资金': round(capital, 2)})
        logger.info(f"    周收益: {week_return * 100:.2f}%, 累计资金: {capital:,.2f}")

    logger.info("  回测结果汇总")
    total_return = (capital - initial_capital) / initial_capital
    logger.info(f"  初始资金: {initial_capital:,}")
    logger.info(f"  期末资金: {capital:,.2f}")
    logger.info(f"  累计收益率: {total_return * 100:.2f}%")

    bench_path = os.path.join(DATA_DIR, '股票行情数据', '沪深300指数_清洗.csv')
    if os.path.exists(bench_path):
        bdf = pd.read_csv(bench_path)
        bdf['td'] = safe_parse_date(bdf['trade_date'])
        bdf = bdf.dropna(subset=['td'])
        bp = bdf[(bdf['td'] >= BACKTEST_START) & (bdf['td'] <= BACKTEST_END)]
        if 'mkt_ret' in bp.columns and len(bp) > 0:
            br = bp['mkt_ret'].sum()
            logger.info(f"  沪深300基准: {br * 100:.2f}%")
            logger.info(f"  超额收益: {(total_return - br) * 100:.2f}%")

    pd.DataFrame(all_trades).to_csv(os.path.join(stage_dir, '交易记录.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(weekly_results).to_csv(os.path.join(stage_dir, '周度收益.csv'), index=False, encoding='utf-8-sig')

    weekly_trade_scores = defaultdict(list)
    for trade in all_trades:
        sv = float(trade.get('预测CAR', 0.5)) if str(trade.get('预测CAR', '')) not in ['', 'N/A'] else 0.5
        weekly_trade_scores[(trade['年'], trade['周'])].append((trade['股票代码'], sv))

    weekly_ratios_map = {}
    for (yr, wk), cs in weekly_trade_scores.items():
        scores = [max(0.01, s) for _, s in cs]
        ss = sum(scores)
        raw = [s / ss for s in scores]
        ratios = [max(0.20, min(0.40, r)) for r in raw]
        rs = sum(ratios)
        ratios = [round(r / rs, 4) for r in ratios]
        for i, (code, _) in enumerate(cs):
            weekly_ratios_map[(yr, wk, code)] = ratios[i]

    seen_wks = []
    for t in all_trades:
        wk = (t['年'], t['周'])
        if wk not in seen_wks:
            seen_wks.append(wk)
    week_date_map = {}
    for idx, (yr, wk) in enumerate(seen_wks):
        try:
            wm = pd.Timestamp.fromisocalendar(yr, wk, 1)
            wf = wm + timedelta(days=4)
            week_date_map[(yr, wk)] = f"第{idx + 1}周({wm.strftime('%m.%d')}-{wf.strftime('%m.%d')})"
        except:
            week_date_map[(yr, wk)] = f"第{idx + 1}周"

    result_rows = []
    for t in all_trades:
        cr = t['股票代码']
        cp = cr.split('.')[0] if '.' in str(cr) else str(cr)
        cp = cp.zfill(6)
        tr = weekly_ratios_map.get((t['年'], t['周'], t['股票代码']), round(1 / max_stocks_per_week, 4))
        score_val = t.get('综合评分', 'N/A')
        if isinstance(score_val, float):
            score_str = f"{score_val:.4f}"
        else:
            score_str = str(score_val)
        car_val = t.get('预测CAR', 'N/A')
        if isinstance(car_val, float):
            car_str = f"CAR={car_val:+.2f}%"
        else:
            car_str = ""
        result_rows.append({
            '周': week_date_map.get((t['年'], t['周']), '?'), '事件名称': t.get('事件名称', ''),
            '标的股票代码': cp, '资金比例': tr,
            '选股依据': f"综合评分={score_str} {car_str}".strip(),
        })

    if result_rows:
        rdf = pd.DataFrame(result_rows)
        try:
            from openpyxl import Workbook
            wb = Workbook()
            ws = wb.active
            ws.append(['周', '事件名称', '标的股票代码', '资金比例', '选股依据'])
            for ri, (_, r) in enumerate(rdf.iterrows(), start=2):
                ws.cell(row=ri, column=3, value=str(r['标的股票代码'])).number_format = '@'
                ws.cell(row=ri, column=1, value=str(r['周']))
                ws.cell(row=ri, column=2, value=str(r['事件名称']))
                ws.cell(row=ri, column=4, value=float(r['资金比例']))
                ws.cell(row=ri, column=5, value=str(r['选股依据']))
            for c, w in [('A', 20), ('B', 28), ('C', 12), ('D', 10), ('E', 18)]:
                ws.column_dimensions[c].width = w
            wb.save(os.path.join(stage_dir, 'result.xlsx'))
        except ImportError:
            rdf.to_excel(os.path.join(stage_dir, 'result.xlsx'), index=False)
        logger.info(f"  result.xlsx已生成: {len(result_rows)} 条记录")

    logger.info("阶段5完成")
    return True


def main():
    logger.info("训练脚本启动")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"数据目录: {DATA_DIR}")

    logger.info("验证数据完整性...")
    required_files = [
        ('事件数据/事件数据_清洗.csv', '事件数据'),
        ('建模核心表/训练集_增强_v3清洗后.csv', '训练集'),
        ('建模核心表/验证集_增强_v3清洗后.csv', '验证集'),
        ('建模核心表/测试集_增强_v3清洗后.csv', '测试集'),
        ('股票行情数据/个股日频行情_清洗.csv', '行情数据'),
    ]
    missing_files = []
    for path, desc in required_files:
        full_path = os.path.join(DATA_DIR, path)
        exists = os.path.exists(full_path)
        if not exists:
            alt = path.replace('_v3清洗后', '').replace('_v3', '').replace('增强_', '')
            if os.path.exists(os.path.join(DATA_DIR, alt)):
                exists = True
        status = 'OK' if exists else '缺失'
        logger.info(f"  {status} {desc}: {path}")
        if not exists:
            missing_files.append(path)

    if missing_files:
        logger.error("数据文件不完整，请检查 TEDDY_DATA_DIR 或数据集目录。")
        return 1

    stages = [
        ('阶段2：事件识别与分类（AHP + TF-IDF+LR）', run_stage2),
        ('阶段3：关联公司挖掘（TF-IDF + 相似度）', run_stage3),
        ('阶段4：影响预测（LGBM回归 + 交互 + 排名 + 贝叶斯）', run_stage4),
        ('阶段5：回测(LR32维+综合评分驱动)', run_stage5),
    ]

    stage_timings = []
    for i, (name, func) in enumerate(stages):
        logger.info(f"准备执行: {name}")
        t0 = time.time()
        try:
            success = func()
            elapsed = time.time() - t0
            status = '完成' if success else '失败'
            logger.info(f"{name} -- {status} (耗时 {elapsed:.1f}秒)")
            stage_timings.append({'阶段': name, '状态': status, '耗时秒': round(elapsed, 1)})
        except Exception as e:
            elapsed = time.time() - t0
            logger.exception(f"{name} -- 异常 (耗时 {elapsed:.1f}秒): {e}")
            stage_timings.append({'阶段': name, '状态': f'异常: {str(e)[:80]}', '耗时秒': round(elapsed, 1)})

    logger.info("训练全部完成")
    total_time = sum(s['耗时秒'] for s in stage_timings)
    for s in stage_timings:
        logger.info(f"  {s['状态']} | {s['阶段']} | {s['耗时秒']:.1f}s")
    logger.info(f"  总耗时: {total_time:.1f}s ({total_time / 60:.1f}分钟)")
    logger.info(f"  结果目录: {RESULT_DIR}")
    return 0 if all(s['状态'] == '完成' for s in stage_timings) else 1


if __name__ == '__main__':
    sys.exit(main())
