#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import json
import pickle
import scipy.stats as scipy_stats
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get('TEDDY_DATA_DIR') or os.path.join(BASE_DIR, '数据集_清洗')
))
RESULT_DIR = os.path.join(BASE_DIR, '训练结果')

STAGE4_DIR = os.path.join(RESULT_DIR, '5.3_事件影响预测')
OUTPUT_DIR = os.path.join(RESULT_DIR, '5.4_投资策略构建_扩展')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 工具函数
def safe_parse_date(series_or_val):
    if isinstance(series_or_val, pd.Series):
        try:
            result = pd.to_datetime(series_or_val.astype(int).astype(str), format='%Y%m%d', errors='coerce')
            if result.notna().sum() > len(result) * 0.5:
                return result
        except:
            pass
        result = pd.to_datetime(series_or_val.astype(str), errors='coerce', format='mixed')
        mask_1970 = result.dt.year == 1970
        if mask_1970.any():
            try:
                result[mask_1970] = pd.to_datetime(
                    series_or_val[mask_1970].astype(int).astype(str),
                    format='%Y%m%d', errors='coerce'
                )
            except:
                pass
        return result
    else:
        return pd.to_datetime(str(series_or_val).strip(), errors='coerce')

def calc_max_drawdown(nav_series):
    if len(nav_series) < 2:
        return 0.0, None, None
    cumulative_max = nav_series.cummax()
    drawdown = (nav_series - cumulative_max) / cumulative_max
    max_dd = drawdown.min()
    if max_dd >= 0:
        return 0.0, None, None
    trough_idx = drawdown.idxmin()
    before_trough = nav_series[:trough_idx]
    if len(before_trough) == 0:
        peak_idx = nav_series.index[0]
    else:
        peak_idx = before_trough.idxmax()
    return max_dd, peak_idx, trough_idx

def calc_sharpe_ratio(weekly_returns, risk_free_rate_annual=0.015):
    if len(weekly_returns) < 2:
        return 0.0
    rf_weekly = (1 + risk_free_rate_annual) ** (1/52) - 1
    excess_returns = weekly_returns - rf_weekly
    if excess_returns.std() < 1e-10:
        return 0.0
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(52)
    return sharpe

def calc_calmar_ratio(annual_return, max_drawdown):
    if abs(max_drawdown) < 1e-10:
        return 0.0
    return annual_return / abs(max_drawdown)

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
    if len(result) > 40:
        result = result[:39] + '…'
    return result if result else str(event_sub)[:40]

def main():
    print("扩展回测脚本 -- 事件驱动型股市投资策略")
    print("综合评分(0.60LR+0.20贝叶斯+0.20CAR) + 多期回测 + 风险指标")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 加载数据
    print("\n[1/5] 加载数据...")

    train_path = os.path.join(DATA_DIR, '建模核心表', '训练集_增强_v3清洗后.csv')
    val_path = os.path.join(DATA_DIR, '建模核心表', '验证集_增强_v3清洗后.csv')
    test_path = os.path.join(DATA_DIR, '建模核心表', '测试集_增强_v3清洗后.csv')

    if not os.path.exists(train_path):
        train_path = os.path.join(DATA_DIR, '建模核心表', '训练集_增强.csv')
        val_path = os.path.join(DATA_DIR, '建模核心表', '验证集_增强.csv')
        test_path = os.path.join(DATA_DIR, '建模核心表', '测试集_增强.csv')

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"  数据集: {os.path.basename(train_path)}")

    for df in [train_df, val_df, test_df]:
        df['event_date_parsed'] = safe_parse_date(df['event_date'])

    print(f"  训练集: {len(train_df)} 条 ({train_df['event_date_parsed'].min().strftime('%Y-%m')} ~ {train_df['event_date_parsed'].max().strftime('%Y-%m')})")
    print(f"  验证集: {len(val_df)} 条 ({val_df['event_date_parsed'].min().strftime('%Y-%m')} ~ {val_df['event_date_parsed'].max().strftime('%Y-%m')})")

    price_df = pd.read_csv(os.path.join(DATA_DIR, '股票行情数据', '个股日频行情_清洗.csv'))
    price_df['trade_date_parsed'] = safe_parse_date(price_df['trade_date'])
    print(f"  行情数据: {len(price_df)} 条")

    bench_path = os.path.join(DATA_DIR, '股票行情数据', '沪深300指数_清洗.csv')
    if os.path.exists(bench_path):
        bench_df = pd.read_csv(bench_path)
        bench_df['trade_date_parsed'] = safe_parse_date(bench_df['trade_date'])
        bench_df = bench_df.dropna(subset=['trade_date_parsed']).sort_values('trade_date_parsed')
        if 'mkt_ret' in bench_df.columns:
            bench_df['mkt_ret'] = pd.to_numeric(bench_df['mkt_ret'], errors='coerce')
        print(f"  沪深300: {len(bench_df)} 条")
    else:
        bench_df = None
        print("  沪深300数据不存在")

    # 2. 加载模型（E4 LR分类 + LGBM回归 + 贝叶斯先验）
    import joblib
    print(f"\n[2/5] 加载模型...")

    from sklearn.linear_model import LogisticRegression

    # 技术因子列（32维纯技术因子）
    tech_factor_cols = [
        'MA5', 'MA10', 'MA20', 'MA60', 'dev_MA5', 'dev_MA10', 'dev_MA20', 'dev_MA60',
        'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'volatility_20d', 'RSI14',
        'MACD_dif', 'MACD_dea', 'MACD_hist', 'BOLL_upper', 'BOLL_lower', 'BOLL_width',
        'BOLL_pos', 'vol_ma5', 'vol_ratio', 'ATR14', 'ATR14_pct',
        'gap_up', 'gap_down', 'gap_pct', 'alpha', 'beta', 'decline_days', 'close_t1'
    ]

    # 2.1 加载E4 LR分类模型
    e4_lr_path = os.path.join(STAGE4_DIR, 'e4_lr_model.pkl')
    e4_lr_alt_path = os.path.join(STAGE4_DIR, 'v15_e4_lr_model.pkl')
    tech_cols_path = os.path.join(STAGE4_DIR, 'tech_factor_cols.json')
    metadata_path = os.path.join(STAGE4_DIR, 'feature_metadata.json')
    legacy_metadata_path = os.path.join(STAGE4_DIR, 'v15_feature_metadata.json')

    available_tech_cols = [c for c in tech_factor_cols if c in train_df.columns]
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        available_tech_cols = metadata.get('tech_cols', available_tech_cols)
        print(f"  加载特征元数据: {len(available_tech_cols)}个技术因子")
    elif os.path.exists(legacy_metadata_path):
        with open(legacy_metadata_path, 'r', encoding='gbk') as f:
            metadata = json.load(f)
        available_tech_cols = metadata.get('tech_cols', available_tech_cols)
        print(f"  加载旧版特征元数据: {len(available_tech_cols)}个技术因子")
    elif os.path.exists(tech_cols_path):
        with open(tech_cols_path, 'r', encoding='utf-8') as f:
            available_tech_cols = json.load(f)
        print(f"  加载技术因子列名: {len(available_tech_cols)}个（来自训练脚本）")
    else:
        print(f"  可用技术因子: {len(available_tech_cols)}/{len(tech_factor_cols)}")

    tech_clf = None
    lr_scaler = None
    for p in [e4_lr_path, e4_lr_alt_path]:
        if os.path.exists(p):
            try:
                tech_clf = joblib.load(p)
                print(f"  加载E4 LR模型: {p}")
                base = p.replace('.pkl', '')
                for sp_try in [base + '_scaler.pkl', base.replace('_model', '') + '_scaler.pkl']:
                    if os.path.exists(sp_try):
                        lr_scaler = joblib.load(sp_try)
                        print(f"  加载LR配套Scaler: {sp_try}")
                        break
                break
            except Exception as e:
                print(f"  LR模型加载失败({p}): {e}")

    # 2.2 加载LGBM回归模型（预测连续CAR）
    lgbm_reg = None
    lgbm_scaler = None
    lgbm_paths_to_try = [
        os.path.join(STAGE4_DIR, 'e4_lgbm_model.txt'),
        os.path.join(STAGE4_DIR, 'v15_e4_lgbm_model.txt'),
        os.path.join(STAGE4_DIR, 'lgbm_regression_best.pkl'),
        os.path.join(STAGE4_DIR, 'lgbm_reg.pkl'),
    ]
    for lgbm_reg_path in lgbm_paths_to_try:
        if os.path.exists(lgbm_reg_path):
            try:
                import lightgbm as lgb
                if lgbm_reg_path.endswith('.txt'):
                    with open(lgbm_reg_path, 'r', encoding='utf-8') as model_file:
                        lgbm_reg = lgb.Booster(model_str=model_file.read())
                else:
                    lgbm_reg = joblib.load(lgbm_reg_path)
                print(f"  加载LGBM回归模型: {lgbm_reg_path}")
                break
            except Exception as e:
                print(f"  LGBM回归模型加载失败({lgbm_reg_path}): {e}")
    if lgbm_reg is None:
        print("  未找到LGBM回归模型，CAR预测概率将使用0.5默认值")

    scaler_candidates = [
        os.path.join(STAGE4_DIR, 'feature_scaler.pkl'),
        os.path.join(STAGE4_DIR, 'v15_feature_scaler.pkl'),
    ]
    scaler_path = next((p for p in scaler_candidates if os.path.exists(p)), None)
    if scaler_path:
        try:
            lgbm_scaler = joblib.load(scaler_path)
            print(f"  加载特征Scaler: {scaler_path}")
        except Exception as e:
            print(f"  特征Scaler加载失败: {e}")

    # 2.3 加载贝叶斯事件先验（James-Stein收缩）
    bayesian_priors = {}
    bayes_path = os.path.join(STAGE4_DIR, '贝叶斯事件先验.csv')
    if not os.path.exists(bayes_path):
        bayes_path = os.path.join(STAGE4_DIR, '贝叶斯事件先验_v15.csv')
    if os.path.exists(bayes_path):
        bayes_df = pd.read_csv(bayes_path)
        for _, row in bayes_df.iterrows():
            bayesian_priors[row['event_sub']] = row.to_dict()
        print(f"  加载贝叶斯事件先验: {len(bayesian_priors)} 个子类")
    else:
        print("  未找到贝叶斯事件先验文件，回退到事件研究法统计")

    car_col = None
    for col in ['CAR_post_rec', 'CAR_post_new', 'CAR_post', 'CAR']:
        if col in train_df.columns:
            car_col = col
            break
    if car_col is None:
        print("  未找到任何CAR列，使用label列")
        car_col = 'label'
    print(f"  CAR列: {car_col}")

    for df in [train_df, val_df, test_df]:
        df[car_col] = pd.to_numeric(df[car_col], errors='coerce')
        df['label_v12'] = df[car_col].apply(
            lambda x: 1 if x > 2 else (-1 if x < -2 else 0)
        )

    label_encode = {-1: 0, 0: 1, 1: 2}
    label_decode = {0: -1, 1: 0, 2: 1}

    train_df['label_encoded'] = train_df['label_v12'].map(label_encode)
    val_df['label_encoded'] = val_df['label_v12'].map(label_encode)
    test_df['label_encoded'] = test_df['label_v12'].map(label_encode)

    print(f"  训练集label分布: {train_df['label_v12'].value_counts().to_dict()}")

    for df in [train_df, val_df, test_df]:
        for col in available_tech_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    train_tech = train_df[available_tech_cols].fillna(0).values
    val_tech = val_df[available_tech_cols].fillna(0).values
    test_tech = test_df[available_tech_cols].fillna(0).values

    # 训练脚本中LR的输入经过了双重标准化:
    # 1) build_lgbm_features内部的StandardScaler (v15_feature_scaler.pkl)
    # 2) LR专用的StandardScaler (v15_e4_lr_scaler.pkl)
    # 回测时必须同样做两层，否则特征尺度不匹配
    if lgbm_scaler is not None and lr_scaler is not None:
        train_tech = lgbm_scaler.transform(train_tech)
        val_tech = lgbm_scaler.transform(val_tech)
        test_tech = lgbm_scaler.transform(test_tech)
        print(f"  第一层标准化完成（feature_scaler）")
    elif lr_scaler is not None:
        print(f"  无feature_scaler，跳过第一层标准化（精度可能下降）")

    train_labels = train_df['label_encoded'].values
    val_labels = val_df['label_encoded'].values
    test_labels = test_df['label_encoded'].values

    # E4 LR预测
    if tech_clf is not None and lr_scaler is not None:
        train_preds = tech_clf.predict(lr_scaler.transform(train_tech))
        val_preds = tech_clf.predict(lr_scaler.transform(val_tech))
        test_preds = tech_clf.predict(lr_scaler.transform(test_tech))
        train_probs = tech_clf.predict_proba(lr_scaler.transform(train_tech))
        val_probs = tech_clf.predict_proba(lr_scaler.transform(val_tech))
        test_probs = tech_clf.predict_proba(lr_scaler.transform(test_tech))
        print(f"  E4 LR预测完成（已加载模型+Scaler）")
    elif tech_clf is not None:
        train_preds = tech_clf.predict(train_tech)
        val_preds = tech_clf.predict(val_tech)
        test_preds = tech_clf.predict(test_tech)
        train_probs = tech_clf.predict_proba(train_tech)
        val_probs = tech_clf.predict_proba(val_tech)
        test_probs = tech_clf.predict_proba(test_tech)
        print(f"  E4 LR预测完成（已加载模型，无Scaler）")
    else:
        from sklearn.preprocessing import StandardScaler as LRScaler
        lr_scaler = LRScaler()
        train_tech_scaled = lr_scaler.fit_transform(train_tech)
        val_tech_scaled = lr_scaler.transform(val_tech)
        test_tech_scaled = lr_scaler.transform(test_tech)
        tech_clf = LogisticRegression(C=0.5, class_weight='balanced', random_state=42, max_iter=500, solver='lbfgs')
        tech_clf.fit(train_tech_scaled, train_labels)
        train_preds = tech_clf.predict(train_tech_scaled)
        val_preds = tech_clf.predict(val_tech_scaled)
        test_preds = tech_clf.predict(test_tech_scaled)
        train_probs = tech_clf.predict_proba(train_tech_scaled)
        val_probs = tech_clf.predict_proba(val_tech_scaled)
        test_probs = tech_clf.predict_proba(test_tech_scaled)
        print(f"  未找到LR模型，重新训练（C=0.5 + StandardScaler）")

    from sklearn.metrics import f1_score

    train_f1 = f1_score(train_labels, train_preds, average='weighted')
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    test_f1 = f1_score(test_labels, test_preds, average='weighted')

    print(f"  E4 LR训练集F1: {train_f1:.4f}")
    print(f"  E4 LR验证集F1: {val_f1:.4f}")
    print(f"  LR系数Top-5:")
    coef_df = pd.DataFrame({'因子': available_tech_cols, '系数': tech_clf.coef_[2]})
    coef_df['abs_coef'] = coef_df['系数'].abs()
    print(coef_df.sort_values('abs_coef', ascending=False).head(5)[['因子', '系数']].to_string(index=False))

    # 确定事件子类列名
    event_sub_col = None
    for c in ['event_sub', 'event_type', '事件子类']:
        if c in train_df.columns:
            event_sub_col = c
            break

    # LGBM回归预测CAR
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    for df_name, df, tech in [('验证集', val_df, val_tech), ('测试集', test_df, test_tech)]:
        df['LR看涨概率'] = None
        df['CAR预测概率'] = 0.5
        df['预测CAR'] = 0.0

    for df, probs in [(val_df, val_probs), (test_df, test_probs)]:
        df['LR看涨概率'] = probs[:, 2]

    # LGBM回归：预测CAR
    lgbm_metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            lgbm_metadata = json.load(f)

    if lgbm_reg is not None and lgbm_metadata is not None:
        # 完整特征工程：技术因子 + 事件one-hot + 市场状态 + 交互 + 排名
        try:
            from sklearn.preprocessing import LabelEncoder
            n_tech = len(available_tech_cols)
            event_sub_classes = lgbm_metadata.get('event_sub_classes', [])
            regime_classes = lgbm_metadata.get('regime_classes', [])
            state_classes = lgbm_metadata.get('state_classes', [])
            full_feature_names = lgbm_metadata.get('feature_names', [])

            event_sub_encoder = LabelEncoder()
            event_sub_encoder.classes_ = np.array(event_sub_classes) if event_sub_classes else np.array(['其他'])
            regime_encoder = LabelEncoder()
            regime_encoder.classes_ = np.array(regime_classes) if regime_classes else np.array(['震荡市'])
            state_encoder = LabelEncoder()
            state_encoder.classes_ = np.array(state_classes) if state_classes else np.array(['中性'])

            for df_name, df in [('验证集', val_df), ('测试集', test_df)]:
                tech_data = df[available_tech_cols].fillna(0).values.astype(np.float32)

                bayes_feat = np.zeros((len(df), 5), dtype=np.float32)
                for i, row in df.iterrows():
                    sub = row.get(event_sub_col, row.get('event_sub', '其他')) if event_sub_col else row.get('event_sub', '其他')
                    if sub is None: sub = '其他'
                    bp = bayesian_priors.get(sub, {})
                    bayes_feat[i, 0] = bp.get('post_mean_car', 0.0)
                    bayes_feat[i, 1] = bp.get('post_win_rate', 0.333)
                    bayes_feat[i, 2] = bp.get('shrinkage_factor', 0.5)
                    bayes_feat[i, 3] = bp.get('info_ratio', 0.0)
                    bayes_feat[i, 4] = 1.0 if bp.get('is_positive', False) else 0.0

                sub_col = event_sub_col if event_sub_col else 'event_sub'
                subs = df[sub_col].fillna('其他').values
                try:
                    sub_encoded = event_sub_encoder.transform(subs)
                except:
                    sub_encoded = np.zeros(len(df), dtype=int)
                n_subs = len(event_sub_classes) if event_sub_classes else 1
                sub_onehot = np.zeros((len(df), n_subs), dtype=np.float32)
                for i, c in enumerate(sub_encoded):
                    if c < n_subs:
                        sub_onehot[i, c] = 1.0

                n_regime = len(regime_classes) if regime_classes else 1
                n_state = len(state_classes) if state_classes else 1
                regime_onehot = np.zeros((len(df), n_regime), dtype=np.float32)
                state_onehot = np.zeros((len(df), n_state), dtype=np.float32)
                # 用RSI判断市场状态
                rsi_vals = df['RSI14'].fillna(50).values
                for i, rsi in enumerate(rsi_vals):
                    if rsi < 40:
                        regime_onehot[i, 0] = 1.0
                        state_onehot[i, 0] = 1.0
                    elif rsi > 60:
                        regime_onehot[i, min(1, n_regime-1)] = 1.0
                        state_onehot[i, min(1, n_state-1)] = 1.0
                    else:
                        if n_regime > 2:
                            regime_onehot[i, 2] = 1.0
                        else:
                            regime_onehot[i, 0] = 1.0
                        if n_state > 2:
                            state_onehot[i, 2] = 1.0
                        else:
                            state_onehot[i, 0] = 1.0

                X_full = np.hstack([tech_data, bayes_feat, sub_onehot, regime_onehot, state_onehot])

                # 交互特征（技术因子 x 事件，取前几个重要因子）
                n_interact = min(5, n_tech)
                interact_feat = np.zeros((len(df), n_interact * min(3, n_subs)), dtype=np.float32)
                for j in range(n_interact):
                    for k in range(min(3, n_subs)):
                        interact_feat[:, j * min(3, n_subs) + k] = tech_data[:, j] * sub_onehot[:, k]
                X_full = np.hstack([X_full, interact_feat])

                # 排名特征（横截面排名）
                rank_feat = np.zeros((len(df), n_tech), dtype=np.float32)
                for j in range(n_tech):
                    col_vals = tech_data[:, j]
                    if col_vals.std() > 1e-10:
                        rank_feat[:, j] = (col_vals - col_vals.mean()) / (col_vals.std() + 1e-8)
                X_full = np.hstack([X_full, rank_feat])

                # 截断或填充到模型期望的维度
                n_expected = lgbm_reg.num_feature() if hasattr(lgbm_reg, 'num_feature') else len(full_feature_names)
                if X_full.shape[1] > n_expected:
                    X_full = X_full[:, :n_expected]
                elif X_full.shape[1] < n_expected:
                    pad = np.zeros((len(df), n_expected - X_full.shape[1]), dtype=np.float32)
                    X_full = np.hstack([X_full, pad])

                # StandardScaler
                if lgbm_scaler is not None:
                    try:
                        X_full = lgbm_scaler.transform(X_full)
                    except:
                        pass

                try:
                    car_preds = lgbm_reg.predict(X_full)
                    df['预测CAR'] = car_preds
                    df['CAR预测概率'] = sigmoid(car_preds)
                    print(f"  {df_name} LGBM回归CAR: mean={car_preds.mean():+.3f}, "
                          f"sigmoid概率: mean={sigmoid(car_preds).mean():.3f}")
                except Exception as e:
                    print(f"  {df_name} LGBM回归预测失败: {e}")
                    df['预测CAR'] = 0.0
                    df['CAR预测概率'] = 0.5
        except Exception as e:
            print(f"  LGBM完整特征工程失败: {e}，CAR预测使用默认0.5")
            for df in [val_df, test_df]:
                df['CAR预测概率'] = 0.5
                df['预测CAR'] = 0.0
    elif lgbm_reg is not None:
        for df_name, df in [('验证集', val_df), ('测试集', test_df)]:
            tech_data = df[available_tech_cols].fillna(0).values.astype(np.float32)
            bayes_feat = np.zeros((len(df), 5), dtype=np.float32)
            X_lgbm = np.hstack([tech_data, bayes_feat])
            try:
                n_expected = lgbm_reg.num_feature() if hasattr(lgbm_reg, 'num_feature') else X_lgbm.shape[1]
                if X_lgbm.shape[1] != n_expected:
                    if X_lgbm.shape[1] > n_expected:
                        X_lgbm = X_lgbm[:, :n_expected]
                    else:
                        X_lgbm = np.hstack([X_lgbm, np.zeros((len(df), n_expected - X_lgbm.shape[1]))])
                car_preds = lgbm_reg.predict(X_lgbm)
                df['预测CAR'] = car_preds
                df['CAR预测概率'] = sigmoid(car_preds)
                print(f"  {df_name} LGBM回归CAR(simple): mean={car_preds.mean():+.3f}")
            except Exception as e:
                print(f"  {df_name} LGBM回归预测失败: {e}")
    else:
        print("  LGBM回归不可用，CAR预测概率使用默认0.5")

    print("\n[3/5] 计算综合评分...")

    # 3.1 贝叶斯事件得分

    def compute_bayesian_event_score(row):
        sub = row.get(event_sub_col, row.get('event_sub', '其他'))
        if sub is None: sub = '其他'

        if sub in bayesian_priors:
            bp = bayesian_priors[sub]
            post_wr = float(bp.get('post_win_rate', 0.333))
            post_car = float(bp.get('post_mean_car', 0.0))
            info_ratio = float(bp.get('info_ratio', 0.0))
            is_sig = bool(bp.get('significant', False))

            if post_wr >= 0.42:
                score = 0.52 + min(0.25, (post_wr - 0.42) * 3.5)
            elif post_wr >= 0.36:
                score = 0.38 + (post_wr - 0.36) * 2.3
            elif post_wr >= 0.30:
                score = 0.28 + (post_wr - 0.30) * 1.7
            else:
                score = max(0.08, 0.18 + (post_wr - 0.30) * 1.5)

            # CAR方向微调
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

            return round(score, 4)
        else:
            return 0.40

    for df in [val_df, test_df]:
        df['贝叶斯事件得分'] = df.apply(compute_bayesian_event_score, axis=1)

    for df, probs, e4_preds in [(val_df, val_probs, val_preds), (test_df, test_probs, test_preds)]:
        lr_up_prob = probs[:, 2].copy()
        ev_scores = df['贝叶斯事件得分'].values.astype(float)
        car_pred_prob = df['CAR预测概率'].values.astype(float)

        # 综合评分: 0.60xLR + 0.20x贝叶斯 + 0.20xCAR预测
        rank_score = lr_up_prob * 0.60 + ev_scores * 0.20 + car_pred_prob * 0.20

        # E4 LR决定方向
        direction = e4_preds.copy().astype(int)

        # LR-CAR矛盾过滤：LR看涨但CAR预测<-0.5则降级为平
        car_pred_raw = df['预测CAR'].values.astype(float) if '预测CAR' in df.columns else np.zeros(len(df))
        car_negative_mask = (direction == 2) & (car_pred_raw < -0.5)
        direction[car_negative_mask] = 1
        print(f"  LR-CAR矛盾过滤: {car_negative_mask.sum()}条(LR看涨但CAR预测<-0.5)")

        # 动态否决阈值
        ev_mean = ev_scores.mean()
        ev_std = ev_scores.std()
        veto_threshold = max(0.10, ev_mean - 0.5 * ev_std)
        ev_q25 = np.percentile(ev_scores, 25)

        veto_mask = (direction == 2) & (ev_scores < veto_threshold)
        direction[veto_mask] = 1

        # 追涨防御
        rsi_values = df['RSI14'].fillna(50).values if 'RSI14' in df.columns else np.full(len(df), 50.0)
        chase_veto_mask = (direction == 2) & (rsi_values > 70) & (ev_scores < ev_q25)
        direction[chase_veto_mask] = 1

        composite_scores = np.zeros(len(df))
        composite_scores[direction == 2] = rank_score[direction == 2]
        composite_scores[direction == 0] = -rank_score[direction == 0]

        fusion_preds_decoded = [label_decode[d] for d in direction]

        fusion_reasons = []
        for i in range(len(df)):
            d = direction[i]
            ev = ev_scores[i]
            rsi = rsi_values[i]
            if d == 2 and ev < veto_threshold:
                fusion_reasons.append('LR看涨但事件极端利空否决')
            elif d == 2 and rsi > 70 and ev < ev_q25:
                fusion_reasons.append('LR看涨但追涨防御否决')
            elif d == 2 and ev >= 0.50:
                fusion_reasons.append('LR看涨+事件利好确认')
            elif d == 2 and ev >= ev_q25:
                fusion_reasons.append('LR看涨+事件中性')
            elif d == 2:
                fusion_reasons.append('LR看涨+事件偏弱')
            elif d == 0 and ev < ev_q25:
                fusion_reasons.append('LR看跌+事件利空共振')
            elif d == 0:
                fusion_reasons.append('LR看跌')
            else:
                fusion_reasons.append('LR看平')

        df['融合_预测标签'] = fusion_preds_decoded
        df['综合评分'] = composite_scores
        df['融合决策原因'] = fusion_reasons
        df['融合_预测置信度'] = composite_scores

    print(f"  事件否决动态阈值: {veto_threshold:.4f}")
    print(f"  验证集预测看涨: {(val_df['综合评分']>0.5).sum()}/{len(val_df)} ({(val_df['综合评分']>0.5).mean()*100:.1f}%)")
    print(f"  验证集综合评分: mean={val_df['综合评分'].mean():.3f}, median={val_df['综合评分'].median():.3f}")

    print("\n[4/5] 执行回测...")

    # 策略参数
    INITIAL_CAPITAL = 100000
    MAX_STOCKS_PER_WEEK = 3
    TRANSACTION_COST = 0.003
    STOP_LOSS_THRESHOLD = 0.03
    MAX_POSITION_RATIO = 0.40
    MAX_PER_SUB_PER_WEEK = 1

    price_df_sorted = price_df.sort_values('trade_date_parsed')
    price_by_code = {}
    for code, group in price_df_sorted.groupby('ts_code'):
        price_by_code[code] = group.set_index('trade_date_parsed')

    def run_backtest(signals_df, start_date, end_date, label=""):
        print(f"\n  回测: {label} ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})")

        date_col = 'event_date_parsed'
        if date_col not in signals_df.columns:
            signals_df[date_col] = safe_parse_date(signals_df['event_date'])

        signals_df[date_col] = pd.to_datetime(signals_df[date_col], errors='coerce')
        signals_df = signals_df.dropna(subset=[date_col])

        score_col = '综合评分'
        bullish = signals_df[signals_df[score_col] > 0.5].copy()

        bullish = bullish[(bullish[date_col] >= start_date) & (bullish[date_col] <= end_date)]
        print(f"    看涨信号(评分>0.5): {len(bullish)} 条")

        if len(bullish) == 0:
            print("    无看涨信号")
            return None

        bullish['iso_week'] = bullish[date_col].dt.isocalendar().week.astype(int)
        bullish['iso_year'] = bullish[date_col].dt.isocalendar().year.astype(int)

        capital = INITIAL_CAPITAL
        weekly_results = []
        all_trades = []
        nav_history = [INITIAL_CAPITAL]

        for year, week in sorted(bullish.groupby(['iso_year', 'iso_week']).groups.keys()):
            week_signals = bullish[(bullish['iso_year'] == year) & (bullish['iso_week'] == week)]
            if len(week_signals) == 0:
                continue

            if score_col in week_signals.columns:
                week_signals = week_signals.sort_values(score_col, ascending=False)

            # 追涨防御：过滤近期涨幅过大的股票
            if 'ret_20d' in week_signals.columns:
                before_count = len(week_signals)
                week_signals = week_signals[
                    week_signals['ret_20d'].fillna(0) < 0.30
                ]
                filtered_count = before_count - len(week_signals)
                if filtered_count > 0:
                    print(f"    追涨防御: 过滤{filtered_count}只近期涨幅>30%股票")

            # ST股过滤：排除*ST/ST股票
            code_col = 'ts_code' if 'ts_code' in week_signals.columns else '股票代码'
            title_col_bt = None
            for col in ['title', '标题_清洗后', '标题', 'event_title']:
                if col in week_signals.columns:
                    title_col_bt = col
                    break
            before_st_count = len(week_signals)
            if title_col_bt:
                st_mask = week_signals[title_col_bt].fillna('').str.contains(r'\*?\s*ST', case=False, regex=True, na=False)
                week_signals = week_signals[~st_mask]
            # 从行情数据中检查股票名称含ST
            if code_col in week_signals.columns and 'name' in price_df.columns:
                st_codes = set(price_df[price_df['name'].str.contains(r'\*?\s*ST', case=False, na=False)]['ts_code'].unique())
                week_signals = week_signals[~week_signals[code_col].isin(st_codes)]
            st_filtered = before_st_count - len(week_signals)
            if st_filtered > 0:
                print(f"    ST股过滤: 排除{st_filtered}只ST/*ST股票")

            week_signals = week_signals.drop_duplicates(subset=[code_col], keep='first')

            # 分层选股：同类event_sub最多1只 + 同一股票最多1只
            selected = []
            seen_subs = set()
            seen_codes = set()
            for _, sig in week_signals.iterrows():
                sub = sig.get(event_sub_col, sig.get('event_sub', '其他'))
                if sub is None:
                    sub = '其他'
                code = sig.get('ts_code', sig.get('股票代码', ''))
                if len(selected) >= MAX_STOCKS_PER_WEEK:
                    break
                if sub in seen_subs and MAX_PER_SUB_PER_WEEK <= 1:
                    continue
                if code in seen_codes:
                    continue
                selected.append(sig)
                seen_subs.add(sub)
                seen_codes.add(code)

            if len(selected) == 0:
                weekly_results.append({'年': year, '周': week, '买入数': 0, '周收益': 0, '累计资金': capital})
                nav_history.append(capital)
                continue

            # 按评分加权资金分配
            tradeable = []
            for sig in selected:
                code = sig.get('ts_code', sig.get('股票代码', ''))
                if not code or code not in price_by_code:
                    continue

                stock_prices = price_by_code[code]

                iso_cal = stock_prices.index.isocalendar()
                week_dates = stock_prices[
                    (iso_cal.year == year) &
                    (iso_cal.week == week)
                ]

                if len(week_dates) == 0:
                    continue

                tues_days = week_dates[week_dates.index.dayofweek == 1]
                if len(tues_days) > 0:
                    buy_price = tues_days.iloc[0].get('open', tues_days.iloc[0].get('close', np.nan))
                else:
                    buy_price = week_dates.iloc[0].get('open', week_dates.iloc[0].get('close', np.nan))

                fri_days = week_dates[week_dates.index.dayofweek == 4]
                if len(fri_days) > 0:
                    sell_price = fri_days.iloc[-1].get('close', np.nan)
                else:
                    sell_price = week_dates.iloc[-1].get('close', np.nan)

                if pd.isna(buy_price) or pd.isna(sell_price) or buy_price <= 0:
                    continue

                ret = (sell_price - buy_price) / buy_price
                ret_after_cost = ret - TRANSACTION_COST * 2
                tradeable.append((sig, buy_price, sell_price, ret, ret_after_cost))

            if len(tradeable) == 0:
                weekly_results.append({'年': year, '周': week, '买入数': 0, '周收益': 0, '累计资金': capital})
                nav_history.append(capital)
                continue

            tradeable_scores = []
            for sig, _, _, _, _ in tradeable:
                s = sig.get(score_col, 0.5) if score_col else 0.5
                tradeable_scores.append(max(0.01, float(s)))
            score_sum = sum(tradeable_scores)
            raw_ratios = [s / score_sum for s in tradeable_scores]
            ratios = [max(0.20, min(0.40, r)) for r in raw_ratios]
            ratio_sum = sum(ratios)
            ratios = [r / ratio_sum for r in ratios]

            week_return = 0
            position_value = capital

            for idx, (sig, buy_price, sell_price, ret, ret_after_cost) in enumerate(tradeable):
                stock_capital = position_value * ratios[idx]
                trade_profit = stock_capital * ret_after_cost
                week_return += ratios[idx] * ret_after_cost

                code = sig.get('ts_code', sig.get('股票代码', ''))
                all_trades.append({
                    '年': year, '周': week, '股票代码': code,
                    '买入价': round(buy_price, 2), '卖出价': round(sell_price, 2),
                    '收益率': f"{ret*100:.2f}%", '扣费收益': f"{ret_after_cost*100:.2f}%",
                    '交易利润': round(trade_profit, 2),
                    '事件名称': refine_event_name(str(sig.get('标题_清洗后', sig.get('title', sig.get('event_sub', '')))), sig.get(event_sub_col, sig.get('event_sub', ''))),
                    '事件子类': sig.get(event_sub_col, sig.get('event_sub', '')),
                    '综合评分': round(sig.get(score_col, 0), 3),
                    '资金比例': round(ratios[idx], 4),
                })

            if week_return < -STOP_LOSS_THRESHOLD:
                week_return = -STOP_LOSS_THRESHOLD

            capital *= (1 + week_return)
            nav_history.append(capital)
            weekly_results.append({
                '年': year, '周': week, '买入数': len(tradeable),
                '周收益': f"{week_return*100:.2f}%", '累计资金': round(capital, 2)
            })

            print(f"    {year}年W{week:02d}: 买入{len(selected)}只, 周收益{week_return*100:.2f}%, 累计{capital:,.0f}")

        if not weekly_results:
            print("    无交易记录")
            return None

        nav_series = pd.Series(nav_history)
        max_dd, peak_idx, trough_idx = calc_max_drawdown(nav_series)

        weekly_rets = []
        for wr in weekly_results:
            r = float(str(wr['周收益']).replace('%', '')) / 100
            weekly_rets.append(r)
        weekly_rets = np.array(weekly_rets)

        total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

        n_weeks = len(weekly_rets)
        if n_weeks > 0:
            annual_return = (1 + total_return) ** (52 / n_weeks) - 1
        else:
            annual_return = 0

        sharpe = calc_sharpe_ratio(weekly_rets)

        calmar = calc_calmar_ratio(annual_return, max_dd)

        win_weeks = (weekly_rets > 0).sum()
        week_win_rate = win_weeks / len(weekly_rets) if len(weekly_rets) > 0 else 0

        bench_return = 0
        if bench_df is not None and 'mkt_ret' in bench_df.columns:
            bench_period = bench_df[
                (bench_df['trade_date_parsed'] >= start_date) &
                (bench_df['trade_date_parsed'] <= end_date)
            ]
            if len(bench_period) > 0:
                bench_return = bench_period['mkt_ret'].sum()

        excess_return = total_return - bench_return

        trade_rets = [float(str(t['扣费收益']).replace('%', '')) / 100 for t in all_trades]
        trade_rets = np.array(trade_rets) if trade_rets else np.array([0])
        win_trades = (trade_rets > 0).sum()
        trade_win_rate = win_trades / len(trade_rets) if len(trade_rets) > 0 else 0

        result = {
            'label': label,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'n_weeks': n_weeks,
            'n_trades': len(all_trades),
            'initial_capital': INITIAL_CAPITAL,
            'final_capital': round(capital, 2),
            'total_return': total_return,
            'annual_return': annual_return,
            'bench_return': bench_return,
            'excess_return': excess_return,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'week_win_rate': week_win_rate,
            'trade_win_rate': trade_win_rate,
            'avg_weekly_return': weekly_rets.mean() if len(weekly_rets) > 0 else 0,
            'std_weekly_return': weekly_rets.std() if len(weekly_rets) > 1 else 0,
            'best_week': weekly_rets.max() if len(weekly_rets) > 0 else 0,
            'worst_week': weekly_rets.min() if len(weekly_rets) > 0 else 0,
            'weekly_results': weekly_results,
            'all_trades': all_trades,
        }

        print(f"\n    回测结果: {label}")
        print(f"    回测期: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({n_weeks}周)")
        print(f"    交易笔数: {len(all_trades)}")
        print(f"    累计收益率: {total_return*100:.2f}%")
        print(f"    年化收益率: {annual_return*100:.2f}%")
        print(f"    沪深300基准: {bench_return*100:.2f}%")
        print(f"    超额收益: {excess_return*100:.2f}%")
        print(f"    最大回撤: {max_dd*100:.2f}%")
        print(f"    夏普比: {sharpe:.3f}")
        print(f"    Calmar比: {calmar:.3f}")
        print(f"    周胜率: {week_win_rate*100:.1f}% ({win_weeks}/{len(weekly_rets)})")
        print(f"    交易胜率: {trade_win_rate*100:.1f}% ({win_trades}/{len(trade_rets)})")
        print(f"    平均周收益: {weekly_rets.mean()*100:.2f}%")
        print(f"    周收益标准差: {weekly_rets.std()*100:.2f}%")
        print(f"    最佳周: {weekly_rets.max()*100:.2f}% / 最差周: {weekly_rets.min()*100:.2f}%")

        return result

    combined_df = pd.concat([val_df, test_df], ignore_index=True)
    print(f"  合并数据集: {len(combined_df)} 条 (2025-10 ~ 2025-12)")

    combined_df['event_date_parsed'] = safe_parse_date(combined_df['event_date'])

    backtest_configs = [
        ("3周(赛题期)", pd.Timestamp('2025-12-08'), pd.Timestamp('2025-12-26')),
        ("12月整月", pd.Timestamp('2025-12-01'), pd.Timestamp('2025-12-31')),
        ("3个月(10-12月)", pd.Timestamp('2025-10-01'), pd.Timestamp('2025-12-31')),
    ]

    all_results = []
    for label, start, end in backtest_configs:
        result = run_backtest(combined_df.copy(), start, end, label)
        if result:
            all_results.append(result)

    print("\n[5/5] 汇总报告")

    print("\n  多期回测对比")
    print(f"  {'指标':<20} {'3周(赛题期)':<16} {'12月整月':<16} {'3个月(10-12月)':<16}")
    print("  " + "-" * 68)

    metrics = [
        ('回测期', 'start_date', '{0} ~ {1}', False),
        ('周数', 'n_weeks', '{0}', False),
        ('交易笔数', 'n_trades', '{0}', False),
        ('累计收益率', 'total_return', '{0:.2%}', True),
        ('年化收益率', 'annual_return', '{0:.2%}', True),
        ('沪深300基准', 'bench_return', '{0:.2%}', True),
        ('超额收益', 'excess_return', '{0:.2%}', True),
        ('最大回撤', 'max_drawdown', '{0:.2%}', True),
        ('夏普比', 'sharpe_ratio', '{0:.3f}', False),
        ('Calmar比', 'calmar_ratio', '{0:.3f}', False),
        ('周胜率', 'week_win_rate', '{0:.1%}', True),
        ('交易胜率', 'trade_win_rate', '{0:.1%}', True),
        ('平均周收益', 'avg_weekly_return', '{0:.2%}', True),
        ('周收益标准差', 'std_weekly_return', '{0:.2%}', True),
        ('最佳周', 'best_week', '{0:.2%}', True),
        ('最差周', 'worst_week', '{0:.2%}', True),
    ]

    for name, key, fmt, is_pct in metrics:
        vals = []
        for r in all_results:
            v = r[key]
            if key == 'start_date':
                vals.append(f"{r['start_date']}~{r['end_date']}")
            else:
                vals.append(fmt.format(v))
        print(f"  {name:<20} {vals[0] if len(vals) > 0 else 'N/A':<16} "
              f"{vals[1] if len(vals) > 1 else 'N/A':<16} "
              f"{vals[2] if len(vals) > 2 else 'N/A':<16}")

    summary_rows = []
    for r in all_results:
        summary_rows.append({
            '回测期': r['label'],
            '起始日期': r['start_date'],
            '结束日期': r['end_date'],
            '周数': r['n_weeks'],
            '交易笔数': r['n_trades'],
            '初始资金': r['initial_capital'],
            '期末资金': r['final_capital'],
            '累计收益率': f"{r['total_return']*100:.2f}%",
            '年化收益率': f"{r['annual_return']*100:.2f}%",
            '沪深300基准': f"{r['bench_return']*100:.2f}%",
            '超额收益': f"{r['excess_return']*100:.2f}%",
            '最大回撤': f"{r['max_drawdown']*100:.2f}%",
            '夏普比': f"{r['sharpe_ratio']:.3f}",
            'Calmar比': f"{r['calmar_ratio']:.3f}",
            '周胜率': f"{r['week_win_rate']*100:.1f}%",
            '交易胜率': f"{r['trade_win_rate']*100:.1f}%",
            '平均周收益': f"{r['avg_weekly_return']*100:.2f}%",
            '周收益标准差': f"{r['std_weekly_return']*100:.2f}%",
            '最佳周': f"{r['best_week']*100:.2f}%",
            '最差周': f"{r['worst_week']*100:.2f}%",
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, '多期回测对比.csv'), index=False, encoding='utf-8-sig')

    for r in all_results:
        label_safe = r['label'].replace('(', '').replace(')', '').replace(' ', '_')
        pd.DataFrame(r['weekly_results']).to_csv(
            os.path.join(OUTPUT_DIR, f'周度收益_{label_safe}.csv'), index=False, encoding='utf-8-sig')
        pd.DataFrame(r['all_trades']).to_csv(
            os.path.join(OUTPUT_DIR, f'交易明细_{label_safe}.csv'), index=False, encoding='utf-8-sig')

    # 生成result.xlsx（赛题提交格式，使用3周赛题期）
    # 综合评分>0.5驱动选股 + 评分加权资金分配
    if len(all_results) > 0:
        result_trades = all_results[0]['all_trades']
        
        from collections import defaultdict
        weekly_trade_scores = defaultdict(list)
        for trade in result_trades:
            score_val = float(trade.get('综合评分', 0.5)) if trade.get('综合评分', '') not in ['', 'N/A'] else 0.5
            weekly_trade_scores[(trade['年'], trade['周'])].append((trade['股票代码'], score_val))
        
        weekly_ratios_map = {}
        for (year, week), code_scores in weekly_trade_scores.items():
            scores = [max(0.01, s) for _, s in code_scores]
            score_sum = sum(scores)
            raw_ratios = [s / score_sum for s in scores]
            ratios = [max(0.20, min(0.40, r)) for r in raw_ratios]
            ratio_sum = sum(ratios)
            ratios = [round(r / ratio_sum, 4) for r in ratios]
            for i, (code, _) in enumerate(code_scores):
                weekly_ratios_map[(year, week, code)] = ratios[i]
        
        from datetime import timedelta
        seen_weeks = []
        for trade in result_trades:
            wk_key = (trade['年'], trade['周'])
            if wk_key not in seen_weeks:
                seen_weeks.append(wk_key)
        week_date_map = {}
        for idx, (yr, wk) in enumerate(seen_weeks):
            try:
                week_monday = pd.Timestamp.fromisocalendar(yr, wk, 1)
                week_friday = week_monday + timedelta(days=4)
                date_range = f"{week_monday.strftime('%m.%d')}-{week_friday.strftime('%m.%d')}"
                week_date_map[(yr, wk)] = f"第{idx+1}周({date_range})"
            except:
                week_date_map[(yr, wk)] = f"第{idx+1}周"

        result_rows = []
        for trade in result_trades:
            code_raw = trade['股票代码']
            code_pure = code_raw.split('.')[0] if '.' in str(code_raw) else str(code_raw)
            code_pure = str(code_pure).zfill(6)
            trade_ratio = weekly_ratios_map.get(
                (trade['年'], trade['周'], trade['股票代码']),
                round(1.0 / MAX_STOCKS_PER_WEEK, 4)
            )
            result_rows.append({
                '周': week_date_map.get((trade['年'], trade['周']), f"第?周"),
                '事件名称': refine_event_name(trade.get('事件名称', trade.get('事件子类', '')), trade.get('事件子类', '')),
                '标的股票代码': code_pure,
                '资金比例': trade_ratio,
                '选股依据': f"综合评分={trade.get('综合评分', 0):.4f}",
            })
        if result_rows:
            result_df = pd.DataFrame(result_rows)
            try:
                from openpyxl import Workbook
                wb = Workbook()
                ws = wb.active
                ws.append(['周', '事件名称', '标的股票代码', '资金比例', '选股依据'])
                for row_idx, (_, row) in enumerate(result_df.iterrows(), start=2):
                    code_cell = ws.cell(row=row_idx, column=3, value=str(row['标的股票代码']))
                    code_cell.number_format = '@'
                    ws.cell(row=row_idx, column=1, value=str(row['周']))
                    ws.cell(row=row_idx, column=2, value=str(row['事件名称']))
                    ws.cell(row=row_idx, column=4, value=float(row['资金比例']))
                    ws.cell(row=row_idx, column=5, value=str(row['选股依据']))
                ws.column_dimensions['A'].width = 20
                ws.column_dimensions['B'].width = 28
                ws.column_dimensions['C'].width = 12
                ws.column_dimensions['D'].width = 10
                ws.column_dimensions['E'].width = 18
                wb.save(os.path.join(OUTPUT_DIR, 'result.xlsx'))
            except ImportError:
                result_df.to_excel(os.path.join(OUTPUT_DIR, 'result.xlsx'), index=False)
            print(f"\n  result.xlsx已生成: {len(result_rows)} 条记录")

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
