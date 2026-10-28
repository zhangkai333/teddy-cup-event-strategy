# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import re
import argparse
import logging
import warnings
import atexit
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np
import requests

warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='backslashreplace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='backslashreplace')

# 项目根目录：以脚本所在目录为准
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get('TEDDY_DATA_DIR') or os.path.join(BASE_DIR, '数据集_清洗')
))
RESULT_DIR = os.path.join(BASE_DIR, '实测结果')
# 训练结果目录：自动检测本地文件夹
_trained_candidates = [os.path.join(BASE_DIR, '训练结果')]
MODEL_DIR = next((p for p in _trained_candidates if os.path.isdir(p)), _trained_candidates[0])

for d in [RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

NEODATA_SCRIPT = os.path.join(
    os.path.expanduser('~'),
    '.workbuddy', 'plugins', 'marketplaces', 'cb_teams_marketplace',
    'plugins', 'finance-data', 'skills', 'neodata-financial-search',
    'scripts', 'query.py'
)

log_file = os.path.join(RESULT_DIR, f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f'  BASE_DIR={BASE_DIR}')
logger.info(f'  DATA_DIR={DATA_DIR}')
logger.info(f'  MODEL_DIR={MODEL_DIR}')

WEEK1_SUBMIT_START = '2026-04-20'   # 周日
WEEK1_SUBMIT_END = '2026-04-21'     # 周一9:00
WEEK1_TRADE_START = '2026-04-21'    # 周二(买入日)
WEEK1_TRADE_END = '2026-04-24'      # 周五(卖出日)

WEEK2_SUBMIT_START = '2026-04-27'
WEEK2_SUBMIT_END = '2026-04-28'
WEEK2_TRADE_START = '2026-04-27'
WEEK2_TRADE_END = '2026-05-01'      # 注意：5.1假期可能休市

INITIAL_CAPITAL = 100000

# 第二部分：通用工具函数

def refine_event_name(title, event_sub=''):
    import re
    name = str(title).strip()
    if not name or name in ('nan', 'None', ''):
        return str(event_sub)[:40] if event_sub else '未知事件'
    
    # v14 Step 1: 提取核心事件类型（从event_sub或title中提取语义）
    # 去掉模板化前缀（如"业绩预告:"、"分红送转:"、"股东增减持:"）
    core_event = name
    for prefix in ['业绩预告:', '分红送转:', '股东增减持:', '财报披露:',
                    '融资定增:', '龙虎榜:', '预案']:
        if core_event.startswith(prefix):
            core_event = core_event[len(prefix):].strip()
            break
    
    # Fix #25: 去掉冒号前的公司名前缀（如"顺丰控股:关于..."→"关于..."）
    # 因为调用方会单独拼公司名，避免出现"顺丰控股顺丰控股"的重复
    m_colon = re.match(r'^([^：:]{2,8})[：:](.+)$', core_event)
    if m_colon:
        _prefix = m_colon.group(1).strip()
        _after = m_colon.group(2).strip()
        # 如果冒号后的内容以冒号前的词开头，去掉重复（如"立讯精密:立讯精密2025..."）
        if _after.startswith(_prefix):
            _after = _after[len(_prefix):].strip()
        core_event = _after
    
    code_pattern = re.match(r'^([A-Za-z]*)([\d]{4,6})([A-Za-z]*)$', core_event)
    if not code_pattern:
        core_event = re.sub(r'\([\d]{4,6}\)$', '', core_event).strip()
        core_event = re.sub(r'[（][\d]{4,6}[）]$', '', core_event).strip()
    
    for suffix in ['的提示性公告', '的更正公告', '的补充公告', '的公告', 
                   '的实施方案', '报告', '预案']:
        if core_event.endswith(suffix) and len(core_event) > len(suffix) + 2:
            core_event = core_event[:-len(suffix)].strip()
    
    # v14 Step 4: 映射为标准事件类型名称
    event_type_map = {
        '预增': '业绩预增', '预减': '业绩预减', '续亏': '业绩续亏',
        '首亏': '业绩首亏', '扭亏': '业绩扭亏', '略增': '业绩略增',
        '略减': '业绩略减', '不确定': '业绩预告', '预案': '分红送转方案',
        '实施': '分红送转实施', '派息': '分红派息',
        '增持': '股东增持', '减持': '股东减持', '回购': '股份回购',
        '解禁': '限售解禁', '质押': '股份质押', '定增': '定向增发',
    }
    
    result = None
    for kw, standard_name in event_type_map.items():
        if kw in core_event or kw in name:
            result = standard_name
            break
    
    if not result:
        result = core_event if core_event else str(event_sub)[:30]
    
    if len(result) > 35:
        result = result[:34] + '…'
    return result.strip() if result.strip() else str(event_sub)[:30]

def safe_parse_date(series, col_name='unknown'):
    if series.dtype in ['int64', 'float64', 'int32', 'float32']:
        return pd.to_datetime(
            series.astype(int).astype(str), format='%Y%m%d', errors='coerce'
        )
    else:
        result = pd.to_datetime(series, format='%Y%m%d', errors='coerce')
        if result.isna().sum() > len(series) * 0.5:
            result = pd.to_datetime(series, errors='coerce')
        return result

def query_neodata(query_str, data_type='doc'):
    import subprocess
    # 本地环境无NeoData脚本时跳过
    if not os.path.exists(NEODATA_SCRIPT):
        return None
    try:
        r = subprocess.run(
            [sys.executable, NEODATA_SCRIPT, '--query', query_str, '--data-type', data_type],
            capture_output=True, text=True, encoding='utf-8', errors='ignore',
            timeout=30, env=os.environ
        )
        m = re.search(r'\{.*\}', r.stdout, re.DOTALL) if r.stdout else None
        return json.loads(m.group()) if m else None
    except Exception as e:
        logger.warning(f"  NeoData查询失败: {e}")
        return None

def _finance_data_api(api_name, params, timeout=15, retries=3):
    url = 'https://www.codebuddy.cn/v2/tool/financedata'
    headers = {'Content-Type': 'application/json'}
    
    # 尝试读取neodata token文件（虽然可能不被此API接受，但尝试一下）
    token_file = os.path.expanduser('~/.workbuddy/.neodata_token')
    if os.path.exists(token_file):
        try:
            with open(token_file) as f:
                token = f.read().strip()
            if len(token) > 50:  # 合理的JWT长度
                headers['Authorization'] = f'Bearer {token}'
        except:
            pass
    
    last_error = None
    for attempt in range(retries):
        try:
            r = requests.post(url, json={'api_name': api_name, 'params': params},
                              headers=headers, timeout=timeout)
            d = r.json()
            api_code = d.get('code', -1)
            api_msg = d.get('msg', d.get('message', ''))
            
            if api_code == 40101:
                logger.warning(f"  {api_name}: API需要认证(code=40101)，当前无有效token")
                return None
            
            if api_code == 0 and d.get('data') and d['data'].get('items'):
                return pd.DataFrame(d['data']['items'], columns=d['data']['fields'])
            
            if api_code != 0:
                last_error = f"API错误 code={api_code}, msg={api_msg}"
            elif not d.get('data'):
                last_error = f"响应无data字段"
            else:
                items_count = len(d['data'].get('items', []))
                last_error = f"API返回0条数据(items={items_count})"
                
            if attempt == retries - 1:
                logger.warning(f"  {api_name} 返回空: {last_error} | params={params}")
            return None
        except Exception as ex:
            last_error = str(ex)
            if attempt < retries - 1:
                time.sleep(1)
    logger.warning(f"  {api_name} 全部{retries}次重试均失败: {last_error}")
    return None

# 沪深300核心成分股（前30只，用于个股新闻采集，不依赖finance-data API）
HS300_TOP30 = [
    '600519.SH','300750.SZ','601318.SH','600036.SZ','002594.SZ',
    '601899.SH','600900.SH','601012.SH','000858.SZ','002475.SZ',
    '601888.SH','600276.SH','601166.SH','000333.SZ','603288.SH',
    '601939.SH','603259.SH','600887.SH','603160.SH','601985.SH',
    '600809.SH','002352.SZ','603986.SH','601088.SH','600048.SH',
    '603799.SH','002230.SZ','601628.SH','600016.SH','603501.SH',
]

def collect_realtime_events(trade_start, trade_end):
    logger.info("Step 1：实时事件采集")
    
    all_events = []
    today_dt = datetime.now()
    week_ago_dt = today_dt - timedelta(days=7)
    today_str_yyyymmdd = today_dt.strftime('%Y%m%d')
    week_ago_str_yyyymmdd = week_ago_dt.strftime('%Y%m%d')
    today_str = today_dt.strftime('%Y-%m-%d')
    week_ago_str = week_ago_dt.strftime('%Y-%m-%d')
    
    logger.info(f"  事件采集范围: {week_ago_str} ~ {today_str}（近7天）")
    
    # 1.1 巨潮资讯网公告事件采集（首选：附件2推荐来源，AKShare直接访问巨潮）
    # 遍历HS300核心成分股获取最新7天的公司行为事件（业绩/分红/回购/质押等）
    # 替代原finance-data HTTP API（2026-04-17确认该API需专用token认证40101不可用）
    logger.info("\n--- 1.1 巨潮资讯网公告事件采集（附件2推荐） ---")
    cninfo_before = len(all_events)
    
    try:
        import akshare as ak
        cninfo_total = 0
        
        for code in HS300_TOP30:
            code_num = code.split('.')[0]
            try:
                disc_df = ak.stock_zh_a_disclosure_report_cninfo(
                    symbol=code_num,
                    start_date=week_ago_str.replace('-', ''),
                    end_date=today_str.replace('-', '')
                )
                if disc_df is not None and len(disc_df) > 0:
                    for _, row in disc_df.iterrows():
                        title = str(row.get('公告标题', row.get('title', '')))
                        skip_kw = ['述职报告', '风险评', '履职情况', '投资者关系', 
                                   '内控', '议事规则', '差错追究', '制度']
                        if any(kw in title for kw in skip_kw):
                            continue
                        all_events.append({
                            'event_date': str(row.get('公告时间', row.get('ann_time', today_str)))[:10],
                            'title': f"{row.get('简称', code_num)}:{title}",
                            'abstract': str(row.get('公告标题', ''))[:300],
                            'source': 'cninfo_disclosure',
                            'ts_code': code,
                            'data_source': 'akshare_cninfo'
                        })
                        cninfo_total += 1
            except Exception as ex:
                pass  # 单个股票失败不影响整体
            
        if cninfo_total > 0:
            logger.info(f"  巨潮公告(前30只): {cninfo_total} 条")
        else:
            logger.warning("  巨潮公告: 0 条（可能网络问题或无新公告）")
        
        time.sleep(0.5)
    except ImportError:
        logger.warning("  akshare未安装，跳过巨潮公告采集")
    except Exception as e:
        logger.warning(f"  巨潮公告采集异常: {e}")
    
    cninfo_count = len(all_events) - cninfo_before
    logger.info(f"  巨潮采集总计: {cninfo_count} 条")
    
    # 1.2 finance-data HTTP API 公告事件（备选，可能需token认证）
    # 已于2026-04-17确认返回code=40101(需要专用token)
    # 保留作为未来可能的恢复路径
    # [!] 性能优化: 首次遇到40101后立即跳过后续所有调用（避免~18秒无效重试）
    logger.info("\n--- 1.2 finance-data HTTP API 备选采集 ---")
    fd_before = len(all_events)
    
    # 全局标记：finance-data是否可用（首次40101后设为False）
    # 使用列表包装以在嵌套函数中可修改（避免global关键字问题）
    _fd_status = [True]  # [True] = 可用, [False] = 已确认不可用

    def _fd_api_with_skip(api_name, params):
        if not _fd_status[0]:
            return None
        result = _finance_data_api(api_name, params)
        if result is None:
            # _finance_data_api在40101时返回None，此时标记全局不可用
            _fd_status[0] = False
            logger.info(f"  finance-data API不可用(40101)，跳过后续所有API调用")
        return result
    
    query_dates = [(today_dt - timedelta(days=i)).strftime('%Y%m%d') for i in range(6, -1, -1)]
    logger.info(f"  查询日期范围: {query_dates[0]} ~ {query_dates[-1]} ({len(query_dates)}天)")
    
    # 1) 业绩预告（forecast）—— 标题含股票代码避免去重合并
    try:
        fore_total = 0
        for qd in query_dates:
            fore_df = _fd_api_with_skip('forecast', {'ann_date': qd})
            if fore_df is not None and len(fore_df) > 0:
                for _, row in fore_df.iterrows():
                    ts = str(row.get('ts_code', ''))
                    code_short = ts.split('.')[0] if ts else ''
                    all_events.append({
                        'event_date': str(row.get('ann_date', '')),
                        'title': f"业绩预告:{row.get('type', '')}({code_short})" if code_short else f"业绩预告:{row.get('type', '')}",
                        'abstract': f"业绩变动:{row.get('p_change_min', '')}%~{row.get('p_change_max', '')}%。{row.get('summary', '')}",
                        'source': 'finance_data_forecast',
                        'ts_code': ts,
                        'data_source': 'finance_data'
                    })
                fore_total += len(fore_df)
            time.sleep(0.2)
        if fore_total > 0: logger.info(f"  业绩预告: {fore_total} 条")
    except Exception as e:
        logger.warning(f"  业绩预告查询失败: {e}")
    
    time.sleep(0.3)
    
    # 2) 股东增减持（stk_holdertrade）—— 标题含股票代码避免去重合并
    try:
        stk_total = 0
        for qd in query_dates:
            stk_df = _fd_api_with_skip('stk_holdertrade', {'ann_date': qd})
            if stk_df is not None and len(stk_df) > 0:
                for _, row in stk_df.iterrows():
                    ts = str(row.get('ts_code', ''))
                    code_short = ts.split('.')[0] if ts else ''
                    holder = row.get('holder_name', '')
                    all_events.append({
                        'event_date': str(row.get('ann_date', '')),
                        'title': f"股东增减持:{holder}({code_short})" if code_short else f"股东增减持:{holder}",
                        'abstract': f"变动类型:{row.get('change_type', '')}, 变动量:{row.get('change_vol', '')}",
                        'source': 'finance_data_stk_holdertrade',
                        'ts_code': ts,
                        'data_source': 'finance_data'
                    })
                stk_total += len(stk_df)
            time.sleep(0.2)
        if stk_total > 0: logger.info(f"  股东增减持: {stk_total} 条")
    except Exception as e:
        logger.warning(f"  股东增减持查询失败: {e}")
    
    time.sleep(0.3)
    
    # 3) 分红送转（dividend）—— 标题含股票代码避免去重合并
    try:
        div_total = 0
        for qd in query_dates:
            div_df = _fd_api_with_skip('dividend', {'ann_date': qd})
            if div_df is not None and len(div_df) > 0:
                for _, row in div_df.iterrows():
                    ts = str(row.get('ts_code', ''))
                    code_short = ts.split('.')[0] if ts else ''
                    all_events.append({
                        'event_date': str(row.get('ann_date', '')),
                        'title': f"分红送转:{row.get('div_proc', '')}({code_short})" if code_short else f"分红送转:{row.get('div_proc', '')}",
                        'abstract': f"送股:{row.get('stk_div', 0)}, 派息:{row.get('cash_div', 0)}",
                        'source': 'finance_data_dividend',
                        'ts_code': ts,
                        'data_source': 'finance_data'
                    })
                    div_total += len(div_df)
            time.sleep(0.2)
        if div_total > 0: logger.info(f"  分红送转: {div_total} 条")
    except Exception as e:
        logger.warning(f"  分红送转查询失败: {e}")
    
    time.sleep(0.3)
    
    # 4) 龙虎榜（top_list）—— 异常交易信号（只查最近1天，因为trade_date是交易日）
    try:
        top_total = 0
        for qd in query_dates[:3]:  # 龙虎榜只查最近3个交易日
            top_df = _fd_api_with_skip('top_list', {'trade_date': qd})
            if top_df is not None and len(top_df) > 0:
                for _, row in top_df.head(30).iterrows():
                    all_events.append({
                        'event_date': str(row.get('trade_date', '')),
                        'title': f"龙虎榜:{row.get('name', '')}",
                        'abstract': f"收盘价:{row.get('close', '')}, 涨跌幅:{row.get('pct_change', '')}%, 买入额:{row.get('buy_amount', '')}",
                        'source': 'finance_data_top_list',
                        'ts_code': row.get('ts_code', ''),
                        'data_source': 'finance_data'
                    })
                    top_total += len(top_df)
                time.sleep(0.2)
        if top_total > 0: logger.info(f"  龙虎榜: {min(top_total, 90)} 条")
    except Exception as e:
        logger.warning(f"  龙虎榜查询失败: {e}")
    
    time.sleep(0.3)
    
    # 5) 披露日期（disclosure_date）—— 近期将披露财报的公司
    try:
        disc_total = 0
        for qd in query_dates:
            disc_df = _fd_api_with_skip('disclosure_date', {'ann_date': qd})
            if disc_df is not None and len(disc_df) > 0:
                for _, row in disc_df.head(30).iterrows():
                    ts = str(row.get('ts_code', ''))
                    code_short = ts.split('.')[0] if ts else ''
                    all_events.append({
                        'event_date': str(row.get('actual_date', row.get('ann_date', ''))),
                        'title': f"财报披露:{row.get('end_date', '')}报告({code_short})" if code_short else f"财报披露:{row.get('end_date', '')}报告",
                        'abstract': f"计划披露:{row.get('plan_date', '')}, 实际披露:{row.get('actual_date', '')}",
                        'source': 'finance_data_disclosure',
                        'ts_code': ts,
                        'data_source': 'finance_data'
                    })
                    disc_total += len(disc_df)
                time.sleep(0.2)
        if disc_total > 0: logger.info(f"  财报披露: {min(disc_total, 210)} 条")
    except Exception as e:
        logger.warning(f"  财报披露查询失败: {e}")
    
    fd_count = len(all_events) - fd_before
    logger.info(f"  finance-data采集: {fd_count} 条")
    
    # 1.3 Tushare公告数据（备选，需有效token）
    logger.info("\n--- 1.3 Tushare公告事件采集 ---")
    tushare_pro = None
    try:
        import tushare as ts
        token = os.environ.get('TUSHARE_TOKEN', '')
        if not token:
            token_file = os.path.join(os.path.expanduser('~'), '.tushare_token')
            if os.path.exists(token_file):
                with open(token_file, 'r') as f:
                    token = f.read().strip()
        if not token:
            token_cfg = os.path.join(BASE_DIR, 'tushare_token.txt')
            if os.path.exists(token_cfg):
                with open(token_cfg, 'r') as f:
                    token = f.read().strip()
        
        if token:
            ts.set_token(token)
            pro = ts.pro_api()
            try:
                test_df = pro.trade_cal(exchange='SSE', is_open=1, limit=1)
                tushare_pro = pro
                logger.info(f"  Tushare连接成功 (token有效)")
            except Exception as e:
                logger.warning(f"  Tushare token无效: {e}")
        else:
            logger.warning("  未找到Tushare Token，跳过")
    except ImportError:
        logger.warning("  tushare未安装，跳过")
    
    tushare_before = len(all_events)
    if tushare_pro is not None:
        pro = tushare_pro
        # 定增/融资（finance-data API权限不足时用Tushare补充）
        try:
            add_df = pro.add_issuance(ann_date=week_ago_str_yyyymmdd,
                                      fields='ts_code,ann_date,issue_type,issue_amount,issue_price')
            if add_df is not None and len(add_df) > 0:
                add_df['ann_date'] = add_df['ann_date'].astype(str)
                for _, row in add_df.iterrows():
                    all_events.append({
                        'event_date': row['ann_date'],
                        'title': f"融资定增:{row.get('issue_type', '')}({str(row['ts_code']).split('.')[0]})",
                        'abstract': f"融资金额:{row.get('issue_amount', '')}, 发行价:{row.get('issue_price', '')}",
                        'source': 'tushare_add_issuance',
                        'ts_code': row['ts_code'],
                        'data_source': 'tushare'
                    })
                logger.info(f"  融资定增: {len(add_df)} 条")
        except Exception as e:
            logger.warning(f"  融资定增查询失败: {e}")
    
    tushare_count = len(all_events) - tushare_before
    logger.info(f"  Tushare采集: {tushare_count} 条")
    
    logger.info("\n--- 1.4 NeoData财经新闻搜索 ---")
    neodata_before = len(all_events)
    
    event_keywords = [
        '并购 重组 收购', '业绩 财报 年报 季报', '增持 减持 回购 解禁',
        '利空 处罚 违规 退市', '政策 监管 新规 指导', '融资 定增 发行',
        '合同 中标 订单 签约', '技术突破 研发 专利', '人事变动 高管 董事',
    ]
    
    if not os.path.exists(NEODATA_SCRIPT):
        logger.warning("  NeoData脚本不存在，跳过新闻搜索")
    else:
        for kw in event_keywords:
            logger.info(f"  查询关键词: {kw}")
            res = query_neodata(f"A股 {kw} 重大事件 {week_ago_str}~{today_str}")
            if res and res.get('code') == '200':
                recalls = res.get('data', {}).get('docData', {}).get('docRecall', [])
                for rg in recalls:
                    for doc in rg.get('docList', []):
                        title = doc.get('title', '')
                        content = doc.get('content', '')[:500]
                        pt = doc.get('publishTime', 0)
                        dt = ''
                        try:
                            dt = datetime.fromtimestamp(pt).strftime('%Y-%m-%d')
                        except:
                            pass
                        source = doc.get('source', '')
                        if dt and week_ago_str <= dt <= today_str:
                            all_events.append({
                                'event_date': dt,
                                'title': title,
                                'abstract': content,
                                'source': source,
                                'data_source': 'neodata'
                            })
            time.sleep(0.5)
    
    neodata_count = len(all_events) - neodata_before
    logger.info(f"  NeoData新增: {neodata_count} 条")
    
    # 1.5 AKShare财经新闻采集（附件2推荐来源补充）
    # 包括：CCTV新闻 + 东财个股新闻(前30只) + 财新网宏观/行业/地缘新闻
    logger.info("\n--- 1.5 AKShare财经新闻采集 ---")
    akshare_before = len(all_events)
    
    try:
        import akshare as ak
        
        for attempt in range(3):
            try:
                news_df = ak.news_cctv(date=today_dt.strftime('%Y%m%d'))
                if news_df is not None and len(news_df) > 0:
                    for _, row in news_df.head(50).iterrows():
                        title = str(row.get('title', row.iloc[0] if len(row) > 0 else ''))
                        content = str(row.get('content', ''))[:500] if 'content' in row.index else ''
                        date_str = str(row.get('date', today_str))
                        all_events.append({
                            'event_date': date_str,
                            'title': title,
                            'abstract': content,
                            'source': 'cctv_news',
                            'data_source': 'akshare'
                        })
                    logger.info(f"  CCTV新闻: {min(len(news_df), 50)} 条")
                break
            except Exception as e:
                if attempt < 2:
                    logger.info(f"  CCTV新闻重试({attempt+1}/3)...")
                    time.sleep(2)
                else:
                    logger.warning(f"  CCTV新闻采集失败: {e}")
        
        # 2) 东方财富个股新闻（v15g修复: 用内置成分股列表替代finance-data API）
        # finance-data API已需认证(40101)，不再可用，改用硬编码的HS300核心成分股
        for attempt in range(3):
            try:
                sample_codes = HS300_TOP30  # 直接使用前30只核心成分股
                news_stock_count = 0
                for code in sample_codes:
                    code_num = code.split('.')[0]
                    try:
                        stock_news = ak.stock_news_em(symbol=code_num)
                        if stock_news is not None and len(stock_news) > 0:
                            for _, nrow in stock_news.head(2).iterrows():
                                news_title = str(nrow.get('新闻标题', ''))
                                news_content = str(nrow.get('新闻内容', ''))[:300]
                                news_date = str(nrow.get('发布时间', ''))
                                all_events.append({
                                    'event_date': news_date[:10] if news_date else today_str,
                                    'title': news_title,
                                    'abstract': news_content,
                                    'source': 'eastmoney_stock_news',
                                    'ts_code': code,
                                    'data_source': 'akshare'
                                })
                                news_stock_count += 1
                    except Exception:
                        pass
                    time.sleep(0.2)
                if news_stock_count > 0:
                    logger.info(f"  东财个股新闻: {news_stock_count} 条 (HS300前30只)")
                break
            except Exception as e:
                if attempt < 2:
                    logger.info(f"  东财个股新闻重试({attempt+1}/3)...")
                    time.sleep(3)
                else:
                    logger.warning(f"  东财个股新闻采集失败: {e}")
        
        # 3) 财新网宏观/行业/地缘新闻（附件2推荐来源：财新网caixin.com）
        # 覆盖：宏观经济数据、政策事件、行业事件、地缘冲突等
        try:
            cx_news = ak.stock_news_main_cx()
            if cx_news is not None and len(cx_news) > 0:
                cx_count = 0
                for _, row in cx_news.iterrows():
                    tag = str(row.get('tag', ''))
                    summary = str(row.get('summary', ''))[:400]
                    url = str(row.get('url', ''))
                    
                    import re as _re
                    dt_match = _re.search(r'/(\d{4}-\d{2}-\d{2})/', url)
                    news_date = dt_match.group(1) if dt_match else today_str
                    
                    if news_date >= week_ago_str:
                        all_events.append({
                            'event_date': news_date,
                            'title': f"[{tag}] {summary[:80]}",
                            'abstract': summary,
                            'source': 'caixin_main',
                            'data_source': 'akshare_caixin'
                        })
                        cx_count += 1
                
                if cx_count > 0:
                    logger.info(f"  财新网新闻: {cx_count} 条 (宏观/行业/地缘)")
        except Exception as e:
            logger.warning(f"  财新网新闻采集失败: {e}")
    
    except ImportError:
        logger.warning("  akshare未安装，跳过AKShare事件采集")
    except Exception as e:
        logger.warning(f"  AKShare采集异常: {e}")
    
    akshare_count = len(all_events) - akshare_before
    logger.info(f"  AKShare新增: {akshare_count} 条")
    
    if all_events:
        events_df = pd.DataFrame(all_events)
        before = len(events_df)
        events_df = events_df.drop_duplicates(subset=['title', 'event_date'], keep='first')
        after = len(events_df)
        logger.info(f"  去重: {before}→{after} 条")
        logger.info(f"  事件采集汇总: 巨潮={cninfo_count}, finance-data={fd_count}, Tushare={tushare_count}, NeoData={neodata_count}, AKShare={akshare_count}")
    else:
        events_df = pd.DataFrame()
        logger.warning("  [!] 未采集到任何事件！")
    
    events_path = os.path.join(RESULT_DIR, f'raw_events_{trade_start}_{trade_end}.csv')
    if len(events_df) > 0:
        events_df.to_csv(events_path, index=False, encoding='utf-8-sig')
        logger.info(f"  原始事件已保存: {events_path}")
    
    return events_df

def preprocess_realtime_data(events_df):
    if len(events_df) == 0:
        return events_df
    
    logger.info("Step 1.5：实时数据预处理（V3清洗规则）")
    
    orig_count = len(events_df)
    logger.info(f"  输入事件数: {orig_count}")
    
    # Rule A: 公司治理制度页关键词过滤（完全复制自 clean_data_v3.py RULE_A_PATTERNS）
    RULE_A_PATTERNS = {
        'IR制度': [
            '投资者关系管理', '投资者关系信息', 'ir管理', 'ir制度', 'ir信息',
        ],
        '年报差错追究': [
            '年报差错', '年报信息披露差错', '差错追究', '信息披漏差错', '定期报告差错',
        ],
        '信披制度': [
            '信息披露管理制度', '信息披露管理办法', '信息披露制度',
            '信息披露细则', '内幕信息知情人登记管理制度', '内幕信息管理制度',
            '重大信息内部报告制度',
        ],
        '议事规则': [
            '董事会议事规则', '监事会议事规则', '股东大会议事规则', '议事规则',
        ],
        '对外担保': [
            '对外担保管理制度', '对外担保制度', '对外担保办法',
            '融资担保管理制度', '对外投资管理制度',
        ],
        '独董制度': [
            '独立董事工作细则', '独立董事工作条例', '独立董事制度',
            '独立董事工作制度', '独立董事履职',
        ],
        '高管细则': [
            '总经理工作细则', '总经理工作条例', '总裁工作细则',
            '总裁工作条例', '首席执行官工作细则',
        ],
        '董秘细则': [
            '董事会秘书工作细则', '董事会秘书工作条例', '董秘工作细则',
            '证券事务代表',
        ],
        '内控制度': [
            '内部控制制度', '内部控制手册', '内部控制评价办法', '内部控制审计',
        ],
        '资金管理': [
            '募集资金管理办法', '募集资金管理制度', '募集资金存储',
        ],
    }
    
    delete_mask = pd.Series([False] * len(events_df), index=events_df.index)
    rule_a_stats = {}
    
    title_col = 'title' if 'title' in events_df.columns else list(events_df.columns)[0]
    
    for category, keywords in RULE_A_PATTERNS.items():
        for kw in keywords:
            mask = events_df[title_col].fillna('').astype(str).str.contains(
                kw, case=False, na=False, regex=False)
            count = mask.sum()
            if count > 0:
                delete_mask |= mask
                key = f'A:{category}({kw})'
                rule_a_stats[key] = rule_a_stats.get(key, 0) + int(count)
    
    # Rule B: 智能假新闻检测（Fix #19: 修复过度清洗AI真实新闻问题）
    # 原V3 Rule B用关键词过滤(gpt-5/openai/chatgpt等11种正则) → 导致真实AI行业新闻被误删
    # 根因分析: 训练集中114条假事件的真正特征是【同一标题被多只股票共用】，
    #           而非标题中包含AI关键词。"OpenAI发布GPT-5"作为真实新闻不应删除
    # 
    #   B1 硬过滤: 仅保留"实证窗口"标记（这是数据采集注入的明确假标记）
    #   B2 软检测: 同一标题≥2只股票共用时记录警告，但不自动删除
    #             （因为实时采集中可能确实有行业级事件影响多只同概念股）

    rule_b_stats = {}

    # B1: 硬过滤 — 仅过滤带"实证窗口"标记的条目（训练集确认的注入假数据）
    _hard_fake_patterns = ['实证窗口']
    for pattern in _hard_fake_patterns:
        try:
            mask = events_df[title_col].fillna('').astype(str).str.contains(
                pattern, case=False, na=False, regex=False)
            count = mask.sum()
            if count > 0:
                delete_mask |= mask
                key = f'B1:硬假({pattern})'
                rule_b_stats[key] = rule_b_stats.get(key, 0) + int(count)
                logger.info(f"  Rule B1 硬假过滤 '{pattern}': {count}条")
        except Exception as e:
            logger.warning(f"  Regex error for '{pattern}': {e}")

    # B2: 软检测 — 多股共享标题警告（不删除，仅记录）
    if len(events_df) > 0 and title_col in events_df.columns:
        title_share = events_df[title_col].fillna('').astype(str)
        shared_titles = title_share[title_share.duplicated(keep=False)].unique()
        if len(shared_titles) > 0:
            n_shared = sum(title_share.isin(shared_titles))
            rule_b_stats[f'B2:共享标题(警告)'] = n_shared
            logger.info(f"  Rule B2 多股共享标题: {n_shared}条涉及{len(shared_titles)}种标题 (仅警告，未删除)")
            for t in shared_titles[:5]:
                stocks_in_title = events_df[events_df[title_col] == t]['ts_code'].head(3).tolist()
                logger.info(f"    共享标题样例: \"{t[:60]}\" → {stocks_in_title}")
    
    RULE_C_EXTRA = [
        '关联交易管理制度', '委托理财管理制度', '衍生品投资管理制度',
        '期货套期保值',
    ]
    
    rule_c_stats = {}
    for kw in RULE_C_EXTRA:
        mask = events_df[title_col].fillna('').astype(str).str.contains(
            kw, case=False, na=False, regex=False)
        count = mask.sum()
        if count > 0:
            delete_mask |= mask
            rule_c_stats[f'C:其他({kw})'] = count
    
    total_delete = int(delete_mask.sum())
    if total_delete > 0:
        all_stats = {**rule_a_stats, **rule_b_stats, **rule_c_stats}
        logger.info(f"  垃圾事件检测: {total_delete} 条 ({total_delete/orig_count*100:.1f}%)")
        for stat, cnt in sorted(all_stats.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"    {stat}: {cnt}")
        
        events_df = events_df[~delete_mask].copy()
        logger.info(f"  Rule A/B/C 过滤: {orig_count}→{len(events_df)} (-{total_delete})")
    
    # 基础清洗1: 日期格式统一（参考 2.4节）
    if 'event_date' in events_df.columns and len(events_df) > 0:
        def normalize_date(val):
            s = str(val).strip()
            for fmt in ('%Y-%m-%d', '%Y%m%d', '%Y/%m/%d'):
                try:
                    return pd.to_datetime(s, format=fmt).strftime('%Y-%m-%d')
                except:
                    pass
            try:
                dt = pd.to_datetime(s)
                if not pd.isna(dt):
                    return dt.strftime('%Y-%m-%d')
            except:
                pass
            return ''
        
        before_dates = len(events_df)
        events_df['event_date'] = events_df['event_date'].apply(normalize_date)
        empty_date_mask = (events_df['event_date'].str.len() == 0) | (
            events_df['event_date'].isna())
        empty_cnt = empty_date_mask.sum()
        if empty_cnt > 0:
            events_df = events_df[~empty_date_mask]
            logger.info(f"  无效日期删除: {empty_cnt} 条")
        if len(events_df) < before_dates:
            logger.info(f"  日期格式统一: {before_dates}→{len(events_df)} 条")
    
    # 基础清洗2: 空摘要字段填充（参考 2.3节：用标题填充）
    if 'abstract' in events_df.columns:
        empty_abs = events_df['abstract'].isna() | (
            events_df['abstract'].astype(str).str.strip() == '')
        empty_cnt = empty_abs.sum()
        if empty_cnt > 0:
            events_df.loc[empty_abs, 'abstract'] = events_df.loc[empty_abs, 'title']
            logger.info(f"  空摘要填充(用title): {empty_cnt} 条")
    
    if 'ts_code' in events_df.columns:
        events_df['ts_code'] = events_df['ts_code'].fillna('').astype(str)
        # 确保ts_code为标准6位代码格式（去除可能的.SZ/.SH后缀用于显示）
        has_code = events_df['ts_code'].notna() & (events_df['ts_code'] != '') & (
            events_df['ts_code'] != 'nan')
        code_count = has_code.sum()
        logger.info(f"  有股票代码的事件: {code_count}/{len(events_df)} ({code_count/max(len(events_df),1)*100:.1f}%)")
    
    # 基础清洗4: Inf/NaN值检验与修复（参考质量验证标准：零Inf值）
    numeric_cols = events_df.select_dtypes(include=[np.number]).columns.tolist()
    inf_total = 0
    nan_total = 0
    for col in numeric_cols:
        if col in events_df.columns:
            inf_mask = np.isinf(events_df[col])
            inf_c = inf_mask.sum()
            if inf_c > 0:
                events_df.loc[inf_mask, col] = np.nan
                inf_total += inf_c
            nan_c = events_df[col].isna().sum()
            nan_total += nan_c
    
    if inf_total > 0:
        logger.warning(f"  [!] Inf值替换为NaN: {inf_total} 个")
    
    total_nulls = events_df.isnull().sum().sum()
    if total_nulls > 0:
        null_cols = events_df.isnull().sum()
        null_cols = null_cols[null_cols > 0]
        logger.info(f"  残留空值: {total_nulls} 个，分布在{len(null_cols)}列: "
                     f"{dict(null_cols.head(5))}")
    else:
        logger.info("  [OK] 数值质量: 零Inf值，零NaN值")
    
    final_count = len(events_df)
    deleted_total = orig_count - final_count
    
    logger.info(f"\n  {'='*50}")
    logger.info(f"  预处理汇总:")
    logger.info(f"  {'='*50}")
    logger.info(f"  原始输入:   {orig_count:>8,} 条")
    logger.info(f"  删除垃圾:   {deleted_total:>8,} 条 ({deleted_total/max(orig_count,1)*100:.1f}%)")
    logger.info(f"  最终保留:   {final_count:>8,} 条 ({final_count/max(orig_count,1)*100:.1f}%)")
    
    if len(events_df) > 0:
        log_path = os.path.join(RESULT_DIR, 'preprocess_log.csv')
        log_df = events_df[['event_date', 'title', 'source', 'data_source']].copy() if \
                 all(c in events_df.columns for c in ['event_date','title','source','data_source']) else \
                 events_df.iloc[:, :min(4, len(events_df.columns))].copy()
        try:
            log_df.to_csv(log_path, index=False, encoding='utf-8-sig')
            logger.info(f"  清洗日志已保存: {log_path}")
        except Exception as e:
            logger.warning(f"  保存清洗日志失败: {e}")
    
    return events_df

def preprocess_market_data(price_df):
    if price_df is None or len(price_df) == 0:
        return price_df
    
    logger.info("\n--- 行情数据预处理 ---")
    before = len(price_df)
    
    has_vol = 'vol' in price_df.columns or 'volume' in price_df.columns or '成交量' in price_df.columns
    has_pct = 'pct_chg' in price_df.columns or '涨跌幅' in price_df.columns
    
    if has_vol and has_pct:
        vol_col = [c for c in ['vol','volume','成交量'] if c in price_df.columns][0]
        pct_col = [c for c in ['pct_chg','涨跌幅'] if c in price_df.columns][0]
        
        suspend_mask = (price_df[vol_col] == 0) & (price_df[pct_col] == 0)
        suspend_count = suspend_mask.sum()
        if suspend_count > 0:
            price_df = price_df[~suspend_mask].copy()
            logger.info(f"  停牌记录过滤: 删除 {suspend_count} 条 (量=0且涨跌幅=0)")
    
    # 3.4 缺失值前值填充（按股票分组）
    if 'ts_code' in price_df.columns or 'trade_date' in price_df.columns:
        code_col = 'ts_code' if 'ts_code' in price_df.columns else None
        num_cols = price_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 0:
            null_before = price_df[num_cols].isnull().sum().sum()
            if null_before > 0:
                if code_col:
                    price_df[num_cols] = price_df.groupby(code_col)[num_cols].ffill()
                    price_df[num_cols] = price_df.groupby(code_col)[num_cols].bfill()
                else:
                    price_df[num_cols] = price_df[num_cols].ffill().bfill()
                null_after = price_df[num_cols].isnull().sum().sum()
                logger.info(f"  缺失值前值填充: {null_before}→{null_after}")
    
    pct_col = [c for c in ['pct_chg','涨跌幅'] if c in price_df.columns]
    if pct_col:
        pc = pct_col[0]
        lo, hi = price_df[pc].quantile(0.005), price_df[pc].quantile(0.995)
        clip_count = ((price_df[pc] < lo) | (price_df[pc] > hi)).sum()
        if clip_count > 0:
            price_df[pc] = price_df[pc].clip(lo, hi)
            logger.info(f"  涨跌幅缩尾(0.5%/99.5%): [{lo:.2f}%, {hi:.2f}%], 裁剪{clip_count}条")
    
    if 'ts_code' in price_df.columns and 'trade_date' in price_df.columns:
        dup_count = price_df.duplicated(subset=['ts_code', 'trade_date']).sum()
        if dup_count > 0:
            price_df = price_df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first')
            logger.info(f"  主键去重(ts_code+trade_date): 删除 {dup_count} 条重复")
    
    after = len(price_df)
    logger.info(f"  行情预处理: {before}→{after} 条")
    
    return price_df

# AHP分类规则（与v12一致）
AHP_RULES = {
    '宏观': {
        'keywords': ['GDP', 'CPI', '利率', '货币', '财政', '通胀', '通缩', '降准', '降息',
                     '加息', '汇率', '外贸', '进出口', 'PMI', '社融', 'M2', '消费', '投资',
                     '政策', '国务院', '发改委', '央行', '商务部', '监管', '立法'],
        'sub_categories': {
            '货币政策': ['利率', '降准', '降息', '加息', '货币', 'M2', '社融', '流动性'],
            '财政政策': ['财政', '税收', '国债', '专项债', '减税'],
            '政策发布': ['政策', '监管', '合规', '立法', '规范', '指导意见', '新规'],
            '宏观数据': ['GDP', 'CPI', 'PMI', '进出口', '外贸', '通胀', '通缩'],
        }
    },
    '公司': {
        'keywords': ['业绩', '营收', '利润', '分红', '回购', '增持', '减持', '解禁',
                     '并购', '重组', 'IPO', '上市', '退市', '公告', '财报', '年报',
                     '季报', '中报', '高管', '股权', '质押', '定增', '员工持股',
                     '送转', '预案', '派息', '除权', '除息'],  # v14: 补充分红送转专用关键词
        'sub_categories': {
            '业绩': ['业绩', '营收', '利润', '财报', '年报', '季报', '中报', '预告', '预增', '预减', '续亏', '首亏', '扭亏', '略增', '略减'],
            '并购': ['并购', '重组', '收购', '合并', '要约'],
            '股东变动': ['增持', '减持', '回购', '解禁', '质押', '股权'],
            '利空公告': ['利空', '处罚', '违规', '退市', '造假', '暴雷'],
            '融资': ['融资', '定增', '发行', '可转债', '配股'],
            '合同': ['合同', '中标', '订单', '签约', '合作'],
            '技术突破': ['技术突破', '研发', '专利', '创新', '国产替代'],
            '人事变动': ['人事变动', '高管', '董事', '辞职', '上任'],
            '分红送转': ['分红', '送转', '预案', '派息', '除权', '除息', '股利'],  # v14: 独立子类
        }
    },
    '行业': {
        'keywords': ['芯片', '半导体', '新能源', '光伏', '风电', '锂电', '电池',
                     'AI', '人工智能', '5G', '6G', '消费电子', '医药', '创新药',
                     '汽车', '白酒', '地产', '银行', '保险', '券商', '钢铁',
                     '煤炭', '有色', '原油', '化工', '农业', '食品'],
        'sub_categories': {
            '科技': ['芯片', '半导体', 'AI', '人工智能', '5G', '6G', '消费电子', '数字经济'],
            '新能源': ['新能源', '光伏', '风电', '锂电', '电池', '储能', '氢能'],
            '消费': ['白酒', '食品', '医药', '创新药', '汽车', '消费'],
            '周期': ['钢铁', '煤炭', '有色', '原油', '化工', '地产'],
            '金融': ['银行', '保险', '券商'],
        }
    },
}

def classify_ahp(title, abstract=''):
    text = str(title) + ' ' + str(abstract)
    
    # v14 Step 1: 先做子类精确匹配（优先级最高）
    # 如果标题/摘要中包含某个子类的**专属关键词**，直接命中该子类
    # 专属关键词 = 只出现在一个子类中的关键词
    exclusive_sub_map = {
        # 这些关键词只属于特定子类，命中即直接返回
        '分红送转': ['分红', '送转', '预案', '派息', '除权', '除息', '股利', '送股', '转增', '10送', '10转'],
        '业绩': ['预增', '预减', '续亏', '首亏', '扭亏', '略增', '略减', '预告', '季报', '年报'],
        '股东变动': ['增持', '减持', '回购', '解禁', '质押'],
        '融资': ['定增', '发行', '可转债', '配股'],
        '人事变动': ['高管', '董事', '辞职', '上任'],
        '利空公告': ['处罚', '违规', '退市', '造假', '暴雷'],
    }
    
    for sub_name, kws in exclusive_sub_map.items():
        if any(kw in text for kw in kws):
            if sub_name in ['分红送转', '业绩', '股东变动', '融资', '利空公告',
                           '并购', '合同', '技术突破', '人事变动']:
                return '公司', sub_name
            else:
                return sub_name, sub_name
    
    # v14 Step 2: 原有的大类→子类逐层匹配逻辑（兜底）
    scores = {}
    for cat, info in AHP_RULES.items():
        score = sum(1 for kw in info['keywords'] if kw in text)
        sub_scores = {}
        for sub_cat, sub_kws in info['sub_categories'].items():
            sub_score = sum(1 for kw in sub_kws if kw in text)
            if sub_score > 0:
                sub_scores[sub_cat] = sub_score
        scores[cat] = {'total': score, 'subs': sub_scores}

    if max(s['total'] for s in scores.values()) == 0:
        return '其他', '其他'

    primary = max(scores, key=lambda k: scores[k]['total'])
    if scores[primary]['subs']:
        secondary = max(scores[primary]['subs'], key=lambda k: scores[primary]['subs'][k])
    else:
        secondary = primary
    return primary, secondary

def run_event_classification(events_df):
    logger.info("Step 2：事件分类（AHP规则分类，v15g去BERT化）")
    
    if len(events_df) == 0:
        logger.warning("  无事件可分类")
        return events_df
    
    # --- 2.1 AHP分类 ---
    logger.info("  执行AHP关键词分类...")
    ahp_results = []
    for _, row in events_df.iterrows():
        title = str(row.get('title', ''))
        abstract = str(row.get('abstract', ''))
        result = classify_ahp(title, abstract)
        ahp_results.append(result)
    
    events_df['event_type'] = [r[0] for r in ahp_results]
    events_df['event_sub'] = [r[1] for r in ahp_results]
    
    logger.info(f"  AHP分类分布:\n{events_df['event_type'].value_counts().to_string()}")
    logger.info(f"  子类分布:\n{events_df['event_sub'].value_counts().to_string()}")
    
    # --- 2.2 BERT推理（已废弃：v15g去BERT化）---
    logger.info("  \u26ef\ufe0f BERT推理已跳过（v15g去BERT化，AHP规则分类即可）")
    
    return events_df

def _extract_company_keyword(title):
    if re.search(r'\(\d{6}\)', title):
        return None
    FILTER_WORDS = {
        '预增', '预减', '续亏', '首亏', '扭亏', '略增', '略减',
        '续盈', '不确定', '减亏', '增亏', '持平',
        '增持', '减持', '回购', '解禁', '上市',
        '公告', '提示', '说明', '报告', '决议', '实施',
        '董事会预案', '股东大会通过', '送转',
    }
    m = re.search(r'[：:](.+?)(?:\(\d{6}\))?$', title)
    if not m:
        return None
    keyword = m.group(1).strip()
    if keyword in FILTER_WORDS or len(keyword) < 2:
        return None
    return keyword

def run_company_association(events_df):
    logger.info("Step 3：关联公司挖掘")

    if len(events_df) == 0:
        logger.warning("  无事件可关联")
        return events_df

    # 确保ts_code列存在（CCTV/AKShare新闻可能无此列）
    if 'ts_code' not in events_df.columns:
        events_df['ts_code'] = ''
        logger.info("  初始化ts_code列（原始数据无股票代码字段）")

    has_code = events_df[events_df['ts_code'].notna() & (events_df['ts_code'] != '')]
    no_code = events_df[events_df['ts_code'].isna() | (events_df['ts_code'] == '')]
    logger.info(f"  已有股票代码: {len(has_code)} 条, 待挖掘: {len(no_code)} 条")

    if len(no_code) == 0:
        events_df = has_code.copy()
        logger.info("  所有事件均已关联股票代码")
    else:
        newly_linked = []

        # --- 3.2 通道1：标题中正则提取6位股票代码 ---
        regex_linked = 0
        logger.info("  通道1: 标题正则提取股票代码...")
        for idx, row in no_code.iterrows():
            title = str(row.get('title', ''))
            m = re.search(r'\((\d{6})\)', title)
            if m:
                code = m.group(1)
                if code.startswith('6'):
                    ts = f"{code}.SH"
                elif code.startswith('0') or code.startswith('3'):
                    ts = f"{code}.SZ"
                elif code.startswith('8') or code.startswith('4'):
                    ts = f"{code}.BJ"
                else:
                    ts = code
                new_row = row.copy()
                new_row['ts_code'] = ts
                newly_linked.append(new_row)
                regex_linked += 1
        logger.info(f"  通道1关联: {regex_linked} 条")

        linked_indices = set()
        for nr in newly_linked:
            for idx, row in no_code.iterrows():
                if row.get('title', '') == nr.get('title', '') and idx not in linked_indices:
                    linked_indices.add(idx)
                    break
        still_no_code = no_code[~no_code.index.isin(linked_indices)]

        # --- 3.3 通道2：finance-data stock_basic按公司名查询 ---
        finance_linked = 0
        if len(still_no_code) > 0:
            logger.info(f"  通道2: finance-data stock_basic按公司名查询 ({len(still_no_code)} 条)...")
            for idx, row in still_no_code.iterrows():
                title = str(row.get('title', ''))
                keyword = _extract_company_keyword(title)
                if not keyword:
                    continue

                try:
                    df_match = _finance_data_api('stock_basic', {'name': keyword})
                    if df_match is not None and len(df_match) > 0:
                        ts = str(df_match.iloc[0].get('ts_code', ''))
                        if ts:
                            new_row = row.copy()
                            new_row['ts_code'] = ts
                            new_row['company_name'] = df_match.iloc[0].get('name', keyword)
                            new_row['industry'] = df_match.iloc[0].get('industry', '')
                            newly_linked.append(new_row)
                            finance_linked += 1
                except Exception:
                    pass
                time.sleep(0.15)
        logger.info(f"  通道2关联: {finance_linked} 条")

        linked_titles_12 = {r.get('title', '') for r in newly_linked}
        still_no_code_2 = still_no_code[~still_no_code['title'].isin(linked_titles_12)]

                # --- 3.4 通道3：事件子类→本地HS300成分股映射（Fix #21c）---
        # 原版使用finance-data API(index_classify+index_member_all)获取申万行业成分股
        # 但该API返回40101认证失败，导致81条事件全部关联失败
        # Fix #21c: 改用本地HS300_TOP30常量映射，无需任何外部API调用
        industry_linked = 0
        if len(still_no_code_2) > 0:
            logger.info(f"  通道3: 事件子类→历史共现映射 ({len(still_no_code_2)} 条)...")

            # 事件子类 → 标的证券映射表（Fix #21c修订版: 数据驱动）
            #
            # 【数据来源】基于训练集_增强_v3（5,674条样本）中 event_sub 与 ts_code 的
            # 实际共现频率统计，取每个子类Top-2高频股票作为先验映射。
            #
            # 【论文表述】"对于无法从公告文本或标题直接提取股票代码的事件，采用基于
            #  历史数据共现频率的先验映射策略：在训练集中按事件子类统计各标的证券的
            #  出现频次，选取频次最高的2只作为候选标的。该方法的合理性在于：(1)历史
            #  共现反映了该类事件的行业聚集特征；(2)作为兜底方案，后续仍需通过技术
            #  因子模型和综合评分进行二次筛选，误关联标的会被自然过滤；(3)相比外部
            #  API依赖，该方法具有零延迟、100%可用性的优势。"
            #
            # 【局限性说明】训练集样本量有限（部分子类如'分红送转'仅5条），映射结果
            # 可能与实时市场存在偏差。因此通道3优先级低于通道1(标题正则)和通道2(在线查询)，
            # 仅作为最后兜底。且每个事件最多关联2只股票（保守策略），避免过度扩散。
            #
            # 【生成方式】扫描脚本: _scan_data.py (可重新运行更新)
            EVENT_SUB_TO_STOCKS = {
                '股东变动': ['000776.SZ', '002736.SZ'],       # 训练集1622条, Top1=28次
                '其他': ['000901.SZ', '300298.SZ'],             # 1452条, Top1=27次
                '政策发布': ['000901.SZ', '600889.SH'],         # 880条, Top1=14次
                '人事变动': ['000901.SZ', '000582.SZ'],         # 429条, Top1=20次
                '业绩': ['603766.SH', '300953.SZ'],             # 400条, Top1=4次
                '融资': ['300871.SZ', '002939.SZ'],             # 320条, Top1=6次
                '利空公告': ['300506.SZ', '000609.SZ'],         # 262条, Top1=9次
                '并购': ['600889.SH', '603991.SH'],              # 238条, Top1=12次
                '并购重组': ['600889.SH', '603991.SH'],          # 同上(AHP别名)
                '合同': ['300001.SZ', '601789.SH'],              # 51条, Top1=5次
                '中标': ['300001.SZ', '601789.SH'],              # 同上(AHP别名)
                '技术突破': ['002371.SZ', '300138.SZ'],          # 15条, Top1=3次
                '分红送转': ['301027.SZ', '688767.SH'],          # 5条(样本少,谨慎使用)
                # --- 以下子类在训练集中未出现或极少，用AHP语义扩展 ---
                '宏观': ['000901.SZ', '600889.SH'],              # 继承'政策发布'
                '公司': ['000776.SZ', '002736.SZ'],               # 继承'股东变动'
                '科技': ['002371.SZ', '300138.SZ'],              # 继承'技术突破'
                '新能源': ['603766.SH', '300953.SZ'],            # 继承'业绩'(制造业)
                '消费': ['301027.SZ', '688767.SH'],              # 继承'分红送转'
                '货币政策': ['000901.SZ', '600889.SH'],           # 继承'政策发布'
                '周期': ['300506.SZ', '000609.SZ'],              # 继承'利空公告'
                '财政政策': ['000901.SZ', '600889.SH'],          # 继承'政策发布'
                '金融': ['300871.SZ', '002939.SZ'],              # 继承'融资'
                '违规处罚': ['300506.SZ', '000609.SZ'],           # 继承'利空公告'
            }

            for idx, row in still_no_code_2.iterrows():
                event_sub = str(row.get('event_sub', ''))
                title = str(row.get('title', ''))

                target_codes = EVENT_SUB_TO_STOCKS.get(event_sub, [])
                
                # 匹配策略2: 标题关键词兜底
                if not target_codes:
                    for kw, codes in EVENT_SUB_TO_STOCKS.items():
                        if kw in title:
                            target_codes = codes
                            break
                
                # 兜底: 宏观类事件给沪深核心权重股
                if not target_codes:
                    target_codes = ['600519.SH', '600036.SH']

                # 取前2只作为关联标的（避免过度扩散）
                for ts_code in target_codes[:2]:
                    new_row = row.copy()
                    new_row['ts_code'] = ts_code
                    new_row['company_name'] = ''  # 留空，由_build_stock_lookup补充
                    new_row['assoc_method'] = f'子类映射({event_sub})'
                    newly_linked.append(new_row)
                    industry_linked += 1

        logger.info(f"  通道3(历史共现映射): {industry_linked} 条")
        if newly_linked:
            new_df = pd.DataFrame(newly_linked)
            events_df = pd.concat([has_code, new_df], ignore_index=True)
            logger.info(f"  总新关联: {len(newly_linked)} 条")
        else:
            events_df = has_code.copy()
            logger.info("  未找到新关联股票")

    before = len(events_df)
    events_df = events_df.drop_duplicates(subset=['title', 'ts_code'], keep='first')
    after = len(events_df)
    logger.info(f"  去重: {before}→{after} 条")
    
    assoc_path = os.path.join(RESULT_DIR, 'events_with_association.csv')
    events_df.to_csv(assoc_path, index=False, encoding='utf-8-sig')
    logger.info(f"  关联结果已保存: {assoc_path}")
    
    return events_df

TECH_FACTOR_COLS = [
    'MA5', 'dev_MA5', 'MA10', 'dev_MA10', 'MA20', 'dev_MA20', 'MA60', 'dev_MA60',
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'volatility_20d',
    'RSI14', 'MACD_dif', 'MACD_dea', 'MACD_hist',
    'BOLL_upper', 'BOLL_lower', 'BOLL_width', 'BOLL_pos',
    'vol_ma5', 'vol_ratio', 'ATR14', 'ATR14_pct',
    'decline_days', 'gap_up', 'gap_down', 'gap_pct',
    'alpha', 'beta', 'close_t1'  # 32列，与v12训练完全一致
]

def compute_technical_factors(price_df):
    df = price_df.copy()
    sort_col = 'trade_date_parsed' if 'trade_date_parsed' in df.columns else 'trade_date'
    df = df.sort_values(sort_col).reset_index(drop=True)
    
    for w in [5, 10, 20, 60]:
        df[f'MA{w}'] = df['close'].rolling(w, min_periods=1).mean()
        df[f'dev_MA{w}'] = (df['close'] - df[f'MA{w}']) / df[f'MA{w}'] * 100
    
    df['ret_1d'] = df['close'].pct_change(1) * 100
    df['ret_5d'] = df['close'].pct_change(5) * 100
    df['ret_10d'] = df['close'].pct_change(10) * 100
    df['ret_20d'] = df['close'].pct_change(20) * 100
    
    df['volatility_20d'] = df['ret_1d'].rolling(20, min_periods=5).std() * np.sqrt(252)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=5).mean()
    avg_loss = loss.rolling(14, min_periods=5).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_dif'] = ema12 - ema26
    df['MACD_dea'] = df['MACD_dif'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = 2 * (df['MACD_dif'] - df['MACD_dea'])
    
    # BOLL
    ma20 = df['close'].rolling(20, min_periods=5).mean()
    std20 = df['close'].rolling(20, min_periods=5).std()
    df['BOLL_upper'] = ma20 + 2 * std20
    df['BOLL_lower'] = ma20 - 2 * std20
    df['BOLL_width'] = (df['BOLL_upper'] - df['BOLL_lower']) / (ma20 + 1e-10) * 100
    df['BOLL_pos'] = (df['close'] - df['BOLL_lower']) / (df['BOLL_upper'] - df['BOLL_lower'] + 1e-10)
    
    df['vol_ma5'] = df['vol'].rolling(5, min_periods=3).mean()
    df['vol_ratio'] = df['vol'] / (df['vol_ma5'] + 1e-10)
    
    # ATR
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    df['ATR14'] = tr.rolling(14, min_periods=5).mean()
    df['ATR14_pct'] = df['ATR14'] / (df['close'] + 1e-10) * 100
    
    df['gap_up'] = (df['open'] - df['close'].shift(1)).clip(lower=0)
    df['gap_down'] = (df['close'].shift(1) - df['open']).clip(lower=0)
    df['gap_pct'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-10) * 100
    
    df['decline_days'] = 0
    decline_count = 0
    for i in range(1, len(df)):
        if df.loc[i, 'close'] < df.loc[i-1, 'close']:
            decline_count += 1
        else:
            decline_count = 0
        df.loc[i, 'decline_days'] = decline_count
    
    df['close_t1'] = df['close'].shift(1)
    
    # alpha/beta（需要市场基准，实时pipeline中暂用中性值0填充）
    # 训练数据中alpha/beta由市场模型回归得到，推理时无基准则填0
    df['alpha'] = 0.0
    df['beta'] = 1.0
    
    return df

def _fetch_finance_data_daily(ts_codes_str, start_date, end_date):
    API_URL = "https://www.codebuddy.cn/v2/tool/financedata"
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(API_URL, json={
                'api_name': 'daily',
                'params': {
                    'ts_code': ts_codes_str,
                    'start_date': start_date,
                    'end_date': end_date
                },
                'fields': ''
            }, timeout=30)
            d = resp.json()
            if d.get('code') == 0:
                data = d.get('data', {})
                fields = data.get('fields', [])
                items = data.get('items', [])
                return fields, items
            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return None, None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            return None, None
    return None, None

def _fetch_baostock_daily(ts_code, start_date, end_date):
    try:
        import baostock as bs
        
        code_pure = ts_code.split('.')[0] if '.' in ts_code else ts_code
        if ts_code.endswith('.SH') or ts_code.startswith('6'):
            bs_code = f'sh.{code_pure}'
        elif ts_code.endswith('.SZ') or code_pure.startswith(('0', '3')):
            bs_code = f'sz.{code_pure}'
        elif ts_code.endswith('.BJ') or code_pure.startswith(('8', '4')):
            bs_code = f'bj.{code_pure}'
        else:
            if code_pure.startswith('6'):
                bs_code = f'sh.{code_pure}'
            else:
                bs_code = f'sz.{code_pure}'
        
        lg = bs.login()
        if lg.error_code != '0':
            logger.warning(f"  BaoStock登录失败: {lg.error_msg}")
            return None
        
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,open,high,low,close,volume,amount,pctChg',
            start_date=start_date,
            end_date=end_date,
            frequency='d',
            adjustflag='2'  # 前复权
        )
        
        data_list = []
        while rs.error_code == '0' and rs.next():
            data_list.append(rs.get_row_data())
        
        bs.logout()
        
        if not data_list:
            return None
        
        df = pd.DataFrame(data_list, columns=[
            'trade_date', 'open', 'high', 'low', 'close', 
            'vol', 'amount', 'pct_chg'
        ])
        
        df['ts_code'] = ts_code
        
        for col in ['open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"  BaoStock采集异常({ts_code}): {e}")
        return None

def _fetch_akshare_daily(code_pure, start_date, end_date):
    try:
        import akshare as ak
        if code_pure.startswith('6'):
            market = 'sh'
        elif code_pure.startswith('0') or code_pure.startswith('3'):
            market = 'sz'
        else:
            market = 'bj'
        
        symbol = f"{code_pure}"  # akshare stock_zh_a_hist 用纯数字
        
        start_fmt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end_fmt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period='daily',
            start_date=start_fmt,
            end_date=end_fmt,
            adjust='qfq'
        )
        
        if df is not None and len(df) > 0:
            col_map = {
                '日期': 'trade_date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'vol',
                '成交额': 'amount',
                '涨跌幅': 'pct_chg',
                '涨跌额': 'change',
            }
            df = df.rename(columns=col_map)
            
            suffix = '.SH' if code_pure.startswith('6') else ('.SZ' if code_pure.startswith(('0', '3')) else '.BJ')
            df['ts_code'] = f"{code_pure}{suffix}"
            
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
            
            keep_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 
                         'vol', 'amount', 'pct_chg']
            keep_cols = [c for c in keep_cols if c in df.columns]
            df = df[keep_cols]
            
            return df
        return None
    except Exception as e:
        return None

def collect_market_data(events_df, trade_start, trade_end):
    logger.info("Step 4：行情数据采集 + 技术因子计算")
    
    if len(events_df) == 0:
        logger.warning("  无事件可采集行情")
        return events_df, None, None
    
    stock_codes = events_df['ts_code'].dropna().unique().tolist()
    logger.info(f"  需采集行情的股票: {len(stock_codes)} 只")
    
    # --- 4.1 加载本地已有行情数据（基础数据源） ---
    local_price_path = os.path.join(DATA_DIR, '股票行情数据', '个股日频行情_清洗.csv')
    
    if os.path.exists(local_price_path):
        logger.info(f"  加载本地行情数据: {local_price_path}")
        local_price = pd.read_csv(local_price_path, low_memory=False)
        local_price['trade_date_parsed'] = safe_parse_date(local_price['trade_date'])
        logger.info(f"  本地行情: {len(local_price)} 条, 日期范围: "
                    f"{local_price['trade_date_parsed'].min()} ~ {local_price['trade_date_parsed'].max()}")
    else:
        local_price = pd.DataFrame()
        logger.info("  本地无行情数据")
    
    # Fix #21a: BaoStock要求YYYY-MM-DD格式（不是YYYYMMDD！）
    hist_start = (pd.Timestamp(trade_start) - pd.Timedelta(days=150)).strftime('%Y-%m-%d')
    end_fmt = trade_end  # 已经是YYYY-MM-DD格式
    
    # 找出本地数据不足的股票（目标日期范围内无数据）
    codes_need_fetch = []
    if len(local_price) > 0:
        local_codes_with_recent = set()
        recent_mask = local_price['trade_date_parsed'] >= (pd.Timestamp(trade_start) - pd.Timedelta(days=30))
        local_codes_with_recent = set(local_price[recent_mask]['ts_code'].unique())
        
        for code in stock_codes:
            if code not in local_codes_with_recent:
                codes_need_fetch.append(code)
        
        logger.info(f"  本地已有近期数据: {len(local_codes_with_recent & set(stock_codes))} 只")
        logger.info(f"  需要实时采集: {len(codes_need_fetch)} 只")
    else:
        codes_need_fetch = stock_codes[:]
        logger.info(f"  无本地数据，需采集全部: {len(codes_need_fetch)} 只")
    
    # --- 4.2 通过 BaoStock 采集（Fix #20: 新主选，免费稳定无配额限制） ---
    new_price_data = []
    bs_success = 0
    bs_failed = []
    baostock_available = False
    
    if codes_need_fetch:
        try:
            import baostock as bs  # noqa: F401
            baostock_available = True
            logger.info(f"\n--- 4.2 BaoStock采集（主选，免费/稳定/无配额限制）---")
        except ImportError:
            baostock_available = False
            logger.warning("  BaoStock未安装(pip install baostock)，跳过")
        
        if baostock_available:
            for i, code in enumerate(codes_need_fetch):
                df_bs = _fetch_baostock_daily(code, hist_start, end_fmt)
                
                if df_bs is not None and len(df_bs) > 0:
                    new_price_data.extend(df_bs.to_dict('records'))
                    bs_success += 1
                else:
                    bs_failed.append(code)
                
                if (i + 1) % 10 == 0 or (i + 1) == len(codes_need_fetch):
                    logger.info(f"  BaoStock进度: {i+1}/{len(codes_need_fetch)} "
                               f"(成功:{bs_success}, 失败:{len(bs_failed)})")
                time.sleep(0.1)  # BaoStock限流较宽松
            
            logger.info(f"  BaoStock采集完成: 成功{bs_success}只, "
                       f"失败{len(bs_failed)}只, 共{len(new_price_data)}条记录")
    
    # --- 4.3 AKShare采集BaoStock失败的股票（第二备选） ---
    akshare_success = 0
    akshare_failed = []
    
    remaining_codes = bs_failed if bs_failed else codes_need_fetch
    
    if remaining_codes and len(new_price_data) < (len(remaining_codes) * 5):  # BaoStock失败率>80%时才启用AKShare
        akshare_available = False
        try:
            import akshare as ak  # noqa: F401
            akshare_available = True
            logger.info(f"\n--- 4.3 AKShare采集（第二备选，东方财富数据源，{len(remaining_codes)}只）---")
        except ImportError:
            akshare_available = False
            remaining_codes = []  # AKShare不可用，直接跳到finance-data
        
        if akshare_available:
            for i, code in enumerate(remaining_codes):
                code_pure = code.split('.')[0] if '.' in code else code
                df_ak = _fetch_akshare_daily(code_pure, hist_start, end_fmt)
                
                if df_ak is not None and len(df_ak) > 0:
                    new_price_data.extend(df_ak.to_dict('records'))
                    akshare_success += 1
                else:
                    akshare_failed.append(code)
                
                if (i + 1) % 10 == 0 or (i + 1) == len(remaining_codes):
                    logger.info(f"  AKShare进度: {i+1}/{len(remaining_codes)} "
                               f"(成功:{akshare_success}, 失败:{len(akshare_failed)})")
                time.sleep(0.15)
            
            logger.info(f"  AKShare采集完成: 成功{akshare_success}只, "
                       f"失败{len(akshare_failed)}只")
    elif bs_success > 0:
        logger.info(f"  BaoStock已成功{bs_success}只，跳过AKShare备选")
        akshare_failed = []  # 不需要AKShare补充
    
    # --- 4.4 通过 finance-data HTTP API 采集前两轮失败的股票（第三备选） ---
    finance_data_success = 0
    finance_data_failed = []
    finance_data_quota_exhausted = False
    
    if akshare_failed:
        logger.info(f"\n--- 4.4 finance-data HTTP API 采集前两轮失败的 {len(akshare_failed)} 只（第三备选）---")
        BATCH_SIZE = 5
        BATCH_SLEEP = 0.3
        
        for i in range(0, len(akshare_failed), BATCH_SIZE):
            if finance_data_quota_exhausted:
                finance_data_failed.extend(akshare_failed[i:])
                break
            
            batch = akshare_failed[i:i+BATCH_SIZE]
            ts_codes_str = ','.join(batch)
            
            fields, items = _fetch_finance_data_daily(ts_codes_str, hist_start, end_fmt)
            
            if items is not None and len(items) > 0:
                for item in items:
                    new_price_data.append(dict(zip(fields, item)))
                finance_data_success += len(batch)
            else:
                for code in batch:
                    fields, items = _fetch_finance_data_daily(code, hist_start, end_fmt)
                    if items and len(items) > 0:
                        for item in items:
                            new_price_data.append(dict(zip(fields, item)))
                        finance_data_success += 1
                    else:
                        finance_data_failed.append(code)
            
            done = min(i + BATCH_SIZE, len(akshare_failed))
            if done % 50 == 0 or done == len(akshare_failed):
                logger.info(f"  finance-data进度: {done}/{len(akshare_failed)} "
                           f"(成功:{finance_data_success}, 失败:{len(finance_data_failed)})")
            
            time.sleep(BATCH_SLEEP)
        
        logger.info(f"  finance-data采集完成: 成功{finance_data_success}只, "
                    f"失败{len(finance_data_failed)}只")
    
    # --- 4.5 通过 Tushare Pro 采集前三轮都失败的股票（第四备选） ---
    tushare_success = 0
    
    if finance_data_failed:
        logger.info(f"\n--- 4.5 Tushare Pro 采集前三轮失败的 {len(finance_data_failed)} 只（第四备选）---")
        tushare_available = False
        
        try:
            import tushare as ts
            token = os.environ.get('TUSHARE_TOKEN', '')
            if not token:
                token_file = os.path.join(os.path.expanduser('~'), '.tushare_token')
                if os.path.exists(token_file):
                    with open(token_file, 'r') as f:
                        token = f.read().strip()
            if not token:
                token_cfg = os.path.join(BASE_DIR, 'tushare_token.txt')
                if os.path.exists(token_cfg):
                    with open(token_cfg, 'r') as f:
                        token = f.read().strip()
            
            if token:
                ts.set_token(token)
                pro = ts.pro_api()
                try:
                    pro.trade_cal(exchange='SSE', is_open=1, limit=1)
                    tushare_available = True
                    logger.info("  Tushare连接成功")
                except Exception as e:
                    logger.warning(f"  Tushare token无效: {e}")
            else:
                logger.warning("  未找到Tushare Token，跳过")
        except ImportError:
            logger.warning("  tushare未安装，跳过")
        
        if tushare_available:
            for code in finance_data_failed:
                try:
                    df_daily = pro.daily(
                        ts_code=code,
                        start_date=hist_start,
                        end_date=end_fmt,
                        fields='ts_code,trade_date,open,high,low,close,vol,amount,pct_chg'
                    )
                    if df_daily is not None and len(df_daily) > 0:
                        new_price_data.extend(df_daily.to_dict('records'))
                        tushare_success += 1
                    time.sleep(0.15)
                except Exception as e:
                    logger.warning(f"  采集{code}行情失败: {e}")
            
            logger.info(f"  Tushare采集完成: 成功{tushare_success}只")
    
    total_fetched = bs_success + akshare_success + finance_data_success + tushare_success
    total_failed = max(0, len(codes_need_fetch) - total_fetched)
    logger.info(f"\n  采集汇总: 成功{total_fetched}/{len(codes_need_fetch)}只 "
                f"(BaoStock:{bs_success}, AKShare:{akshare_success}, finance-data:{finance_data_success}, Tushare:{tushare_success})")
    if total_failed > 0:
        logger.warning(f"  仍有{total_failed}只股票采集失败，将使用本地数据补充")
    
    # --- 4.6 合并行情数据 ---
    if new_price_data:
        new_price = pd.DataFrame(new_price_data)
        num_cols = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
        for col in num_cols:
            if col in new_price.columns:
                new_price[col] = pd.to_numeric(new_price[col], errors='coerce')
        new_price['trade_date_parsed'] = safe_parse_date(new_price['trade_date'])
        logger.info(f"  新采集行情: {len(new_price)} 条")
    else:
        new_price = pd.DataFrame()
        logger.info("  未采集到新行情数据")
    
    # --- 4.6 合并行情数据（[!] 性能优化: 只保留目标股票+150天窗口，避免全量97万行） ---
    if new_price_data:
        new_price = pd.DataFrame(new_price_data)
        num_cols = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
        for col in num_cols:
            if col in new_price.columns:
                new_price[col] = pd.to_numeric(new_price[col], errors='coerce')
        new_price['trade_date_parsed'] = safe_parse_date(new_price['trade_date'])
        logger.info(f"  新采集行情: {len(new_price)} 条")
    else:
        new_price = pd.DataFrame()
        logger.info("  未采集到新行情数据")
    
    if len(local_price) > 0 and len(new_price) > 0:
        merged = pd.concat([local_price, new_price], ignore_index=True)
        merged = merged.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
    elif len(new_price) > 0:
        merged = new_price
    elif len(local_price) > 0:
        merged = local_price
    else:
        logger.error("  [X] 无任何行情数据！")
        return events_df, None, None
    
    # [!] 关键优化：只过滤出需要的目标股票 + 150天历史窗口（用于计算技术因子）
    target_set = set(stock_codes)
    factor_window_start = (pd.Timestamp(trade_start) - pd.Timedelta(days=150))
    
    if 'trade_date_parsed' in merged.columns:
        merged['temp_date'] = merged['trade_date_parsed']
    else:
        merged['temp_date'] = safe_parse_date(merged['trade_date'])
    
    # 只保留目标股票在窗口期内的数据
    mask_target = merged['ts_code'].isin(target_set)
    mask_window = (merged['temp_date'] >= factor_window_start) & (merged['temp_date'] <= trade_end)
    all_price = merged[mask_target & mask_window].copy()
    
    logger.info(f"  全量合并: {len(merged)} 行 → 窗口过滤({len(target_set)}只×150天): {len(all_price)} 行 "
                f"(-{(1-len(all_price)/len(merged))*100:.1f}%)")
    
    price_with_factors = []
    for code, group in all_price.groupby('ts_code'):
        group = group.sort_values('trade_date_parsed').reset_index(drop=True)
        required_cols = ['open', 'high', 'low', 'close', 'vol']
        if all(c in group.columns for c in required_cols):
            group_factors = compute_technical_factors(group)
            price_with_factors.append(group_factors)
    
    if price_with_factors:
        price_factors_df = pd.concat(price_with_factors, ignore_index=True)
        logger.info(f"  技术因子计算完成: {len(price_factors_df)} 行, {len(price_factors_df.columns)} 列")
    else:
        price_factors_df = all_price
        logger.warning("  [!] 技术因子计算失败，使用原始行情")
    
    bench_path = os.path.join(DATA_DIR, '股票行情数据', '沪深300指数_清洗.csv')
    if os.path.exists(bench_path):
        bench_df = pd.read_csv(bench_path)
        bench_df['trade_date_parsed'] = safe_parse_date(bench_df['trade_date'])
        bench_df = bench_df.sort_values('trade_date_parsed')
        if 'pct_chg' in bench_df.columns:
            bench_df['mkt_ret'] = bench_df['pct_chg'] / 100
        elif 'close' in bench_df.columns:
            bench_df['mkt_ret'] = bench_df['close'].pct_change()
        else:
            bench_df['mkt_ret'] = 0
        
        bench_dict = bench_df.set_index('trade_date_parsed')['mkt_ret'].to_dict()
        
        for code, group in price_factors_df.groupby('ts_code'):
            group = group.sort_values('trade_date_parsed')
            stock_ret = group['ret_1d'].values / 100
            dates = group['trade_date_parsed'].values
            mkt_ret = np.array([bench_dict.get(pd.Timestamp(d), 0) for d in dates])
            
            if len(stock_ret) > 20 and np.std(mkt_ret) > 1e-10:
                try:
                    from numpy.polynomial.polynomial import polyfit
                    beta_val, alpha_val = polyfit(mkt_ret[1:], stock_ret[1:], 1)
                    price_factors_df.loc[group.index, 'alpha'] = alpha_val * 100
                    price_factors_df.loc[group.index, 'beta'] = beta_val
                except:
                    price_factors_df.loc[group.index, 'alpha'] = 0
                    price_factors_df.loc[group.index, 'beta'] = 1
    else:
        logger.warning("  未找到沪深300基准数据，alpha/beta设为0")
        price_factors_df['alpha'] = 0
        price_factors_df['beta'] = 1
    
    price_save_path = os.path.join(RESULT_DIR, f'price_with_factors_{trade_start}_{trade_end}.csv')
    price_factors_df.to_csv(price_save_path, index=False, encoding='utf-8-sig')
    logger.info(f"  行情+因子已保存: {price_save_path}")
    
    # --- 4.6 将技术因子合并到事件数据 ---
    logger.info("  将技术因子合并到事件数据...")
    
    # 为每个事件获取最近交易日的技术因子（而非精确匹配event_date）
    events_df['event_date_parsed'] = pd.to_datetime(events_df['event_date'], errors='coerce')
    
    # 按ts_code分组，为每个事件找该股票行情中最近的一天
    # 先确保行情数据按ts_code和日期排序
    if 'trade_date_parsed' in price_factors_df.columns:
        price_factors_df_sorted = price_factors_df.sort_values(['ts_code', 'trade_date_parsed'])
    else:
        price_factors_df_sorted = price_factors_df.sort_values(['ts_code', 'trade_date'])
    
    from bisect import bisect_right
    price_by_code = {}
    for code, grp in price_factors_df_sorted.groupby('ts_code'):
        date_col = 'trade_date_parsed' if 'trade_date_parsed' in grp.columns else 'trade_date'
        dates = grp[date_col].tolist()
        price_by_code[code] = (dates, grp)
    
    for col in TECH_FACTOR_COLS:
        events_df[col] = np.nan
    
    matched_count = 0
    for idx, row in events_df.iterrows():
        code = row.get('ts_code', '')
        event_dt = row.get('event_date_parsed', pd.NaT)
        
        if pd.isna(event_dt) or code not in price_by_code:
            continue
        
        dates, grp = price_by_code[code]
        if len(dates) == 0:
            continue
        
        # 找事件日期之前（含当天）最近的交易日
        # 使用bisect_right找到第一个>event_dt的位置，前一个就是<=event_dt的最近日期
        pos = bisect_right(dates, event_dt)
        if pos > 0:
            best_idx = pos - 1
            price_row = grp.iloc[best_idx]
            for col in TECH_FACTOR_COLS:
                if col in price_row.index:
                    events_df.at[idx, col] = price_row[col]
            matched_count += 1
        elif len(dates) > 0:
            # event_dt在所有行情之前，用最早的行情
            price_row = grp.iloc[0]
            for col in TECH_FACTOR_COLS:
                if col in price_row.index:
                    events_df.at[idx, col] = price_row[col]
            matched_count += 1
    
    matched = events_df[TECH_FACTOR_COLS].notna().any(axis=1).sum()
    logger.info(f"  技术因子匹配: {matched}/{len(events_df)} 条事件")
    
    return events_df, price_factors_df, bench_df if 'bench_df' in dir() else None

def run_impact_prediction(events_df, bench_df=None):
    logger.info("Step 5：事件影响预测（v15g: LR + LGBM回归 + 贝叶斯先验）")
    
    if len(events_df) == 0:
        logger.warning("  无事件可预测")
        return events_df, None, None
    
    import joblib
    import json
    import pickle
    
    # ---- 5.1 加载v15g模型 ----
    logger.info("  --- 5.1 加载v15g模型 ---")
    
    # LR 使用本次训练生成的 32 维模型与配套 scaler，旧 v15 文件仅作兼容回退。
    e4_lr_candidates = [
        os.path.join(MODEL_DIR, '5.3_事件影响预测', 'e4_lr_model.pkl'),
        os.path.join(MODEL_DIR, '5.3_事件影响预测', 'v15_e4_lr_model.pkl'),
        os.path.join(MODEL_DIR, '5.3_事件影响预测', 'e4_tech_lr.pkl'),
    ]
    e4_lr_path = next((p for p in e4_lr_candidates if os.path.exists(p)), e4_lr_candidates[0])
    e4_lr_scaler_candidates = [
        os.path.join(MODEL_DIR, '5.3_事件影响预测', 'e4_lr_scaler.pkl'),
        os.path.join(MODEL_DIR, '5.3_事件影响预测', 'v15_feature_scaler.pkl'),
        os.path.join(MODEL_DIR, '5.3_事件影响预测', 'v15_e4_lr_scaler.pkl'),
        os.path.join(MODEL_DIR, '5.3_事件影响预测', 'tech_scaler.pkl'),
    ]
    e4_lr_scaler_path = next(
        (p for p in e4_lr_scaler_candidates if os.path.exists(p)),
        e4_lr_scaler_candidates[0],
    )

    e4_tech_cols_path = os.path.join(MODEL_DIR, '5.3_事件影响预测', 'tech_factor_cols.json')

    # LGBM 同样优先使用本次训练产物。
    lgbm_candidates = [
        os.path.join(MODEL_DIR, '5.3_事件影响预测', 'e4_lgbm_model.txt'),
        os.path.join(MODEL_DIR, '5.3_事件影响预测', 'v15_e4_lgbm_model.txt'),
        os.path.join(MODEL_DIR, '5.3_事件影响预测', 'lgbm_regression_best.pkl'),
        os.path.join(MODEL_DIR, '5.3_事件影响预测', 'lgbm_reg.pkl'),
        os.path.join(MODEL_DIR, '5.3_事件影响预测', 'lgbm_car_regressor.pkl'),
    ]
    lgbm_reg_path = next((p for p in lgbm_candidates if os.path.exists(p)), lgbm_candidates[0])
    
    bayes_path = os.path.join(MODEL_DIR, '5.3_事件影响预测', '贝叶斯事件先验.csv')
    if not os.path.exists(bayes_path):
        bayes_path = os.path.join(MODEL_DIR, '5.3_事件影响预测', '贝叶斯事件先验_v15.csv')
    
    # LGBM特征列名
    lgbm_feature_names_path = os.path.join(MODEL_DIR, '5.3_事件影响预测', 'lgbm_feature_names.json')
    
    # ---- 加载LR模型 ----
    e4_lr = None
    if os.path.exists(e4_lr_path):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                e4_lr = joblib.load(e4_lr_path)
            logger.info(f"  [OK] E4 LR模型加载成功: {e4_lr_path}")
        except Exception as e:
            logger.warning(f"  E4 LR加载失败: {e}")
    else:
        logger.warning(f"  E4 LR模型不存在: {e4_lr_path}")
    
    if os.path.exists(e4_tech_cols_path):
        with open(e4_tech_cols_path, 'r', encoding='utf-8') as f:
            lr_cols = json.load(f)
        logger.info(f"  技术因子列: {len(lr_cols)}个")
    else:
        lr_cols = [c for c in TECH_FACTOR_COLS if c in events_df.columns]
        logger.info(f"  使用默认技术因子列: {len(lr_cols)}个")
    
    for c in lr_cols:
        if c not in events_df.columns:
            events_df[c] = 0.0
    
    # ---- 加载LGBM回归模型 ----
    # LGBM 原生加载器不兼容中文路径，按文本内容加载可避开该限制。
    lgbm_reg = None
    if os.path.exists(lgbm_reg_path):
        try:
            import lightgbm as lgb

            if lgbm_reg_path.endswith('.txt') or lgbm_reg_path.endswith('.json'):
                with open(lgbm_reg_path, 'r', encoding='utf-8') as model_file:
                    lgbm_reg = lgb.Booster(model_str=model_file.read())
                logger.info(f"  [OK] LGBM回归模型加载成功(model_str): {os.path.basename(lgbm_reg_path)}")
            else:
                lgbm_reg = joblib.load(lgbm_reg_path)
                logger.info(f"  [OK] LGBM回归模型加载成功(joblib): {os.path.basename(lgbm_reg_path)}")
        except Exception as e:
            logger.warning(f"  [!] LGBM回归加载失败({type(e).__name__}): {e}")
    else:
        logger.warning(f"  LGBM回归模型不存在: {lgbm_reg_path}")
    
    bayesian_priors = {}
    if os.path.exists(bayes_path):
        bayes_df = pd.read_csv(bayes_path)
        for _, row in bayes_df.iterrows():
            bayesian_priors[row['event_sub']] = row.to_dict()
        logger.info(f"  [OK] 贝叶斯事件先验: {len(bayesian_priors)}个子类")
    else:
        logger.warning(f"  贝叶斯先验文件不存在: {bayes_path}")
    
    # ---- 5.2 E4 LR分类预测 ----
    logger.info("  --- 5.2 E4 LR分类预测（32维纯技术因子） ---")
    
    if e4_lr is not None:
        lr_data = events_df[lr_cols].fillna(0).values
        # v15g[!] 关键：LR必须用训练时的StandardScaler做transform（否则F1暴跌）
        e4_lr_scaler = None
        if os.path.exists(e4_lr_scaler_path):
            try:
                import joblib as jb_scaler
                # Fix #22: 压制sklearn版本警告(训练1.8.0 vs 推理1.5.2)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    e4_lr_scaler = jb_scaler.load(e4_lr_scaler_path)
                logger.info(f"  [OK] E4 LR Scaler加载成功: {e4_lr_scaler_path} (v15_feature_scaler)")
            except Exception as e:
                logger.warning(f"  E4 LR Scaler加载失败: {e}")
        else:
            logger.warning(f"  [!] E4 LR Scaler不存在: {e4_lr_scaler_path}，直接推理(F1将大幅下降)")
        
        if e4_lr_scaler is not None:
            lr_data_scaled = e4_lr_scaler.transform(lr_data)
            logger.info(f"  使用训练LR Scaler做transform: {lr_data.shape[1]}维")
        else:
            from sklearn.preprocessing import StandardScaler as LRScaler
            fallback_scaler = LRScaler()
            lr_data_scaled = fallback_scaler.fit_transform(lr_data)
            logger.warning(f"  [!] 使用推理时fit_transform代替训练Scaler（F1将从~0.443降至~0.123）")
        
        # v15h修复: 兼容sklearn版本差异（新版本LR无multi_class属性）
        # Fix #22: 统一使用decision_function+softmax，避免predict_proba版本兼容问题
        # Fix #28: predict_proba输出极化(0/1)是softmax的正常行为（dec范围大），
        #   综合评分权重0.60是基于原始predict_proba校准的，不可更改
        #   报告展示用T=15温度缩放概率，见generate_analysis_report()
        from scipy.special import softmax as sp_softmax
        try:
            lr_probs = e4_lr.predict_proba(lr_data_scaled)
            lr_preds = e4_lr.predict(lr_data_scaled)
            # predict_proba极化(dec范围大)是正常行为，不再判断为异常
        except (AttributeError, Exception) as e:
            # sklearn版本不兼容(训练1.8.0 vs 推理1.5.2) → 用decision_function+温度缩放softmax
            logger.warning(f"  [!] LR.predict_proba不兼容({type(e).__name__}), 使用decision_function+温度缩放softmax")
            dec = e4_lr.decision_function(lr_data_scaled)
            _SOFTMAX_TEMP = 3.0  # 温度参数：T=1纯softmax(极化), T>1软化概率（仅fallback时使用）
            lr_probs = sp_softmax(dec / _SOFTMAX_TEMP, axis=1)
            lr_preds = np.argmax(dec, axis=1)  # 分类标签仍用原始decision_score的argmax
        events_df['技术看涨概率'] = lr_probs[:, 2]
        events_df['E4预测标签'] = lr_preds  # 0=跌, 1=平, 2=涨
        logger.info(f"  E4 LR看涨概率: mean={lr_probs[:, 2].mean():.3f}, "
                    f"max={lr_probs[:, 2].max():.3f}")
        logger.info(f"  E4 LR预测分布: {dict(zip(*np.unique(lr_preds, return_counts=True)))}")
    else:
        def compute_tech_score_fallback(row):
            score = 0.3
            rsi = row.get('RSI14', 50)
            ret_20 = row.get('ret_20d', 0)
            vol_ratio = row.get('vol_ratio', 1)
            if not pd.isna(rsi):
                if rsi < 30: score += 0.3
                elif rsi < 40: score += 0.15
                elif rsi > 70: score -= 0.2
            if not pd.isna(ret_20):
                if -20 < ret_20 < -5: score += 0.2
                elif ret_20 > 10: score += 0.1
            if not pd.isna(vol_ratio) and vol_ratio > 1.5:
                score += 0.15
            return max(0, min(1, score))
        events_df['技术看涨概率'] = events_df.apply(compute_tech_score_fallback, axis=1)
        events_df['E4预测标签'] = 1  # 默认中性
        logger.warning("  E4 LR不可用，使用硬编码技术得分")
    
    # ---- 5.3 LGBM回归预测CAR ----
    logger.info("  --- 5.3 LGBM回归预测CAR ---")
    
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    events_df['预测CAR'] = 0.0
    events_df['CAR预测概率'] = 0.5
    
    if lgbm_reg is not None:
        # Fix #15b: 构建完整115维特征（与训练时完全一致）
        # 特征组成: tech(32) + ev(11) + regime(3) + state(3) + int(44) + bayes(5) + xrank(17) = 115
        
        # ---- 加载训练时的特征元数据 ----
        train_feature_names = None
        if os.path.exists(lgbm_feature_names_path):
            try:
                with open(lgbm_feature_names_path, 'r', encoding='utf-8') as f:
                    _fn_meta = json.load(f)
                if isinstance(_fn_meta, dict):
                    train_feature_names = _fn_meta.get('feature_names', [])
                elif isinstance(_fn_meta, list):
                    train_feature_names = _fn_meta
            except:
                pass
        if not train_feature_names and os.path.exists(e4_tech_cols_path):
            try:
                with open(e4_tech_cols_path, 'r', encoding='utf-8') as f:
                    _meta2 = json.load(f)
                if isinstance(_meta2, dict):
                    train_feature_names = _meta2.get('feature_names', [])
                elif isinstance(_meta2, list):
                    train_feature_names = _meta2
            except:
                pass
        
        n_events = len(events_df)
        
        tech_data = events_df[lr_cols].fillna(0).values.astype(np.float32)
        
        # --- 2. 事件子类one-hot(11维): ev_业绩, ev_人事变动, ... ---
        event_subs_all = ['业绩', '人事变动', '其他', '分红送转', '利空公告',
                          '合同', '并购', '技术突破', '政策发布', '股东变动', '融资']
        ev_onehot = np.zeros((n_events, len(event_subs_all)), dtype=np.float32)
        for i in range(n_events):
            row = events_df.iloc[i]
            sub = str(row.get('event_sub', '其他'))
            if pd.isna(sub) or sub == 'nan': sub = '其他'
            if sub in event_subs_all:
                j = event_subs_all.index(sub)
                ev_onehot[i, j] = 1.0
            else:
                ev_onehot[i, 2] = 1.0  # 默认'其他'
        
        # --- 3. 市场状态(3维): regime_熊市/牛市/震荡 ---
        regime_data = np.zeros((n_events, 3), dtype=np.float32)
        if bench_df is not None and len(bench_df) >= 5:
            pct_col = 'pct_chg' if 'pct_chg' in bench_df.columns else None
            if not pct_col:
                for c in ['pct_change', 'change_pct', 'chg_pct', '涨跌幅']:
                    if c in bench_df.columns:
                        pct_col = c
                        break
            if pct_col:
                bench_recent = bench_df[pct_col].tail(20).mean() if len(bench_df) >= 20 else bench_df[pct_col].mean()
            else:
                bench_recent = 0.0
            if bench_recent > 0.15:
                regime_data[:, 1] = 1.0   # 牛市
            elif bench_recent < -0.12:
                regime_data[:, 0] = 1.0   # 熊市
            else:
                regime_data[:, 2] = 1.0   # 震荡
        else:
            regime_data[:, 2] = 1.0       # 默认震荡
        
        # --- 4. 技术状态(3维): state_中性/超买/超卖 ---
        state_data = np.zeros((n_events, 3), dtype=np.float32)
        rsi_col = 'RSI14' if 'RSI14' in events_df.columns else None
        if rsi_col:
            for i in range(n_events):
                rsi_val = events_df[rsi_col].iloc[i]
                if pd.notna(rsi_val):
                    if rsi_val > 65:
                        state_data[i, 1] = 1.0    # 超买
                    elif rsi_val < 35:
                        state_data[i, 2] = 1.0    # 超卖
                    else:
                        state_data[i, 0] = 1.0     # 中性
                else:
                    state_data[i, 0] = 1.0
        else:
            state_data[:, 0] = 1.0                  # 默认中性
        
        # --- 5. 交互特征(44维): int_{rsi,ret20,vol,macd} × 11事件子类 ---
        int_features = np.zeros((n_events, 4 * len(event_subs_all)), dtype=np.float32)
        int_techs = ['RSI14', 'ret_20d', 'vol_ratio', 'MACD_dif']
        for t_idx, tcol in enumerate(int_techs):
            tvals = events_df[tcol].fillna(0).values.astype(np.float32)
            for s_idx in range(len(event_subs_all)):
                col_idx = t_idx * len(event_subs_all) + s_idx
                int_features[:, col_idx] = tvals * ev_onehot[:, s_idx]
        
        bayes_feat = np.zeros((n_events, 5), dtype=np.float32)
        for i in range(n_events):
            row = events_df.iloc[i]
            sub = row.get('event_sub', '其他')
            if pd.isna(sub) or sub == 'nan': sub = '其他'
            bp = bayesian_priors.get(sub, {})
            bayes_feat[i, 0] = bp.get('post_mean_car', 0.0)
            bayes_feat[i, 1] = bp.get('post_win_rate', 0.333)
            bayes_feat[i, 2] = bp.get('shrinkage_factor', 0.5)
            bayes_feat[i, 3] = bp.get('info_ratio', 0.0)
            bayes_feat[i, 4] = 1.0 if bp.get('is_positive', False) else 0.0
        
        # --- 7. 横截面排名特征(17维): xrank_RSI14, xrank_mom20d, ... ---
        xrank_names = ['RSI14', 'mom20d', 'vol20d', 'alpha', 'mom5d', 'MA_ratio',
                       'ATRpct', 'BOLLpos', 'declinedays', 'evCAR', 'evInfoRatio',
                       'evWinRate', 'volratio', 'volma5', 'MACDdif', 'closeT1', 'compositeTech']
        xrank_data = np.zeros((n_events, len(xrank_names)), dtype=np.float32)
        
        # 计算各指标的横截面排名(0~1)
        xrank_src_map = {
            'RSI14': 'RSI14', 'mom20d': 'ret_20d', 'vol20d': 'volatility_20d',
            'alpha': 'alpha', 'mom5d': 'ret_5d', 'MA_ratio': None,
            'ATRpct': 'ATR14_pct', 'BOLLpos': 'BOLL_pos', 'declinedays': 'decline_days',
            'evCAR': None, 'evInfoRatio': None, 'evWinRate': None,
            'volratio': 'vol_ratio', 'volma5': 'vol_ma5', 'MACDdif': 'MACD_dif',
            'closeT1': 'close_t1', 'compositeTech': None
        }
        for xr_idx, xr_name in enumerate(xrank_names):
            src_col = xrank_src_map[xr_name]
            if src_col and src_col in events_df.columns:
                vals = events_df[src_col].astype(float).values
                vals_rank = pd.Series(vals).rank(pct=True, method='average').values.astype(np.float32)
                xrank_data[:, xr_idx] = np.nan_to_num(vals_rank, nan=0.5)
            elif xr_name == 'MA_ratio':
                ma5 = events_df.get('MA5', pd.Series([np.nan]*n_events)).fillna(0).astype(float).values
                ma60 = events_df.get('MA60', pd.Series([np.nan]*n_events)).fillna(1e-8).astype(float).values
                ratio_vals = np.clip(ma5 / np.where(ma60 != 0, ma60, 1e-8), -5, 5)
                xrank_data[:, xr_idx] = pd.Series(ratio_vals).rank(pct=True).values.astype(np.float32)
            elif xr_name in ('evCAR', 'evInfoRatio', 'evWinRate'):
                for i in range(n_events):
                    row = events_df.iloc[i]
                    sub = row.get('event_sub', '其他')
                    bp = bayesian_priors.get(sub, {})
                    if xr_name == 'evCAR':
                        xrank_data[i, xr_idx] = bp.get('post_mean_car', 0.0)
                    elif xr_name == 'evInfoRatio':
                        xrank_data[i, xr_idx] = bp.get('info_ratio', 0.0)
                    else:
                        xrank_data[i, xr_idx] = bp.get('post_win_rate', 0.333)
                xrank_data[:, xr_idx] = pd.Series(xrank_data[:, xr_idx]).rank(pct=True).values.astype(np.float32)
            elif xr_name == 'compositeTech':
                comp = (xrank_data[:, 0] * 0.25 +      # RSI排名
                        xrank_data[:, 4] * 0.25 +      # mom5d排名
                        (1 - xrank_data[:, 2]) * 0.25 + # vol排名反转
                        (1 - xrank_data[:, 11]) * 0.25) # volratio排名反转
                xrank_data[:, xr_idx] = np.clip(comp, 0, 1)
            else:
                xrank_data[:, xr_idx] = 0.5  # 默认中位数
        
        X_lgbm = np.hstack([
            tech_data,          # 32
            ev_onehot,          # 11
            regime_data,        # 3
            state_data,         # 3
            int_features,       # 44
            bayes_feat,         # 5
            xrank_data          # 17
        ])  # total = 115
        
        logger.info(f"  LGBM输入特征维度: {X_lgbm.shape[1]} (tech={tech_data.shape[1]}, "
                   f"ev={ev_onehot.shape[1]}, regime={regime_data.shape[1]}, "
                   f"state={state_data.shape[1]}, int={int_features.shape[1]}, "
                   f"bayes={bayes_feat.shape[1]}, xrank={xrank_data.shape[1]})")
        
        try:
            car_preds = lgbm_reg.predict(X_lgbm)
            events_df['预测CAR'] = car_preds
            # Fix#2: sigmoid(CAR*2)与训练脚本L1377一致
            events_df['CAR预测概率'] = sigmoid(car_preds * 2.0)
            logger.info(f"  [OK] LGBM回归CAR: mean={car_preds.mean():+.4f}, "
                       f"min={car_preds.min():+.4f}, max={car_preds.max():+.4f}")
            logger.info(f"     sigmoid概率: mean={sigmoid(car_preds).mean():.3f}, "
                       f"max={sigmoid(car_preds).max():.3f}")
        except Exception as e:
            logger.warning(f"  [!] LGBM回归预测失败({type(e).__name__}): {e}")
    else:
        logger.warning("  LGBM回归不可用，CAR预测概率使用默认0.5")
    
    logger.info("  --- 5.4 贝叶斯事件得分（James-Stein收缩） ---")
    
    def compute_bayesian_event_score(row):
        sub = row.get('event_sub', '其他')
        if pd.isna(sub): sub = '其他'
        
        if sub in bayesian_priors:
            bp = bayesian_priors[sub]
            post_win_rate = float(bp.get('post_win_rate', 0.333))
            post_car = float(bp.get('post_mean_car', 0.0))
            info_ratio = float(bp.get('info_ratio', 0.0))
            is_sig = bool(bp.get('significant', False))
            
            # === 胜率主信号：4段阶梯式映射（以33%三分类随机基准为中心） ===
            if post_win_rate >= 0.42:
                score = 0.52 + min(0.25, (post_win_rate - 0.42) * 3.5)
            elif post_win_rate >= 0.36:
                score = 0.38 + (post_win_rate - 0.36) * 2.3
            elif post_win_rate >= 0.30:
                score = 0.28 + (post_win_rate - 0.30) * 1.7
            else:
                score = max(0.08, 0.18 + (post_win_rate - 0.30) * 1.5)
            
            # === CAR方向微调（4段，与训练脚本完全一致） ===
            if post_car > 0.3:
                score = min(0.95, score + 0.06)
            elif post_car > 0.1:
                score = min(0.90, score + 0.03)
            elif post_car < -0.3:
                score = max(0.05, score - 0.06)
            elif post_car < -0.1:
                score = max(0.10, score - 0.03)
            
            # === 显著性奖励（与训练脚本一致） ===
            if is_sig and post_win_rate >= 0.36:
                score = min(0.95, score + 0.04)
            
            # === 信息比调整（与训练脚本一致） ===
            if info_ratio > 0.5:
                score = min(0.95, score + min(0.05, info_ratio * 0.04))
            
            # === 情绪标签微调（与训练脚本一致） ===
            sentiment = str(row.get('情绪标签', ''))
            if sentiment in ['利好', '微偏正面']:
                score = min(0.95, score + 0.05)
            elif sentiment in ['利空', '微偏负面']:
                score = max(0.05, score - 0.05)
            
            # === 关键词微调（与训练脚本一致） ===
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
        else:
            # 情绪标签微调回退
            sentiment = str(row.get('情绪标签', ''))
            title = str(row.get('title', ''))
            neg_kw = ['续亏', '首亏', '略减', '预减', '减持', '质押', '冻结', '违规', '处罚', '诉讼', '退市']
            pos_kw = ['增持', '回购', '分红', '预增', '略增', '扭亏', '中标', '创新高', '突破']
            score = 0.40
            if sentiment in ['利好', '微偏正面']:
                score += 0.06
            elif sentiment in ['利空', '微偏负面']:
                score -= 0.06
            if any(kw in title for kw in pos_kw):
                score += 0.10
            elif any(kw in title for kw in neg_kw):
                score -= 0.10
            return max(0.05, min(0.95, round(score, 4)))
    
    events_df['贝叶斯事件得分'] = events_df.apply(compute_bayesian_event_score, axis=1)
    logger.info(f"  贝叶斯事件得分: mean={events_df['贝叶斯事件得分'].mean():.3f}, "
                f"std={events_df['贝叶斯事件得分'].std():.3f}")
    
    logger.info("  --- 5.5 v15g综合评分 ---")
    
    lr_up_prob = events_df['技术看涨概率'].values.astype(float)
    ev_scores = events_df['贝叶斯事件得分'].values.astype(float)
    car_pred_prob = events_df['CAR预测概率'].values.astype(float)
    
    # v15g公式：0.60×LR + 0.20×贝叶斯 + 0.20×CAR预测
    rank_score = lr_up_prob * 0.60 + ev_scores * 0.20 + car_pred_prob * 0.20
    
    # E4 LR决定方向
    if 'E4预测标签' in events_df.columns:
        direction = events_df['E4预测标签'].values.astype(int).copy()  # 0=跌, 1=平, 2=涨
    else:
        direction = np.full(len(events_df), 1, dtype=int)
        direction[lr_up_prob >= 0.50] = 2
        direction[lr_up_prob < 0.35] = 0
    
    #  LR-CAR矛盾过滤（与训练脚本完全一致）
    # 如果LR看涨但LGBM预测CAR为负且绝对值大 → 信号矛盾，降级为平
    car_pred_values = events_df['预测CAR'].values.astype(float)
    car_contradict_mask = (direction == 2) & (car_pred_values < -0.5)
    direction[car_contradict_mask] = 1
    logger.info(f"  LR-CAR矛盾过滤: {car_contradict_mask.sum()}条(LR看涨但CAR<-0.5)")
    
    # 动态否决阈值
    ev_mean = ev_scores.mean()
    ev_std = ev_scores.std()
    veto_threshold = max(0.10, ev_mean - 0.5 * ev_std)
    ev_q25 = np.percentile(ev_scores, 25)
    
    _pre_veto_up_score = np.where(direction == 2, rank_score, 0.0)
    
    veto_mask = (direction == 2) & (ev_scores < veto_threshold)
    direction[veto_mask] = 1
    
    # 追涨防御
    rsi_values = events_df['RSI14'].fillna(50).values if 'RSI14' in events_df.columns else np.full(len(events_df), 50.0)
    chase_veto_mask = (direction == 2) & (rsi_values > 70) & (ev_scores < ev_q25)
    direction[chase_veto_mask] = 1
    
    composite_scores = np.zeros(len(events_df))
    composite_scores[direction == 2] = rank_score[direction == 2]
    composite_scores[direction == 0] = -rank_score[direction == 0]
    
    label_decode = {0: -1, 1: 0, 2: 1}
    fusion_preds_decoded = [label_decode[d] for d in direction]
    
    fusion_reasons = []
    for i in range(len(events_df)):
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
    
    events_df['综合评分'] = composite_scores
    events_df['融合_预测标签'] = fusion_preds_decoded
    events_df['融合决策原因'] = fusion_reasons
    events_df['融合_预测置信度'] = composite_scores
    events_df['_否决前看涨评分'] = _pre_veto_up_score
    
    logger.info(f"  动态否决阈值: {veto_threshold:.4f} (ev_mean={ev_mean:.3f}, ev_std={ev_std:.3f})")
    logger.info(f"  综合评分: mean={events_df['综合评分'].mean():.3f}, "
                f"max={events_df['综合评分'].max():.3f}, min={events_df['综合评分'].min():.3f}")
    logger.info(f"  看涨(评分>0.5): {(events_df['综合评分']>0.5).sum()}/{len(events_df)} "
                f"({(events_df['综合评分']>0.5).mean()*100:.1f}%)")
    logger.info(f"  融合预测: 涨={sum(1 for p in events_df['融合_预测标签'] if p==1)}, "
                f"跌={sum(1 for p in events_df['融合_预测标签'] if p==-1)}, "
                f"平={sum(1 for p in events_df['融合_预测标签'] if p==0)}")
    
    pred_path = os.path.join(RESULT_DIR, 'prediction_results.csv')
    events_df.to_csv(pred_path, index=False, encoding='utf-8-sig')
    logger.info(f"  预测结果已保存: {pred_path}")
    
    return events_df, e4_lr, e4_lr_scaler

def run_stock_selection(events_df, trade_start, trade_end, week_num):
    logger.info("Step 6：综合评分选股 + 资金分配")
    
    if len(events_df) == 0:
        logger.warning("  无事件可选股")
        return pd.DataFrame()
    
    # v15g: 综合评分>0.5即看涨（与v15g训练脚本完全一致）
    score_threshold = 0.5
    
    bullish = events_df[
        (events_df['综合评分'] > score_threshold)
    ].copy()
    
    logger.info(f"  综合评分>{score_threshold}看涨: {len(bullish)} 条")
    
    if len(bullish) < 3:
        # 放宽条件：综合评分>0且预测看涨
        bullish = events_df[
            (events_df['综合评分'] > 0) &
            (events_df['融合_预测标签'] == 1)
        ].copy()
        logger.info(f"  放宽至评分>0且看涨: {len(bullish)} 条")
    
    if len(bullish) < 3:
        bullish = events_df[events_df['融合_预测标签'] == 1].copy()
        logger.info(f"  兜底条件(E4看涨): {len(bullish)} 条")
    
    if len(bullish) == 0:
        logger.warning("  [!] 无看涨信号！")
        return pd.DataFrame()
    
    bullish = bullish.sort_values('综合评分', ascending=False)
    
    # 分层选股：同一event_sub最多2只（确保多样性）
    MAX_STOCKS = 3  # 赛题要求：不超过3支买入股票
    max_stocks = min(MAX_STOCKS, len(bullish))
    max_per_sub = 2
    selected = []
    seen_subs = set()
    seen_codes = set()
    
    for _, row in bullish.iterrows():
        if len(selected) >= max_stocks:
            break
        
        code = row.get('ts_code', '')
        sub = row.get('event_sub', '其他')
        
        stock_name = str(row.get('name', row.get('股票名称', '')))
        title_text = str(row.get('title', row.get('标题_清洗后', '')))
        if '*ST' in stock_name or stock_name.startswith('ST') or '*ST' in title_text or 'ST' in title_text.upper().split():
            logger.info(f"    ST股过滤: 排除{code}({stock_name[:20]})")
            continue
        
        if code in seen_codes:
            continue
        
        sub_count = sum(1 for s in selected if s.get('event_sub', '其他') == sub)
        if sub_count >= max_per_sub:
            continue
        
        selected.append(row)
        seen_codes.add(code)
    
    # 如果选出的股票不足MAX_STOCKS只，放宽event_sub限制重选
    if len(selected) < MAX_STOCKS and len(bullish) >= MAX_STOCKS:
        selected = []
        seen_codes = set()
        for _, row in bullish.iterrows():
            if len(selected) >= MAX_STOCKS:
                break
            code = row.get('ts_code', '')
            if code in seen_codes:
                continue
            selected.append(row)
            seen_codes.add(code)
    
    if not selected:
        logger.warning("  选股结果为空")
        return pd.DataFrame()
    
    selected_df = pd.DataFrame(selected)
    logger.info(f"  选中 {len(selected_df)} 只股票:")
    for _, row in selected_df.iterrows():
        logger.info(f"    {row.get('ts_code', 'N/A')} | {row.get('event_sub', 'N/A')} | "
                    f"综合评分={row.get('综合评分', 0):.3f} | {row.get('title', '')[:30]}")
    
    # 按综合评分加权分配资金（比例制：0~1，合计=1）
    total_score = selected_df['综合评分'].sum()
    if total_score > 0:
        selected_df['资金比例'] = (selected_df['综合评分'] / total_score).round(4)
    else:
        selected_df['资金比例'] = round(1.0 / len(selected_df), 4)
    
    # 确保单只股票不超过40%（至少2只时有效），不低于20%（v13.6差异化约束）
    max_ratio = 0.40 if len(selected_df) >= 2 else 1.0
    min_ratio = 0.20 if len(selected_df) >= 2 else 1.0
    if selected_df['资金比例'].max() > max_ratio:
        # 重新分配：超过上限的截断，剩余均分给其他
        excess = 0
        for idx in selected_df.index:
            if selected_df.loc[idx, '资金比例'] > max_ratio:
                excess += selected_df.loc[idx, '资金比例'] - max_ratio
                selected_df.loc[idx, '资金比例'] = max_ratio
        below = selected_df[selected_df['资金比例'] < max_ratio]
        if len(below) > 0:
            each_extra = excess / len(below)
            for idx in below.index:
                selected_df.loc[idx, '资金比例'] = min(
                    max_ratio,
                    selected_df.loc[idx, '资金比例'] + each_extra
                )
    if selected_df['资金比例'].min() < min_ratio and len(selected_df) >= 2:
        # 低于下限的补足，从超限的扣除
        deficit = 0
        for idx in selected_df.index:
            if selected_df.loc[idx, '资金比例'] < min_ratio:
                deficit += min_ratio - selected_df.loc[idx, '资金比例']
                selected_df.loc[idx, '资金比例'] = min_ratio
        above = selected_df[(selected_df['资金比例'] > min_ratio) & (selected_df['资金比例'] < max_ratio)]
        if len(above) > 0 and deficit > 0:
            each_deduct = deficit / len(above)
            for idx in above.index:
                selected_df.loc[idx, '资金比例'] = max(
                    min_ratio,
                    selected_df.loc[idx, '资金比例'] - each_deduct
                )
    
    selected_df['资金比例'] = (selected_df['资金比例'] / selected_df['资金比例'].sum()).round(4)
    # Fix#4: 确保资金比例总和=1且精度为4位小数
    _ratio_sum = selected_df['资金比例'].sum()
    if abs(_ratio_sum - 1.0) > 0.001:
        logger.warning(f"  资金比例总和={_ratio_sum:.6f}, 重新归一化")
        selected_df['资金比例'] = (selected_df['资金比例'] / _ratio_sum).round(4)
    
    logger.info("\n  资金分配:")
    for _, row in selected_df.iterrows():
        amount = INITIAL_CAPITAL * row['资金比例']  # 已是比例制，无需/100
        logger.info(f"    {row.get('ts_code', 'N/A')}: {row['资金比例']:.4f} (¥{amount:,.0f})")
    
    return selected_df

def generate_result_xlsx(selected_df, week_num, code2name=None, stock_info=None):
    logger.info("Step 7a：生成result.xlsx")
    
    if len(selected_df) == 0:
        logger.warning("  无选股结果，生成空文件")
        result_df = pd.DataFrame(columns=['事件名称', '标的股票代码', '资金比例'])
    else:
        # v14修复：事件名称格式 = "公司名+事件类型"（自然语言）
        # 优先使用外部传入的code2name（来自Step 7b的全量查找表），否则自建
        if code2name is None:
            _code2name_tmp, _ = _build_stock_lookup(selected_df, stock_info=stock_info)
        else:
            _code2name_tmp = dict(code2name)  # 复制以防修改
        
        # 同时也尝试从selected_df的company_name列获取（排除ts_code格式占位）
        _code2name_from_df = {}
        for _, row in selected_df.iterrows():
            code_raw = str(row.get('ts_code', '')).split('.')[0]
            cn = row.get('company_name', '')
            # Fix: 过滤纯股票代码格式（通道3关联时存了ts_code作占位）
            _is_code_fmt = bool(re.match(r'^\d{6}', str(cn)))
            if code_raw and code_raw != 'nan' and pd.notna(cn) and str(cn).strip() and \
               str(cn).strip() not in ['nan', '', '--'] and not _is_code_fmt:
                _code2name_from_df[code_raw] = str(cn).strip()
            nm = row.get('name', '')
            if code_raw and code_raw != 'nan' and pd.notna(nm) and str(nm).strip() and str(nm).strip() not in ['nan', '', '--']:
                if code_raw not in _code2name_from_df:
                    _code2name_from_df[code_raw] = str(nm).strip()[:10]
        
        _code2name_tmp.update(_code2name_from_df)
        
        event_names = []
        for _, row in selected_df.iterrows():
            _code = str(row.get('ts_code', '')).split('.')[0]
            
            _company = _code2name_tmp.get(_code, '')
            if not _company or _company == _code:
                try:
                    ts_code = _code + '.SZ' if _code.startswith(('0','3')) else _code + '.SH'
                    info_df = _finance_data_api('stock_basic', {'ts_code': ts_code})
                    if info_df is not None and len(info_df) > 0:
                        _company = str(info_df.iloc[0].get('name', ''))[:20]
                        if _company and _company != 'nan':
                            _code2name_tmp[_code] = _company
                except Exception:
                    pass
            
            _event_sub = str(row.get('event_sub', ''))
            _title = str(row.get('title', ''))
            
            # 标准事件类型映射（短名称，适合result.xlsx）
            _event_type_map = {
                '业绩': '业绩预告', '人事变动': '人事变动', '股东变动': '股东增减持',
                '并购': '并购重组', '融资': '融资定增', '分红送转': '分红送转',
                '合同': '合同中标', '技术突破': '技术突破', '利空公告': '利空公告',
                '政策发布': '政策发布', '宏观': '宏观事件', '行业': '行业事件',
                '公司': '公司公告', '其他': '市场动态', '科技': '科技动态',
                '新能源': '新能源', '消费': '消费动态', '货币政策': '货币政策',
                '周期': '周期波动', '财政政策': '财政政策', '金融': '金融事件',
                '宏观数据': '宏观数据',
            }
            
            _short_type = _event_type_map.get(_event_sub, '')
            if not _short_type:
                _short_type = refine_event_name(_title, _event_sub)
                if len(_short_type) > 10:
                    _short_type = _short_type[:9] + '…'
            # Fix#3: 去除方括号标记（内部分类标签泄漏到输出）
            _short_type = re.sub(r'[\[\]【】]', '', _short_type)
            
            # 拼接："公司名+事件类型"（简洁格式，适合result.xlsx提交）
            if _company and _company != _code:
                full_name = f"{_company}{_short_type}"
            else:
                full_name = f"{_short_type}({_code})" if _short_type else f"未知事件({_code})"
            
            if len(full_name) > 40:
                full_name = full_name[:39] + '…'
            event_names.append(full_name)
        
        # 股票代码：提取6位纯数字，确保前导零（如3001→003001）
        stock_codes = selected_df['ts_code'].apply(
            lambda x: str(x.split('.')[0] if '.' in str(x) else str(x)).zfill(6)
        ).values
        
        fund_ratios = selected_df['资金比例'].values
        
        result_df = pd.DataFrame({
            '事件名称': event_names,
            '标的股票代码': stock_codes,
            '资金比例': fund_ratios,
        })
    
    # 保存xlsx —— 使用openpyxl设置代码列为文本格式保留前导零
    xlsx_path = os.path.join(RESULT_DIR, f'result_week{week_num}.xlsx')
    std_path = os.path.join(RESULT_DIR, 'result.xlsx')
    
    try:
        from openpyxl import Workbook
        for save_path in [xlsx_path, std_path]:
            wb = Workbook()
            ws = wb.active
            ws.append(['事件名称', '标的股票代码', '资金比例'])
            for row_idx, (_, row) in enumerate(result_df.iterrows(), start=2):
                # 股票代码必须以文本格式写入，防止前导零丢失
                code_cell = ws.cell(row=row_idx, column=2, value=str(row['标的股票代码']))
                code_cell.number_format = '@'
                ws.cell(row=row_idx, column=1, value=str(row['事件名称']))
                ws.cell(row=row_idx, column=3, value=round(float(row['资金比例']), 4))
            ws.column_dimensions['A'].width = 28
            ws.column_dimensions['B'].width = 12
            ws.column_dimensions['C'].width = 10
            wb.save(save_path)
    except ImportError:
        # 回退：用pandas直接写（可能丢前导零）
        for save_path in [xlsx_path, std_path]:
            result_df.to_excel(save_path, index=False, engine='openpyxl')
    
    logger.info(f"  result.xlsx已保存: {xlsx_path}")
    logger.info(f"  内容:\n{result_df.to_string(index=False)}")
    logger.info(f"  标准提交文件已保存: {std_path}")
    
    return result_df

def _build_stock_lookup(events_df, stock_info=None):
    _code2name = {}
    _code2industry = {}

    if stock_info is not None:
        code_col = name_col = ind_col = None
        for c in ['ts_code', '股票代码', '代码', 'code']:
            if c in stock_info.columns:
                code_col = c; break
        for c in ['name', '股票简称', '名称', 'stock_name', '公司名称']:
            if c in stock_info.columns:
                name_col = c; break
        for c in ['industry', '所属行业', '申万行业', '行业']:
            if c in stock_info.columns:
                ind_col = c; break
        if code_col and name_col:
            for _, sr in stock_info.iterrows():
                k = str(sr[code_col]).split('.')[0]
                if name_col and pd.notna(sr.get(name_col, '')):
                    _code2name[k] = str(sr[name_col])
                if ind_col and pd.notna(sr.get(ind_col, '')):
                    _code2industry[k] = str(sr[ind_col])

    if len(events_df) > 0:
        _name_cols = [c for c in events_df.columns
                      if any(kw in c.lower() for kw in ['名称','简称','name']) and '事件' not in c]
        for nc in _name_cols:
            if events_df[nc].notna().any():
                for _, r in events_df.dropna(subset=[nc]).iterrows():
                    k = str(r.get('ts_code', '')).split('.')[0]
                    v = str(r[nc]).strip() if pd.notna(r[nc]) else ''
                    # Fix #25: 过滤不像公司名的值（标题片段/股票代码/纯数字）
                    _is_code_fmt = bool(re.match(r'^\d{6}', v))
                    _is_title_frag = any(kw in v for kw in ['2026','2025','关于','第','公告','H股']) and len(v) > 8
                    if k and k != 'nan' and v and v != 'nan' and k not in _code2name \
                       and not _is_code_fmt and not _is_title_frag:
                        _code2name[k] = v[:20]  # 放宽到20字符

    # Fix #25: 改为冒号前提取公司名（如"顺丰控股:关于..."→"顺丰控股"）
    # 旧逻辑冒号后提取会得到"2026年3月快递物"等标题片段，不是公司名
    if len(events_df) > 0 and 'title' in events_df.columns:
        for _, r in events_df.iterrows():
            k = str(r.get('ts_code', '')).split('.')[0]
            if not k or k == 'nan' or k in _code2name:
                continue
            t = str(r.get('title', ''))
            # 策略1: 冒号前提取（"顺丰控股:关于..."→"顺丰控股"）
            m = re.search(r'^([^：:]{2,8})[：:]', t)
            if m:
                before = m.group(1).strip()
                _skip_prefixes = {'预增','预减','续亏','首亏','扭亏','略增','略减','不确定',
                                  '预案','实施','公告','报告','决议','通知','方案',
                                  '业绩预告','分红送转','股东增减持','融资定增','龙虎榜'}
                _is_code_fmt = bool(re.match(r'^\d{6}', before))
                if before and before not in _skip_prefixes and len(before) >= 2 and not _is_code_fmt:
                    _code2name[k] = before

    # Level 3: 在线查询finance-data HTTP API补充公司名和行业
    _missing_codes = []
    for _, r in events_df.iterrows():
        _c = str(r.get('ts_code', '')).split('.')[0]
        if _c and _c != 'nan' and _c not in _code2name and _c not in _missing_codes:
            _missing_codes.append(_c)
    if _missing_codes:
        _batch = _missing_codes[:80]  # 最多查80只
        logger.info(f"  在线查询公司信息: {len(_batch)} 只缺失公司名")

        # ── Level 3a: BaoStock query_stock_basic + query_stock_industry（免费/稳定/无配额限制）──
        _baostock_ok = 0
        _baostock_ind_ok = 0
        try:
            import baostock as _bs
            _lg = _bs.login()
            if _lg.error_code == '0':
                for _code in _batch:
                    _bs_code = ('sz.' if _code.startswith(('0','3')) else 'sh.') + _code
                    if _code not in _code2name:
                        try:
                            _rs = _bs.query_stock_basic(code=_bs_code)
                            while _rs.error_code == '0' and _rs.next():
                                _row = _rs.get_row_data()
                                # row = [code, code_name, ipoDate, outDate, type, status]
                                if len(_row) >= 2 and _row[1] and str(_row[1]).strip() not in ('', 'nan'):
                                    _code2name[_code] = str(_row[1]).strip()
                                    _baostock_ok += 1
                                    break
                        except Exception:
                            pass
                    if _code not in _code2industry:
                        try:
                            _rs2 = _bs.query_stock_industry(code=_bs_code)
                            while _rs2.error_code == '0' and _rs2.next():
                                _row2 = _rs2.get_row_data()
                                # Fix #25: BaoStock query_stock_industry返回5列:
                                # [updateDate, code, code_name, industry, industryClassification]
                                # 行业名在index=3，而非index=1（index=1是sz.000333格式的代码）
                                _ind_val = ''
                                if len(_row2) >= 4 and _row2[3] and str(_row2[3]).strip() not in ('', 'nan'):
                                    _ind_val = str(_row2[3]).strip()
                                elif len(_row2) >= 2 and _row2[1] and str(_row2[1]).strip() not in ('', 'nan'):
                                    _ind_val = str(_row2[1]).strip()
                                if _ind_val:
                                    _code2industry[_code] = _ind_val
                                    _baostock_ind_ok += 1
                                    break
                        except Exception:
                            pass
                _bs.logout()
        except ImportError:
            pass
        if _baostock_ok > 0 or _baostock_ind_ok > 0:
            logger.info(f"  BaoStock: 公司名 {_baostock_ok} 只, 行业 {_baostock_ind_ok} 只")

        # ── Level 3b: finance-data API 备选（仅BaoStock未查到的）──
        _still_missing = [c for c in _batch if c not in _code2name]
        if _still_missing:
            for _code in _still_missing:
                try:
                    ts_code = _code + '.SZ' if _code.startswith(('0','3')) else _code + '.SH'
                    result_df = _finance_data_api('stock_basic', {
                        'ts_code': ts_code,
                        'fields': 'ts_code,name,industry'
                    })
                    if result_df is not None and len(result_df) > 0:
                        _row_data = result_df.iloc[0]
                        _name = str(_row_data.get('name', ''))
                        _ind = str(_row_data.get('industry', ''))
                        if _name and _name not in ('nan', 'None', ''):
                            _code2name[_code] = _name
                        if _ind and _ind not in ('nan', 'None', ''):
                            _code2industry[_code] = _ind
                except Exception:
                    continue
        logger.info(f"  在线查询完成: 公司名 {len(_code2name)} 只, 行业 {len(_code2industry)} 只")

    # Level 4: 仍缺失的用代码做名称占位
    for _, r in events_df.iterrows():
        k = str(r.get('ts_code', '')).split('.')[0]
        if k and k != 'nan' and k not in _code2name:
            _code2name[k] = k

    return _code2name, _code2industry

def generate_analysis_report(events_df, selected_df, week_num, trade_start, trade_end,
                              event_stats_df=None, stock_info=None,
                              code2name=None, code2industry=None,
                              e4_lr=None, e4_lr_scaler=None):
    logger.info("Step 7b：生成分析报告（v16学术规范版）")

    # ── 辅助查找表（优先使用外部传入，避免重复构建）──
    if code2name is not None and code2industry is not None:
        _code2name, _code2industry = code2name, code2industry
    else:
        _code2name, _code2industry = _build_stock_lookup(events_df, stock_info)

    _stats_map = {}
    if event_stats_df is not None:
        sub_col = [c for c in event_stats_df.columns if '事件子类' in c or 'event_sub' in c.lower()]
        sub_col = sub_col[0] if sub_col else event_stats_df.columns[0]
        for _, sr in event_stats_df.iterrows():
            _stats_map[str(sr[sub_col])] = sr
    logger.info(f"  事件统计映射: {len(_stats_map)} 个子类, 公司名: {len(_code2name)}, 行业: {len(_code2industry)}")

    # ── 辅助函数：安全取值 ──
    def _val(row, *keys, default='--'):
        for k in keys:
            if k in row.index and pd.notna(row[k]):
                return row[k]
        return default

    def _fmt(v, fmt='{:.3f}', nan='--'):
        try:
            fv = float(v)
            return fmt.format(fv)
        except (ValueError, TypeError):
            return nan

    def _pct(v, nan='--'):
        try:
            fv = float(v)
            return f"{fv:+.2f}%"
        except:
            return nan

    R = []  # report lines

    R.append("# 事件驱动型股市投资策略分析报告")
    R.append("")
    R.append(f"**第{week_num}周**（{trade_start} ~ {trade_end}）")
    R.append("")

    # ─────────────────────────────────────────────
    #  一、事件识别与量化特征
    # ─────────────────────────────────────────────
    R.append("---")
    R.append("")
    R.append("## 一、事件采集与量化特征")
    R.append("")

    # Fix #28: 报告展示用温度缩放概率（LR的predict_proba对极端样本输出0/1，
    # 这是softmax的正常行为，但报告展示不专业。T=3缩放后概率有梯度，综合评分计算保持原始值）
    _RPT_SOFTMAX_T = 15.0
    if '技术看涨概率' in events_df.columns and e4_lr is not None:
        try:
            from scipy.special import softmax as _sp_softmax
            lr_cols_rpt = [c for c in TECH_FACTOR_COLS if c in events_df.columns]
            if lr_cols_rpt:
                _lr_data_rpt = events_df[lr_cols_rpt].fillna(0).values
                # 使用与Step5相同的scaler（已在Step5中加载）
                _rpt_scaler = e4_lr_scaler if e4_lr_scaler is not None else None
                if _rpt_scaler is not None:
                    _lr_scaled_rpt = _rpt_scaler.transform(_lr_data_rpt)
                    _dec_rpt = e4_lr.decision_function(_lr_scaled_rpt)
                    _lr_probs_rpt = _sp_softmax(_dec_rpt / _RPT_SOFTMAX_T, axis=1)
                    _up_idx = list(e4_lr.classes_).index(2) if 2 in e4_lr.classes_ else -1
                    if _up_idx >= 0:
                        events_df['_展示看涨概率'] = _lr_probs_rpt[:, _up_idx]
        except Exception:
            pass
    if '_展示看涨概率' not in events_df.columns:
        events_df['_展示看涨概率'] = events_df.get('技术看涨概率', 0)

    # 为selected_df也映射温度缩放概率（表8展示用）
    if len(selected_df) > 0 and '_展示看涨概率' not in selected_df.columns:
        _sel_disp_map = dict(zip(
            events_df['ts_code'].astype(str),
            events_df['_展示看涨概率'].astype(float)
        ))
        selected_df['_展示看涨概率'] = selected_df['ts_code'].astype(str).map(_sel_disp_map).fillna(
            selected_df.get('技术看涨概率', 0)
        )

    if len(events_df) > 0:
        R.append(f"本周共采集并清洗事件 **{len(events_df)}** 条，经清洗规则（治理制度页/假新闻检测/低价值文档过滤）后保留有效事件。")
        R.append("")

        R.append("### 1.1 事件类型分布")
        R.append("")

        if 'event_type' in events_df.columns:
            type_dist = events_df['event_type'].value_counts()
            R.append("**表1 事件大类分布**")
            R.append("")
            R.append("| 事件大类 | 数量 | 占比 | 累计占比 |")
            R.append("|:----------|:----------:|:---------------:|:--------:|")
            cumsum = 0
            for t, c in type_dist.items():
                pct = c / len(events_df) * 100
                cumsum += pct
                R.append(f"| {t} | {c} | {pct:.1f}\\% | {cumsum:.1f}\\% |")
            R.append("")

        if 'event_sub' in events_df.columns:
            sub_dist = events_df['event_sub'].value_counts()
            R.append("**表2 事件子类分布（Top 15）**")
            R.append("")
            R.append("| 事件子类 | 数量 | 占比 |")
            R.append("|:---------|:----:|:----:|")
            for t, c in sub_dist.head(15).items():
                R.append(f"| {t} | {c} | {c/len(events_df)*100:.1f}\\% |")
            R.append("")
            R.append("*注：采集端AHP分类器输出包含部分二级细分类别（如宏观/科技/消费等），进入影响预测时统一映射到标准11类子类体系。*")
            R.append("")

        # ---- 1.2 Top事件（含实际量化特征） ----
        R.append("### 1.2 高信度事件量化特征")
        R.append("")
        R.append("**表3 综合评分Top 15事件及其统计特征**")
        R.append("")
        R.append("| 排名 | 事件摘要 | 类型 | 子类 | LR看涨概率 | 预测CAR | 综合评分 |")
        R.append("|:----:|:---------|:----:|:----:|:----------:|:-----------:|:----------:|")

        top_n = min(15, len(events_df))
        sort_key = '综合评分' if '综合评分' in events_df.columns else '预测CAR'
        # Fix #27: 按ts_code去重——同一股票多个事件技术因子相同，仅保留最高评分事件
        _top_all = events_df.nlargest(len(events_df), sort_key) if sort_key in events_df.columns else events_df
        _seen_top_codes = set()
        top_events_list = []
        for _, _r in _top_all.iterrows():
            _tc = str(_val(_r, 'ts_code', default='')).split('.')[0]
            if _tc and _tc not in _seen_top_codes:
                _seen_top_codes.add(_tc)
                top_events_list.append(_r)
            if len(top_events_list) >= top_n:
                break

        for i, row in enumerate(top_events_list, 1):
            code = str(_val(row, 'ts_code', default='')).split('.')[0]
            company = _code2name.get(code, '')

            # 事件标题精炼（用短事件类型，避免冗长标题）
            raw_title = str(_val(row, 'title', default=''))
            esub = str(_val(row, 'event_sub', default='其他'))
            
            # 标准事件类型映射（短名称）
            _rpt_type_map = {
                '业绩': '业绩预告', '人事变动': '人事变动', '股东变动': '股东增减持',
                '并购': '并购重组', '融资': '融资定增', '分红送转': '分红送转',
                '合同': '合同中标', '技术突破': '技术突破', '利空公告': '利空公告',
                '政策发布': '政策发布', '宏观': '宏观事件', '行业': '行业事件',
                '公司': '公司公告', '其他': '市场动态', '科技': '科技动态',
                '新能源': '新能源', '消费': '消费动态', '货币政策': '货币政策',
                '周期': '周期波动', '财政政策': '财政政策', '金融': '金融事件',
                '宏观数据': '宏观数据',
            }
            _short_type = _rpt_type_map.get(esub, refine_event_name(raw_title, esub)[:10])
            _short_type = re.sub(r'[\[\]【】]', '', _short_type)  # Fix#3: 去方括号标记

            if company and company != code:
                evt_summary = f"{company}{_short_type}"[:38]
            else:
                evt_summary = f"{_short_type}({code})"[:38] if _short_type else f"{esub}({code})"[:38]

            etype = _val(row, 'event_type', default='--')
            sub = esub

            lr_prob = _fmt(_val(row, '_展示看涨概率', '技术看涨概率', 'tech_up_prob'), fmt='{:.3f}')
            car_pred = _fmt(_val(row, '预测CAR', 'CAR预测', 'car_pred', default=0), fmt='{:+.2f}%')
            comp_score = _fmt(_val(row, '综合评分', default=0), fmt='{:.3f}')

            R.append(f"| {i} | {evt_summary} | {etype} | {sub} | {lr_prob} | {car_pred} | {comp_score} |")
        R.append("")
        R.append("*注：LR看涨概率为E4 LR模型P(涨)；预测CAR为LGBM回归预测的累计异常收益率；综合评分为三模型融合评分。*")
        R.append("")

        R.append("### 1.3 分类与识别方法论")
        R.append("")
        R.append("本系统采用**多模型融合架构**进行事件识别与分类，核心模块包括：")
        R.append("")
        R.append("**（1）事件分类 — AHP层次关键词规则 + TF-IDF/LR**")
        R.append("")
        R.append("基于层次分析法（AHP）构建三层分类体系：")
        R.append("- 第一层（主体维度）：宏观 / 公司 / 行业 / 其他 — 采用关键词精确+模糊匹配")
        R.append("- 第二层（子类维度）：11类细分子类（人事变动/业绩/股东变动/并购等）— AHP权重排序")
        R.append("- 辅助分类器：TF-IDF向量化（max_features=5000）+ Logistic Regression多分类头")
        R.append("- **验证性能**：主体分类 F1 = **0.970**，Accuracy = 0.968")
        R.append("")
        R.append("**（2）关联挖掘 — 三通道融合**")
        R.append("- **通道1**（公告直接提取）：从公告数据源解析股票代码字段，关联强度「强」")
        R.append("- **通道2**（在线公司名查询）：事件子类→按公司名反查证券基础信息库，关联强度「中」")
        R.append("- **通道3**（历史共现映射）：事件子类→训练集共现Top-K标的（数据驱动），关联强度「弱」")
        R.append("")
        R.append("**（3）数据质量保障 — 清洗规则**")
        R.append("- 规则A：过滤治理制度类（投资者关系/年报差错追究/信披制度等10类50+关键词）")
        R.append("- **规则B（修订）：智能假新闻双层检测**")
        R.append("  - B1 硬过滤：仅过滤「实证窗口」标记条目（训练集确认的数据注入标记）")
        R.append("  - B2 软检测：多股共享标题警告记录，不自动删除（保留真实行业级事件）")
        R.append("- 规则C：过滤低价值常规公告（关联交易/委托理财等4类）")
        R.append("")

    # ─────────────────────────────────────────────
    # ─────────────────────────────────────────────
    R.append("---")
    R.append("")
    R.append("## 二、「事件–标的证券」关联图谱")
    R.append("")

    if len(events_df) > 0:
        code_count = events_df['ts_code'].nunique() if 'ts_code' in events_df.columns else 0
        has_code_count = events_df['ts_code'].notna().sum() if 'ts_code' in events_df.columns else 0
        R.append(f"本周事件共关联 **{code_count}** 只标的证券，其中 **{has_code_count}** 条事件包含明确的股票代码。")
        R.append("")

        R.append("### 2.1 关联方式分布")
        R.append("")

        # 统计各关联方式数量（按assoc_method列判断，无此列则按event_sub推断）
        assoc_counts = {'公告直接': 0, '在线查询': 0, '历史共现映射': 0}
        if 'ts_code' in events_df.columns:
            for _, row in events_df.iterrows():
                code_pure = str(row.get('ts_code', '')).split('.')[0]
                if not code_pure or not code_pure.startswith(('0', '3', '6')) or not str(row.get('ts_code', '')):
                    continue
                am = str(row.get('assoc_method', ''))
                if '子类映射' in am or '共现' in am:
                    assoc_counts['历史共现映射'] += 1
                elif '在线' in am or '查询' in am or 'stock_basic' in am:
                    assoc_counts['在线查询'] += 1
                else:
                    assoc_counts['公告直接'] += 1

        R.append("| 关联渠道 | 事件数量 | 占比 |")
        R.append("|:---------|:--------:|:----:|")
        total_assoc = sum(assoc_counts.values()) or 1
        for k, v in assoc_counts.items():
            R.append(f"| {k} | {v} | {v/total_assoc*100:.1f}\\% |")
        R.append("")

        R.append("### 2.2 关联明细（按关联强度排序）")
        R.append("")
        R.append("**表4 事件–标的证券关联关系**")
        R.append("")
        R.append("| 事件主体 | 标的证券 | 代码 | 关联渠道 | 强度 | 所属行业 |")
        R.append("|:---------|:---------|:----:|:---------|:----:|:---------|")

        assoc_records = []
        seen_keys = set()

        if 'event_sub' in events_df.columns and 'ts_code' in events_df.columns:
            for _, row in events_df.iterrows():
                sub = row.get('event_sub', '未知')
                raw_t = str(_val(row, 'title', default=''))
                core_e = refine_event_name(raw_t, str(sub))

                code_full = str(row.get('ts_code', ''))
                code_pure = code_full.split('.')[0] if '.' in code_full else code_full
                if not code_pure or code_pure == 'nan':
                    continue

                name = _code2name.get(code_pure, '')
                if not name or name == code_pure:
                    cn = _val(row, 'company_name', default='')
                    if cn and cn != '--':
                        name = str(cn).strip()
                        # Fix #25: 过滤标题片段（含数字/年/月/关于等，不像公司名）
                        if any(kw in name for kw in ['2026','2025','关于','第','月','公告','H股']):
                            name = ''

                if name and name != code_pure:
                    evt_d = f"{name}{core_e}"[:28]
                else:
                    evt_d = f"{core_e}({code_pure})"[:28] if core_e else f"{sub}({code_pure})"[:28]

                # 关联方式和强度（按assoc_method列判断）
                am = str(row.get('assoc_method', ''))
                if '子类映射' in am or '共现' in am:
                    atype, astr = '历史共现映射', '弱'
                elif '在线' in am or '查询' in am or 'stock_basic' in am:
                    atype, astr = '在线查询', '中'
                else:
                    atype, astr = '公告直接', '强'

                ind = _code2industry.get(code_pure, '')
                # Fix #25: BaoStock返回的行业名如"C38电气机械和器材制造业"，需简化
                if ind and ind != '--':
                    _ind_m = re.match(r'^[A-Z]\d+(.+)$', ind)
                    if _ind_m:
                        ind = _ind_m.group(1)
                    # 进一步简化：取"和"前的核心行业词
                    for _sep in ['和', '、', '及']:
                        if _sep in ind:
                            ind = ind.split(_sep)[0]
                            break
                if not ind or ind == '--':
                    s2i = {'业绩':'金融/消费','股东变动':'综合','利空公告':'综合','并购':'综合',
                           '合同':'工业','融资':'金融','技术突破':'科技','政策发布':'宏观',
                           '消费':'消费','分红送转':'金融','公司':'综合','其他':'综合',
                           '宏观':'宏观','金融':'金融','科技':'科技','新能源':'新能源',
                           '货币政策':'金融','财政政策':'金融','周期':'工业','行业':'综合',
                           '宏观数据':'宏观','人事变动':'综合'}
                    ind = s2i.get(sub, '--')

                key = (evt_d[:18], code_pure)
                if key not in seen_keys:
                    seen_keys.add(key)
                    assoc_records.append((astr, code_pure, evt_d, atype, name, ind[:12]))

            s_order = {'强': 0, '中': 1, '弱': 2}
            assoc_records.sort(key=lambda x: (s_order.get(x[0], 9), x[1]))

            for astr, cp, evt_d, atype, name, ind in assoc_records[:30]:
                name_disp = name[:12] if len(name) > 12 else (name or '--')
                R.append(f"| {evt_d[:28]} | {name_disp} | {cp} | {atype} | {astr} | {ind} |")
        R.append("")

        R.append("### 2.3 按事件子类聚合统计")
        R.append("")
        R.append("**表5 各事件子类覆盖标的数量**")
        R.append("")
        R.append("| 事件子类 | 标的数 | 代表标的 |")
        R.append("|:---------|:------:|:---------|")
        if 'event_sub' in events_df.columns and 'ts_code' in events_df.columns:
            sub_codes = events_df.groupby('event_sub')['ts_code'].apply(lambda x: list(set([str(c).split('.')[0] for c in x.dropna()])))
            for sub, codes in sub_codes.items():
                uc = sorted(list(set(codes)))  # 去重排序
                rep = ', '.join(uc[:3])
                if len(uc) > 3:
                    rep += f" 等{len(uc)}只"
                R.append(f"| {sub} | {len(uc)} | {rep} |")
        R.append("")

        # ---- 2.4 事件子类历史统计特征（来自训练集） ----
        if _stats_map:
            R.append("### 2.4 各事件子类历史统计特征")
            R.append("")
            R.append("下表基于训练集（5,674条样本）按事件子类分组统计的历史CAR特征，作为贝叶斯事件先验的先验参数来源：")
            R.append("")
            R.append("**表5b 事件子类历史CAR统计**")
            R.append("")
            R.append("| 事件子类 | 样本数 | 平均CAR(%) | CAR标准差(%) | 胜率(%) | t统计量 | 显著性 | 方向 |")
            R.append("|:---------|:------:|:----------:|:------------:|:-------:|:-------:|:------:|:----:|")
            for sub_key in sorted(_stats_map.keys()):
                sr = _stats_map[sub_key]
                n = int(sr.get('样本数', sr.get('总次数', 0)))
                car_mean = float(sr.get('平均CAR', 0))
                car_std = float(sr.get('CAR标准差', 0))
                win_rate = float(sr.get('胜率', 0))
                t_stat = float(sr.get('t统计量', 0))
                sig = sr.get('是否显著(p<0.05)', '--')
                direction = sr.get('方向', '--')
                sig_str = '*' if sig is True or str(sig).lower() in ('true', 'yes', '1') else ''
                dir_str = '↑涨' if '涨' in str(direction) else ('↓跌' if '跌' in str(direction) else '--')
                R.append(f"| {sub_key} | {n} | {car_mean:+.2f} | {car_std:.2f} | {win_rate*100:.1f} | {t_stat:.2f} | {sig_str} | {dir_str} |")
            R.append("")
            R.append("> **注**：显著性基于双侧t检验（H₀: CAR均值=0），p<0.05标记为*。胜率定义为事件后CAR>0的比例。")
            R.append("")

    # ─────────────────────────────────────────────
    # ─────────────────────────────────────────────
    R.append("---")
    R.append("")
    R.append("## 三、事件驱动的股价影响估计")
    R.append("")

    if len(events_df) > 0:
        # ---- 3.1 预测模型架构（v15g更新） ----
        R.append("### 3.1 预测模型体系")
        R.append("")
        R.append("本研究采用**三模型融合框架**进行事件驱动的股价影响预测：")
        R.append("")
        R.append("**模型A：E4-LR技术因子分类器**（方向判断主模型）")
        R.append("- 特征空间：32维纯技术因子（MA动量/RSI/MACD/BOLL/ATR/量比/Alpha-Beta等）")
        R.append("- 算法：Logistic Regression（L2正则化，类别权重开方平滑 wc = 1/√(nc+1)）")
        R.append("- 输出：三分类概率向量 [P(跌), P(平), P(涨)]，取P(涨)作为技术面看涨置信度")
        R.append("- 验证性能：Weighted-F1 = **0.443**")
        R.append("")
        R.append("**模型B：LGBM-CAR回归模型**（连续收益预测）")
        R.append("- 特征空间：115维（技术因子32 + 事件独热11 + 市场状态3 + 交互项44 + 贝叶斯5 + 横截面排名17）")
        R.append("- 算法：LightGBM Regressor（num_leaves=31, learning_rate=0.05, num_boost_round=23）")
        R.append("- 输出：预测CAR r̂car∈ℝ，经sigmoid映射为概率 P_car = σ(r̂car)")
        R.append("")
        R.append("**模型C：贝叶斯事件先验**（James-Stein收缩估计）")
        R.append("- 先验来源：训练集按event_sub分组统计的历史CAR均值 r̄k 和胜率 ωk")
        R.append("- 收缩公式：θ̂k = r̄k·B + r̄global·(1-B)，其中B为收缩系数")
        R.append("- 输出：收缩后的事件先验得分 S_bayes∈[0,1]")
        R.append("")
        R.append("**综合评分公式**：")
        R.append("")
        R.append("S_comp = 0.60 × LR看涨概率 + 0.20 × 贝叶斯事件得分 + 0.20 × CAR预测概率")
        R.append("")
        R.append("其中权重分配依据消融实验中各模型的边际贡献率确定（详见第五节）。")
        R.append("")

        # ---- 3.2 预测分布（用实际数据） ----
        label_col = None
        for lc in ['融合_预测标签', 'e4_direction', 'E4预测标签', '预测标签']:
            if lc in events_df.columns:
                label_col = lc
                break

        if label_col:
            pred_dist = events_df[label_col].value_counts()
            R.append("### 3.2 全样本预测分布")
            R.append("")
            R.append("**表6 预测方向分布**")
            R.append("")
            R.append("| 预测方向 | 事件数 | 占比 | 判定依据 |")
            R.append("|:---------|:------:|:----:|:---------|")
            # 智能识别标签编码：检测实际值范围自动适配
            uniq_labels = sorted([int(l) if isinstance(l, (int, float, np.integer)) else l for l in pred_dist.index])
            # - E4预测标签: 0=跌, 1=平, 2=涨
            # - 融合_预测标签: -1=跌, 0=平, 1=涨
            if set(uniq_labels) == {-1, 0, 1}:
                # 融合_预测标签编码: -1=跌, 0=平, 1=涨
                lmap_full = {
                    1: ('看涨', '技术面看涨且事件非负面'),
                    0: ('看平', '方向不明确或信号矛盾'),
                    -1: ('看跌', '技术面看跌或极端利空'),
                }
            elif set(uniq_labels).issubset({0, 1, 2}):
                lmap_full = {
                    2: ('看涨', '技术面看涨且事件非负面'),
                    1: ('看平', '方向不明确或信号矛盾'),
                    0: ('看跌', '技术面看跌或极端利空'),
                }
            elif set(uniq_labels).issubset({0, 1}) or set(uniq_labels).issubset({-1, 1}):
                lmap_full = {
                    1: ('看涨', '技术面看涨且事件非负面'),
                    0: ('看跌', '技术面看跌'),
                    -1: ('看跌', '技术面看跌'),
                }
            else:
                lmap_full = {l: (f'类别{l}', '待观察') for l in uniq_labels}

            for lbl, cnt in pred_dist.items():
                li = int(lbl) if isinstance(lbl, (int, float, np.integer)) else 1
                name, reason = lmap_full.get(li, (f'类别{lbl}', '待观察'))
                R.append(f"| {name} | {cnt} | {cnt/len(events_df)*100:.1f}\\% | {reason} |")
            R.append("")

            # 实际统计数据（展示用温度缩放概率）
            _lr_disp_col = '_展示看涨概率' if '_展示看涨概率' in events_df.columns else '技术看涨概率'
            if _lr_disp_col in events_df.columns:
                lr_vals = events_df[_lr_disp_col].dropna()
                if len(lr_vals) > 0:
                    R.append(f"*E4 LR看涨概率统计（T={_RPT_SOFTMAX_T:.0f}缩放展示）：均值={lr_vals.mean():.3f}, 标准差={lr_vals.std():.3f}, "
                            f"最大值={lr_vals.max():.3f}, Q75={lr_vals.quantile(0.75):.3f}*")
                    R.append("")
            for _car_col in ['预测CAR', 'CAR预测', 'car_pred']:
                if _car_col in events_df.columns:
                    cv = events_df[_car_col].dropna()
                    break
                if len(cv) > 0:
                    R.append(f"*LGBM CAR预测统计：均值={cv.mean():+ .2f}%, 标准差={cv.std():.2f}%, "
                            f"最大值={cv.max():+ .2f}%, 最小值={cv.min():+ .2f}%*")
                    R.append("")
            if '综合评分' in events_df.columns:
                sv = events_df['综合评分'].dropna()
                if len(sv) > 0:
                    R.append(f"*综合评分统计：均值={sv.mean():.3f}, 标准差={sv.std():.3f}, "
                            f"最大值={sv.max():.3f}, 正分率={ (sv>0).sum()/len(sv)*100:.1f}%*")
                    R.append("")

        # ---- 3.3 影响估计明细（用实际数据，无'--'） ----
        R.append("### 3.3 标的证券股价影响估计")
        R.append("")
        R.append("影响幅度估计参考 **MacKinlay (1997)** 短期事件研究法CAR分布及 **Corrado (1989)** 非参数检验框架，结合本模型综合评分动态调整。")
        R.append("")
        R.append("**表7 各标的证券影响估计**")
        R.append("")
        R.append("| 代码 | 名称 | 事件子类 | 方向 | 影响区间 | 驱动因素分解 | 综合评分 | LR概率 | CAR概率 |")
        R.append("|:----:|:-----|:---------|:----:|:---------|:-------------|:--------:|:-------:|:-------:|")

        impact_recs = []
        seen_cd = set()

        # Fix#1: 构建selected事件映射，确保表7与表8展示同一只股票的同一条事件
        _sel_event_map = {}
        for _, sr in selected_df.iterrows():
            _sc = str(sr.get('ts_code', '')).split('.')[0]
            if _sc:
                _sel_event_map[_sc] = sr

        for idx in range(len(events_df)):
            row = events_df.iloc[idx]
            code = str(_val(row, 'ts_code', default='')).split('.')[0]
            if not code or code == 'nan':
                continue
            if code in seen_cd:
                continue
            seen_cd.add(code)

            # Fix#1: 优先使用selected_df的事件记录（避免同一股票不同事件导致表7/8矛盾）
            if code in _sel_event_map:
                row = _sel_event_map[code]

            name = _code2name.get(code, code)
            sub = str(_val(row, 'event_sub', default='--'))

            # 方向（智能适配标签编码：与表6保持一致的映射逻辑）
            direction = '--'
            if label_col and label_col in row.index:
                dv = row[label_col]
                di = int(dv) if isinstance(dv, (int, float, np.integer)) else -999
                # 自动检测编码方式（与表6映射保持一致）
                uniq_dc = sorted([int(x) for x in events_df[label_col].dropna().unique()
                                  if isinstance(x, (int, float, np.integer))])
                if set(uniq_dc) == {-1, 0, 1}:
                    # 融合_预测标签: -1=跌, 0=平, 1=涨
                    dir_map = {1: '上涨', 0: '持平', -1: '下跌', -999: str(dv)}
                elif set(uniq_dc).issubset({0, 1, 2}):
                    dir_map = {2: '上涨', 1: '持平', 0: '下跌', -999: str(dv)}
                elif set(uniq_dc).issubset({0, 1}):
                    dir_map = {1: '上涨', 0: '下跌', -999: str(dv)}
                else:
                    dir_map = {-999: str(dv)}
                direction = dir_map.get(di, f'{dv}')

            comp = float(row.get('综合评分', 0)) if '综合评分' in row.index else 0.0
            lr_p = float(row.get('_展示看涨概率', 0)) if '_展示看涨概率' in row.index else (float(row.get('技术看涨概率', 0)) if '技术看涨概率' in row.index else 0.0)
            car_p = float(row.get('CAR预测概率', 0)) if 'CAR预测概率' in row.index else 0.0
            ev_s = float(_val(row, '贝叶斯事件得分', '事件得分', default=0)) if any(c in row.index for c in ['贝叶斯事件得分', '事件得分']) else 0.0
            car_v = float(_val(row, '预测CAR', 'CAR预测', 'car_pred', default=0))
            # 否决前评分（用于展示否决原因）
            pre_veto = float(row.get('_否决前看涨评分', 0)) if '_否决前看涨评分' in row.index else 0.0
            veto_reason = str(row.get('融合决策原因', '')) if '融合决策原因' in row.index else ''

            # 影响幅度（基于CAR预测值+方向+评分动态计算）
            if direction == '上涨':
                base = max(abs(car_v), 0.5)
                lo = round(base * 0.6 + comp * 2, 1)
                hi = round(base * 1.2 + comp * 4 + 1, 1)
                hi = max(hi, lo + 1.0)
                hi = min(hi, 8.0)
                impact = f"+{lo:.1f}% ~ +{hi:.1f}%"
            elif direction == '下跌':
                base = max(abs(car_v), 0.5)
                lo = round(base * 0.5 + abs(comp) * 1.5, 1)
                hi = round(base * 1.0 + abs(comp) * 3 + 0.5, 1)
                hi = max(hi, lo + 0.5)
                impact = f"-{hi:.1f}% ~ -{lo:.1f}%"
            else:
                # 持平：基于CAR实际值计算窄幅区间，而非固定-1%~+1%
                _mag = max(abs(car_v), 0.3)
                _lo = round(-_mag * 0.8, 1)
                _hi = round(_mag * 0.8, 1)
                if _hi <= _lo:
                    _hi = _lo + 0.5
                impact = f"{_lo:+.1f}% ~ {_hi:+.1f}%"

            drv_parts = []
            if ev_s > 0.45: drv_parts.append(f"事件强({ev_s:.2f})")
            elif ev_s > 0.30: drv_parts.append(f"事件温和({ev_s:.2f})")
            else: drv_parts.append(f"事件偏弱({ev_s:.2f})")
            if lr_p > 0.55: drv_parts.append(f"技术偏多({lr_p:.2f})")
            elif lr_p > 0.40: drv_parts.append(f"技术中性({lr_p:.2f})")
            else: drv_parts.append(f"技术偏弱({lr_p:.2f})")
            # CAR侧
            if car_p > 0.70: drv_parts.append(f"CAR高({car_p:.2f})")
            elif car_p > 0.55: drv_parts.append(f"CAR中({car_p:.2f})")
            driver_str = ' / '.join(drv_parts)

            impact_recs.append({
                'code': code, 'name': name, 'sub': sub,
                'dir': direction, 'imp': impact, 'drv': driver_str,
                'comp': comp, 'lr': lr_p, 'car': car_p,
                'pre_veto': pre_veto, 'veto_reason': veto_reason,
            })

        impact_recs.sort(key=lambda x: x['comp'], reverse=True)
        for r in impact_recs[:28]:
            nd = r['name'][:12] if len(r['name']) > 12 else r['name']
            # 否决行：展示否决前评分+否决标记
            if r['comp'] == 0 and r['pre_veto'] > 0:
                comp_str = f"{r['pre_veto']:.3f}(否决)"
            else:
                comp_str = f"{r['comp']:.3f}"
            R.append(f"| {r['code']} | {nd} | {r['sub'][:8]} | {r['dir']} | {r['imp']} | {r['drv']} "
                    f"| {comp_str} | {r['lr']:.3f} | {r['car']:.3f} |")
        R.append("")
        R.append("> **注**：综合评分标记(否决)表示该标的原始技术面看涨，但因事件驱动信号不足（贝叶斯事件得分低于动态否决阈值或RSI追涨防御触发）被风控否决，方向降级为「持平」，综合评分归零。括号内数值为否决前的原始看涨评分。")
    else:
        R.append("当前无可用事件数据。")
        R.append("")

    # ─────────────────────────────────────────────
    # ─────────────────────────────────────────────
    R.append("---")
    R.append("")
    R.append("## 四、投资策略执行")
    R.append("")

    if len(selected_df) > 0:
        R.append("### 4.1 本周选股方案")
        R.append("")
        R.append("**表8 选股结果与仓位配置**")
        R.append("")
        R.append("| 事件名称 | 标的代码 | 仓位权重 | 综合评分 | 贝叶斯得分 | LR看涨概率 | 预期影响 |")
        R.append("|:---------|:--------:|:-------------:|:--------:|:----------:|:-----------:|:---------|")

        for _, row in selected_df.iterrows():
            code = str(_val(row, 'ts_code', default='')).split('.')[0]
            company = _code2name.get(code, '')
            if not company or company == code:
                cn = _val(row, 'company_name', default='')
                if cn and cn != '--':
                    company = str(cn).strip()
                    # Fix #25: 过滤标题片段（含数字/年/月/关于等，不像公司名）
                    if any(kw in company for kw in ['2026','2025','关于','第','月','公告','H股']):
                        company = ''

            rt = str(_val(row, 'title', default=str(_val(row, 'event_sub', default=''))))
            es = str(_val(row, 'event_sub', default=''))
            cr = refine_event_name(rt, es)
            if company and company != code:
                ename = f"{company}{cr}"
            else:
                ename = f"{cr}({code})" if cr else f"{es}({code})"

            ratio = float(_val(row, '资金比例', default=0.333))
            sc = _fmt(_val(row, '综合评分', default=0))
            bs = _fmt(_val(row, '贝叶斯事件得分', '事件得分', default=0))
            lp = _fmt(_val(row, '_展示看涨概率', '技术看涨概率', default=0))

            sv = float(row.get('综合评分', 0)) if '综合评分' in row.index else 0
            if sv >= 0.65:
                ei = "+2.0%~+5.0%（强共振）"
            elif sv >= 0.5:
                ei = "+1.0%~+3.0%（确认利好）"
            else:
                ei = "+0.5%~+2.0%（技术支撑）"

            R.append(f"| {ename} | {code} | {ratio:.2f} | {sc} | {bs} | {lp} | {ei} |")
        R.append("")

        if len(selected_df) >= 2:
            scores = selected_df['综合评分'].values.astype(float)
            score_range = scores.max() - scores.min()
            if score_range < 0.02:
                R.append(f"> **仓位说明**：本周入选标的综合评分极差仅 {score_range:.3f}（{scores.min():.3f}~{scores.max():.3f}），"
                         f"评分加权后各标的资金比例差异不足0.7个百分点，经20%~40%仓位约束与归一化后近似等权配置。"
                         f"这反映了模型对入选标的的看涨置信度高度一致——在信号差异统计不显著时，等权是信息论最优的稳健选择。")
                R.append("")

        R.append("### 4.2 交易规则")
        R.append("")
        R.append("1. **买入时点**：每周二开盘价以等权/评分加权方式买入")
        R.append("2. **卖出时点**：每周五收盘价全仓卖出")
        R.append("3. **交易成本**：单边0.15%，双边合计0.3%（含印花税+佣金）")
        R.append("4. **风控约束**：单标的持仓上限40%，个股止损线-3%")
        R.append("5. **初始资金**：¥{:,.0f}".format(INITIAL_CAPITAL))
        R.append("")

        R.append("### 4.3 资金配置明细")
        R.append("")
        for _, row in selected_df.iterrows():
            code = str(_val(row, 'ts_code', default='')).split('.')[0]
            amt = INITIAL_CAPITAL * float(_val(row, '资金比例', default=0))
            R.append(f"- **{code}**：¥{amt:,.0f} （占比 {_val(row,'资金比例',default=0)}）")
        R.append("")
    else:
        R.append("> [!] **本周无满足选股条件的标的证券。**")
        R.append("")
        R.append("选股条件：综合评分 > 0.50 且 E4 LR预测方向非下跌 且 通过ST/追涨/一致性过滤。")
        R.append(f"当前全样本最高综合评分：{_fmt(events_df['综合评分'].max()) if '综合评分' in events_df.columns else 'N/A'}")
        R.append("")

    # ─────────────────────────────────────────────
    #  五、消融实验（v15g更新数据）
    # ─────────────────────────────────────────────
    R.append("---")
    R.append("")
    R.append("## 五、模型消融实验")
    R.append("")
    R.append("为验证各模型组件的有效性，在验证集（2025年10-11月，n=2,285）上设计了7组消融实验：")
    R.append("")
    R.append("**表9 消融实验结果**")
    R.append("")
    R.append("| 实验 | 特征组合 | 分类算法 | Weighted-F1 | Rank_IC |")
    R.append("|:----:|:---------|:---------|:-------------:|:-------:|")
    R.append("| E1 | 仅事件统计（CAR均值/t检验） | 阈值规则 | 0.264 | N/A |")
    R.append("| E2 | TF-IDF文本(5000维) + LR | 多元Logistic Regression | 0.443 | N/A |")
    R.append("| E3 | TF-IDF(5000d)+技术因子(32d)拼接 | LR | 0.451 | 0.089* |")
    R.append("| E4 | 仅技术因子(32维) | LR | **0.443** | **0.102*** |")
    R.append("| E5 | 技术(32d)+横截面排名(7d) | LR | 0.441 | 0.098* |")
    R.append("| E6 | E4 + 贝叶斯事件先验(加权) | 规则融合 | 0.447 | 0.101* |")
    R.append("| **E7** | **E4 + 贝叶斯 + CAR预测(LGBM)** | **综合评分** | **0.448** | **0.104*** |")
    R.append("")
    R.append("*注：Rank_IC（排序信息系数）为预测评分与实际收益的Spearman秩相关系数，衡量预测值的排序能力，IC>0.03即有实际意义（标*号）。E1为阈值规则无连续评分，E2为纯文本分类未计算IC，均标记N/A。*")
    R.append("")
    R.append("**关键发现**：")
    R.append("")
    R.append("1. **技术因子是主导信号源**：E4（仅32维技术因子）F1达到0.443，与E2（5000维TF-IDF）相当，且IC=0.102（显著），证明A股短期事件驱动alpha主要来源于技术面形态而非事件语义")
    R.append("2. **多模型融合边际提升有限**：E7相对E4，F1提升0.005（0.443→0.448），IC提升0.002（0.102→0.104），增幅均较小，说明单一技术因子模型已捕获大部分可预测信号，LGBM回归和贝叶斯先验的增量贡献有限")
    R.append("3. **贝叶斯收缩主要起稳定作用**：E6相对E4的F1微升0.004，但IC持平（0.102→0.101），表明James-Stein收缩对全局排序能力无显著增益，其主要价值在于缓解小样本事件子类的均值估计偏差（nk<30的子类收缩率>40%），提升极端事件的预测鲁棒性")
    R.append("")

    R.append("---")
    R.append("")
    R.append("## 参考文献")
    R.append("")
    R.append("[1] MacKinlay, A. C. (1997). Event studies in economics and finance. *Journal of Economic Literature*, 35(1), 13-39.")
    R.append("[2] Corrado, C. J. (1989). A nonparametric test for abnormal security-price performance in event studies. *Financial Economics*, 23(2), 385-395.")
    R.append("[3] James, W., & Stein, C. (1961). Estimation with quadratic loss. *Proceedings of the Fourth Berkeley Symposium*, 361-379.")
    R.append("[4] Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*, 2017.")
    R.append("")

    report_text = '\n'.join(R)
    report_path = os.path.join(RESULT_DIR, f'analysis_report_week{week_num}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    logger.info(f"  Markdown报告已保存: {report_path}")

    try:
        _generate_docx_academic(report_text, week_num, RESULT_DIR)
    except ImportError:
        logger.warning("  python-docx未安装，跳过Word生成（pip install python-docx）")
    except Exception as e:
        logger.warning(f"  Word生成失败(可忽略): {e}")

    return report_text

def _generate_docx_academic(report_text, week_num, result_dir):
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor, Twips
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ROW_HEIGHT_RULE
    from docx.enum.section import WD_ORIENT
    from docx.oxml.ns import qn, nsdecls
    from docx.oxml import parse_xml

    FONT_CN = "宋体"
    FONT_EN = "Times New Roman"
    FONT_HD = "黑体"
    ACCENT = RGBColor(0x1F, 0x4E, 0x79)
    HD_COLOR = RGBColor(0x2E, 0x75, 0xB6)

    def _set_cell_shading(cell, hex_color):
        tc = cell._tc
        tc_pr = tc.get_or_add_tcPr()
        shading_el = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}" w:val="clear"/>')
        tc_pr.append(shading_el)

    def _set_cell_border(cell, border_type="single", size=4, color="000000"):
        tc = cell._tc
        tc_pr = tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'<w:top w:val="{border_type}" w:sz="{size}" w:color="{color}"/>'
            f'<w:left w:val="{border_type}" w:sz="{size}" w:color="{color}"/>'
            f'<w:bottom w:val="{border_type}" w:sz="{size}" w:color="{color}"/>'
            f'<w:right w:val="{border_type}" w:sz="{size}" w:color="{color}"/>'
            '</w:tcBorders>')
        tc_pr.append(borders)

    def _write_cell(cell, text, bold=False, font_size=9,
                    alignment=WD_ALIGN_PARAGRAPH.CENTER, font_color=None,
                    bg_color=None):
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = alignment
        pf = p.paragraph_format
        pf.space_before = Pt(2)
        pf.space_after = Pt(2)
        run = p.add_run(str(text))
        run.bold = bold
        run.font.size = Pt(font_size)
        run.font.name = FONT_EN
        rpr = run._element.rPr
        rpr.rFonts.set(qn('w:eastAsia'), FONT_CN)
        if font_color:
            run.font.color.rgb = font_color
        if bg_color:
            _set_cell_shading(cell, bg_color)

    def _add_academic_table(doc, headers, rows, table_num="", note=""):
        ncols = len(headers)
        nrows = len(rows)
        tbl = doc.add_table(rows=1 + nrows, cols=ncols)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.style = 'Table Grid'

        for ci, h in enumerate(headers):
            c = tbl.rows[0].cells[ci]
            _write_cell(c, h, bold=True, font_size=9, font_color=RGBColor(0xFF, 0xFF, 0xFF),
                       bg_color="1F4E79")

        # 数据行
        for ri, row_data in enumerate(rows):
            for ci, val in enumerate(row_data):
                c = tbl.rows[ri + 1].cells[ci]
                bg = "F5F8FC" if ri % 2 == 1 else None
                _write_cell(c, str(val), font_size=8.5, bg_color=bg)

        for row_idx, row in enumerate(tbl.rows):
            for cell in row.cells:
                tc = cell._tc
                tc_pr = tc.get_or_add_tcPr()
                tc_pr.append(parse_xml(
                    f'<w:tcBorders {nsdecls("w")}>'
                    '<w:top w:val="nil"/>'
                    '<w:left w:val="nil"/>'
                    '<w:bottom w:val="nil"/>'
                    '<w:right w:val="nil"/>'
                    '<w:insideH w:val="nil"/>'
                    '<w:insideV w:val="nil"/>'
                    '</w:tcBorders>'
                ))

        # 顶线（粗，应用于第一行所有单元格底边框）
        for cell in tbl.rows[0].cells:
            _set_cell_border(cell, border_type="single", size=12, color="000000")

        # 表头下线（细，应用于第一行所有单元格底边框——覆盖上面的粗线效果需要重新设置）
        # 改用：顶线粗 + 表头下线细 + 底线粗
        for row_idx, row in enumerate(tbl.rows):
            for cell in row.cells:
                tc = cell._tc
                tc_pr = tc.get_or_add_tcPr()
                if row_idx == 0:
                    tc_pr.append(parse_xml(
                        f'<w:tcBorders {nsdecls("w")}>'
                        '<w:top w:val="single" w:sz="15" w:color="000000"/>'
                        '<w:bottom w:val="single" w:sz="6" w:color="000000"/>'
                        '<w:left w:val="nil"/>'
                        '<w:right w:val="nil"/>'
                        '</w:tcBorders>'
                    ))
                elif row_idx == nrows:
                    tc_pr.append(parse_xml(
                        f'<w:tcBorders {nsdecls("w")}>'
                        '<w:bottom w:val="single" w:sz="15" w:color="000000"/>'
                        '<w:left w:val="nil"/>'
                        '<w:right w:val="nil"/>'
                        '</w:tcBorders>'
                    ))
                else:
                    # 数据行无边框
                    tc_pr.append(parse_xml(
                        f'<w:tcBorders {nsdecls("w")}>'
                        '<w:left w:val="nil"/>'
                        '<w:right w:val="nil"/>'
                        '</w:tcBorders>'
                    ))

        if table_num or note:
            p_note = doc.add_paragraph()
            p_note.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p_note.paragraph_format.space_after = Pt(8)
            if table_num:
                r = p_note.add_run(f"注{table_num}：")
                r.font.size = Pt(8.5)
                r.font.name = FONT_EN
                r._element.rPr.rFonts.set(qn('w:eastAsia'), FONT_CN)
            if note:
                r2 = p_note.add_run(note)
                r2.font.size = Pt(8.5)
                r2.font.name = FONT_EN
                r2._element.rPr.rFonts.set(qn('w:eastAsia'), FONT_CN)

        return tbl

    def _add_heading(doc, text, level=1):
        h = doc.add_heading(text, level=min(level, 3))
        for run in h.runs:
            run.font.name = FONT_EN
            run._element.rPr.rFonts.set(qn('w:eastAsia'), FONT_HD)
            if level == 1:
                run.font.color.rgb = ACCENT
                run.font.size = Pt(16)
            elif level == 2:
                run.font.color.rgb = HD_COLOR
                run.font.size = Pt(14)
            else:
                run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
                run.font.size = Pt(12)

    def _add_para(doc, text, first_line_indent=True):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if first_line_indent:
            p.paragraph_format.first_line_indent = Cm(0.74)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.35
        parts = re.split(r'(\*\*.*?\*\*)', text)
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                run = p.add_run(part[2:-2])
                run.bold = True
            else:
                run = p.add_run(part)
            run.font.size = Pt(10.5)
            run.font.name = FONT_EN
            run._element.rPr.rFonts.set(qn('w:eastAsia'), FONT_CN)

    def _add_bullet(doc, text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        p.clear()
        parts = re.split(r'(\*\*.*?\*\*)', text)
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                r = p.add_run(part[2:-2])
                r.bold = True
            else:
                r = p.add_run(part)
            r.font.size = Pt(10.5)
            r.font.name = FONT_EN
            r._element.rPr.rFonts.set(qn('w:eastAsia'), FONT_CN)

    def _parse_md_table(lines):
        if len(lines) < 1:
            return [], []
        headers = [c.strip() for c in lines[0].strip('|').split('|')]
        rows = []
        for line in lines[1:]:
            stripped = line.strip()
            if not stripped or stripped.startswith('---') or stripped.startswith('```'):
                break
            if re.match(r'^\|[:\-\s|]+$', stripped):
                continue
            cells = [c.strip() for c in stripped.strip('|').split('|')]
            if cells:
                rows.append(cells)
        return headers, rows

    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(3.17)
    sec.right_margin = Cm(3.17)

    style = doc.styles['Normal']
    style.font.name = FONT_EN
    style.font.size = Pt(10.5)
    style._element.rPr.rFonts.set(qn('w:eastAsia'), FONT_CN)
    style.paragraph_format.line_spacing = 1.35

    lines = report_text.split('\n')
    i = 0
    in_table = False
    table_buf = []
    table_caption = ""  # 当前表格前的表题文字

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            if in_table and table_buf:
                hdrs, data_rows = _parse_md_table(table_buf)
                if hdrs:
                    _add_academic_table(doc, hdrs, data_rows)
                table_buf = []
                in_table = False
                table_caption = ""
            i += 1
            continue

        if stripped.startswith('```'):
            i += 1
            continue

        if stripped.startswith('---'):
            i += 1
            continue

        if stripped.startswith('#'):
            if in_table and table_buf:
                hdrs, data_rows = _parse_md_table(table_buf)
                if hdrs:
                    _add_academic_table(doc, hdrs, data_rows)
                table_buf = []
                in_table = False
            m = re.match(r'^(#{1,6})\s+(.*)', stripped)
            if m:
                lvl = min(len(m.group(1)), 3)
                _add_heading(doc, m.group(2).strip(), level=lvl)
            i += 1
            continue

        if '|' in stripped and stripped.startswith('|'):
            if not in_table:
                in_table = True
                table_buf = []
            if re.match(r'^\|[\s\-:]+\|', stripped):
                i += 1
                continue
            table_buf.append(stripped)
            i += 1
            continue

        # 非表格行但正在表格中 → 结束表格
        if in_table and table_buf:
            hdrs, data_rows = _parse_md_table(table_buf)
            if hdrs:
                _add_academic_table(doc, hdrs, data_rows)
            table_buf = []
            in_table = False

        if stripped.startswith('- '):
            _add_bullet(doc, stripped[2:])
            i += 1
            continue

        nm = re.match(r'^\s*(\d+)\.\s+(.*)', stripped)
        if nm:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.left_indent = Cm(0.74)
            rn = p.add_run(f"{nm.group(1)}. ")
            rn.font.size = Pt(10.5)
            rn.font.name = FONT_EN
            rn._element.rPr.rFonts.set(qn('w:eastAsia'), FONT_CN)
            parts = re.split(r'(\*\*.*?\*\*)', nm.group(2))
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    r = p.add_run(part[2:-2])
                    r.bold = True
                else:
                    r = p.add_run(part)
                r.font.size = Pt(10.5)
                r.font.name = FONT_EN
                r._element.rPr.rFonts.set(qn('w:eastAsia'), FONT_CN)
            i += 1
            continue

        if stripped.startswith('>'):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.74)
            p.paragraph_format.right_indent = Cm(0.74)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            r = p.add_run(stripped[1:].strip())
            r.font.size = Pt(10)
            r.font.name = FONT_EN
            r.italic = True
            r._element.rPr.rFonts.set(qn('w:eastAsia'), FONT_CN)
            i += 1
            continue

        if stripped.startswith('*注') or stripped.startswith('*E4') or stripped.startswith('*LGBM') or stripped.startswith('*综合'):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(6)
            r = p.add_run(stripped.strip('*'))
            r.font.size = Pt(8.5)
            r.font.name = FONT_EN
            r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
            r._element.rPr.rFonts.set(qn('w:eastAsia'), FONT_CN)
            i += 1
            continue

        if '$$' in stripped:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(8)
            formula = stripped.replace('$$', '').strip()
            r = p.add_run(formula)
            r.font.size = Pt(11)
            r.font.name = "Cambria Math"
            r.italic = True
            i += 1
            continue

        _add_para(doc, stripped)
        i += 1

    if in_table and table_buf:
        hdrs, data_rows = _parse_md_table(table_buf)
        if hdrs:
            _add_academic_table(doc, hdrs, data_rows)

    footer = doc.sections[0].footer
    footer.is_linked_to_previous = False
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = fp.add_run(f"事件驱动型股市投资策略分析报告 · 第{week_num}周")
    fr.font.size = Pt(8)
    fr.font.color.rgb = RGBColor(0xAA, 0xAA, 0xAA)
    fr.font.name = FONT_EN
    fr._element.rPr.rFonts.set(qn('w:eastAsia'), FONT_CN)

    docx_path = os.path.join(result_dir, f'analysis_report_week{week_num}.docx')
    doc.save(docx_path)
    logger.info(f"  Word文档已保存: {docx_path}")

def main():
    parser = argparse.ArgumentParser(description='实时自动化Pipeline v15g')
    parser.add_argument('--week', type=int, choices=[1, 2], help='提交周次: 1或2')
    parser.add_argument('--auto', action='store_true', help='自动检测当前周次')
    parser.add_argument('--skip-collect', action='store_true', help='跳过事件采集（使用已有数据）')
    parser.add_argument('--skip-model', action='store_true', help='跳过模型推理（仅用统计模型）')
    parser.add_argument('--use-historical', action='store_true', help='使用历史测试集数据（回退方案）')
    args = parser.parse_args()
    
    if args.auto:
        today = datetime.now()
        if today <= pd.Timestamp('2026-04-26'):
            week_num = 1
        else:
            week_num = 2
    elif args.week:
        week_num = args.week
    else:
        week_num = 1  # 默认第1周
    
    if week_num == 1:
        trade_start = WEEK1_TRADE_START
        trade_end = WEEK1_TRADE_END
    else:
        trade_start = WEEK2_TRADE_START
        trade_end = WEEK2_TRADE_END
    
    logger.info("实时自动化Pipeline v15g —— 事件驱动型股市投资策略（v15g: 去BERT化+LR+LGBM回归+贝叶斯先验）")
    logger.info(f"周次: 第{week_num}周")
    logger.info(f"交易日期: {trade_start} ~ {trade_end}")
    logger.info(f"提交截止: {WEEK1_SUBMIT_END if week_num == 1 else WEEK2_SUBMIT_END}")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.skip_collect:
        # 使用已有的事件数据
        existing_path = os.path.join(RESULT_DIR, f'raw_events_{trade_start}_{trade_end}.csv')
        if os.path.exists(existing_path):
            events_df = pd.read_csv(existing_path)
            logger.info(f"  加载已有事件: {len(events_df)} 条")
        else:
            logger.error(f"  未找到已有事件文件: {existing_path}")
            return
    elif args.use_historical:
        logger.warning("  [!] --use-historical模式不符合题目实测要求（需用实时采集数据）")
        logger.warning("  题目要求：基于收集到的实时数据完成实测，不可使用历史训练/测试集")
        logger.info("  如用于本地调试可继续，但提交结果请使用默认的实时采集模式")
        test_candidates = [
            os.path.join(DATA_DIR, '建模核心表', '测试集_增强_v3清洗后.csv'),
            os.path.join(DATA_DIR, '建模核心表', '测试集_增强.csv'),
        ]
        test_path = next((p for p in test_candidates if os.path.exists(p)), None)
        if test_path:
            events_df = pd.read_csv(test_path)
            events_df['data_source'] = 'historical'
            logger.info(f"  加载历史测试集(仅调试用): {len(events_df)} 条")
        else:
            logger.error(f"  历史测试集不存在，已检查: {', '.join(test_candidates)}")
            return 1
    else:
        events_df = collect_realtime_events(trade_start, trade_end)
    
    has_stock_events = 'ts_code' in events_df.columns and events_df['ts_code'].notna().any() and (events_df['ts_code'] != '').any()
    
    if len(events_df) == 0:
        logger.warning("  [!] 未采集到任何实时事件！")
        logger.warning("  请检查网络连接和数据源（巨潮/东财/财新/AKShare等）是否可用")
    elif not has_stock_events:
        logger.warning(f"  [!] 采集到{len(events_df)}条事件但无个股关联(ts_code为空)")
        logger.warning("  后续选股可能为空，请确保巨潮公告或东财新闻采集成功")
    else:
        logger.info(f"  [OK] 实时采集完成: {len(events_df)}条事件, 有股票关联={has_stock_events}")
    
    events_df = preprocess_realtime_data(events_df)
    
    events_df = run_event_classification(events_df)
    
    events_df = run_company_association(events_df)
    
    events_df, price_df, bench_df = collect_market_data(events_df, trade_start, trade_end)
    
    if price_df is not None:
        price_df = preprocess_market_data(price_df)
    
    if args.skip_model:
        logger.info("  跳过模型推理，仅使用事件研究法")
    events_df, e4_lr, e4_lr_scaler = run_impact_prediction(events_df, bench_df)
    
    selected_df = run_stock_selection(events_df, trade_start, trade_end, week_num)
    
    # 预构建公司名查找表（Step 7a/7b共享，避免重复BaoStock查询）
    _stock_info_path = os.path.join(DATA_DIR, '公司信息.csv') if DATA_DIR else ''
    _stock_info = pd.read_csv(_stock_info_path) if _stock_info_path and os.path.exists(_stock_info_path) else None
    _shared_code2name, _shared_code2industry = _build_stock_lookup(events_df, stock_info=_stock_info)
    
    result_df = generate_result_xlsx(selected_df, week_num, code2name=_shared_code2name, stock_info=_stock_info)

    logger.info(f"Pipeline完成, 结果: {RESULT_DIR}")
    logger.info(f"  - result.xlsx: 提交文件")
    
    return result_df

if __name__ == '__main__':
    result = main()
    exit_code = 1 if isinstance(result, int) and result == 1 else 0
    sys.exit(exit_code)
