# -*- coding: utf-8 -*-
import os, sys, warnings, re, time, argparse, glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.font_manager as fm
from scipy import stats
warnings.filterwarnings('ignore')


def _find_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isdir(os.path.join(d, '数据集_清洗')) or os.path.isdir(os.path.join(d, '结果（图表，分析）')):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))

ROOT = _find_root()
DATA_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get('TEDDY_DATA_DIR') or os.path.join(ROOT, '数据集_清洗')
))


def _find_data(name_hints, search_dirs=None):
    if search_dirs is None:
        search_dirs = [
            '', DATA_DIR, os.path.join(DATA_DIR, '建模核心表'),
            '训练结果', '训练结果/5.3_事件影响预测',
            '训练结果/5.4_投资策略构建_扩展',
            '训练结果', '训练结果/5.4_投资策略构建_扩展',
        ]
    for hint in ([name_hints] if isinstance(name_hints, str) else name_hints):
        for sd in search_dirs:
            p = sd if os.path.isabs(sd) else os.path.join(ROOT, sd)
            if os.path.isdir(p):
                matches = glob.glob(os.path.join(p, f'*{hint}*'))
                if matches:
                    return sorted(matches, key=lambda x: (
                        0 if os.path.basename(x) == hint + '.csv' else 1,
                        len(os.path.basename(x))
                    ))[0]
    return ''


D_MAIN = _find_data(['事件公司股价三元组'])
D_TRAIN = _find_data(['训练集'])
D_VAL   = _find_data(['验证集'])
D_TEST  = _find_data(['测试集'])

E4_DIR = ''
for c in ['5.3_事件影响预测']:
    p = os.path.join(ROOT, '云端训练结果', c)
    if os.path.isdir(p): E4_DIR = p; break

BT_DIR = ''
for c in ['5.4_投资策略构建_扩展']:
    p = os.path.join(ROOT, '云端训练结果', c)
    if os.path.isdir(p): BT_DIR = p; break
    p2 = os.path.join(ROOT, '训练结果', c)
    if os.path.isdir(p2): BT_DIR = p2; break

OUT = {}
for k, v in {'eda':'第三章_数据描述与探索性分析/图形','t1':'第六章_任务一求解',
             't2':'第七章_任务二求解','t3':'第八章_任务三求解','t4':'第九章_任务四求解',
             'rpt':'第三章_数据描述与探索性分析/报表'}.items():
    OUT[k] = os.path.join(ROOT, '结果（图表，分析）', v)
    os.makedirs(OUT[k], exist_ok=True)

WC = 'C:/Windows/Fonts/simhei.ttf'

CN = {
  '000001.SZ':'平安银行','000002.SZ':'万科A','000063.SZ':'中兴通讯',
  '000333.SZ':'美的集团','000338.SZ':'潍柴动力','000651.SZ':'格力电器',
  '000725.SZ':'京东方A','000858.SZ':'五粮液','002352.SZ':'顺丰控股',
  '002415.SZ':'海康威视','002475.SZ':'立讯精密','002594.SZ':'比亚迪',
  '300750.SZ':'宁德时代','600036.SH':'招商银行','600519.SH':'贵州茅台',
  '600887.SH':'伊利股份','601318.SH':'中国平安','601888.SH':'中国中免',
  '601985.SH':'中国核电','688981.SH':'中芯国际','300871.SZ':'晶方科技'
}

plt.rcParams.update({'figure.dpi':150,'savefig.dpi':300,'figure.facecolor':'white',
  'axes.linewidth':0.8,'axes.grid':True,'grid.color':'#E0E0E0','grid.linestyle':'--',
  'grid.linewidth':0.5,'xtick.direction':'in','ytick.direction':'in'})


def sf():
    avail = {f.name for f in fm.fontManager.ttflist}
    for c in ['SimHei','Microsoft YaHei','WenQuanYi Micro Hei','STHeiti']:
        if c in avail:
            plt.rcParams['font.sans-serif'] = [c]
            plt.rcParams['axes.unicode_minus'] = False
            return c
    return None


def sv(fig,nm,g='eda'):
    p = os.path.join(OUT[g], nm)
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  OK {nm}')
    return p


def sp(ax,cats,vals):
    s = np.array(vals, float); t = s.sum()
    ax.pie(s, labels=cats,
           autopct=lambda p: '' if p/100.*t < t*.03 else f'{p:.1f}%',
           explode=[0.02 if x/s.sum()<0.05 else 0 for x in s],
           labeldistance=1.12, pctdistance=0.75)


_m=_tr=_va=_te=None; _dl=False

def ld():
    global _m,_tr,_va,_te,_dl
    if _dl: return
    def r(p): return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    _m=r(D_MAIN);_tr=r(D_TRAIN);_va=r(D_VAL);_te=r(D_TEST)
    print(f'数据:主{len(_m)} 训{len(_tr)} 验{len(_va)} 测{len(_te)}'); _dl=True

def pdt(col):
    for fmt in ('%Y%m%d','%Y-%m-%d'):
        try:
            d=pd.to_datetime(col,format=fmt,errors='coerce')
            if d.notna().mean()>0.5: return d
        except: pass
    return pd.to_datetime(col,errors='coerce')

def e01():
    sf(); ld(); df = _m.copy(); df['_d'] = pdt(df['event_date']); df = df[df['_d'].notna()]
    mc = df['_d'].dt.to_period('M').value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(14, 5.5)); xs = range(len(mc))
    b = ax.bar(xs, mc.values, color='#2B83BA', alpha=.75, width=.75)
    mi = mc.values.argmax(); b[mi].set_color('#D73027')
    ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in mc.index], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('月份'); ax.set_ylabel('事件数')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.text(.98,.97,f'总计{len(df)}\n峰值{mc.index[mi]}({mc.values[mi]})',
            transform=ax.transAxes,fontsize=8,ha='right',va='top',bbox=dict(fc='w',alpha=.9))
    plt.tight_layout()
    return sv(fig,'E01_事件时间分布.png')

def e02():
    sf(); ld(); fig, ax = plt.subplots(figsize=(9, 7))
    vc = _m['event_type'].value_counts()
    sp(ax,vc.index.tolist(),vc.values.tolist())
    ax.legend(loc='center left',bbox_to_anchor=(1.05,.5),fontsize=9,title='一级分类')
    plt.tight_layout()
    return sv(fig,'E02_一级分类分布.png')

def e03():
    sf(); ld(); fig, ax = plt.subplots(figsize=(10, 7))
    sub_col = 'event_sub' if 'event_sub' in _m.columns else 'event_type'
    vc = _m[sub_col].value_counts().head(15); y = range(len(vc))[::-1]
    ax.barh(y,vc.values,height=.65,color=plt.cm.Blues(np.linspace(.3,.85,len(vc)))[::-1])
    for i,v in enumerate(vc.values):
        ax.text(v+.02*max(vc),len(vc)-1-i,str(v),va='center',fontsize=9,fontweight='bold')
    ax.set_yticks(y); ax.set_yticklabels(vc.index[::-1],fontsize=9.5); ax.set_xlabel('事件数')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return sv(fig,'E03_二级分类分布.png')

def e04():
    sf(); ld(); fig, ax = plt.subplots(figsize=(8, 7))
    lc = next((c for c in ['label','label_v12'] if c in _m.columns),None)
    if not lc: return False
    vc = _m[lc].value_counts(); nm={-1:'跌(-1)',0:'平(0)',1:'涨(+1)'}
    lbs=[nm.get(int(k),str(k)) for k in vc.index]
    ax.pie(vc.values,labels=lbs,autopct=lambda p:f'{p:.1f}%',
           colors=['#1A9850','#FDB863','#D73027'][:len(vc)],
           explode=[.03]*len(vc),startangle=90,textprops={'fontsize':11})
    ax.legend(lbs,loc='center left',bbox_to_anchor=(1,.5),fontsize=10)
    plt.tight_layout()
    return sv(fig,'E04_标签分布.png')

def e05():
    sf(); ld(); fig, ax = plt.subplots(figsize=(9, 5.5))
    ds=[('训',len(_tr),'#2166AC'),('验',len(_va),'#FDB863'),('测',len(_te),'#D73027')]
    ns=[d[1] for d in ds]; cs=[d[2] for d in ds]; ls=[d[0] for d in ds]; tot=sum(ns)
    b=ax.bar(ls,ns,color=cs,alpha=.82,edgecolor='w',linewidth=.8)
    for bi,v in zip(b,ns):
        ax.text(bi.get_x()+bi.get_width()/2,v+.01*tot,f'{v}\n({v/tot*100:.1f}%)',ha='center',fontsize=10,fontweight='bold')
    ax.set_ylabel('样本数'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return sv(fig,'E05_数据集划分.png')

def e06():
    sf(); ld()
    if not os.path.exists(WC): return False
    import jieba
    from collections import Counter
    from wordcloud import WordCloud
    titles=_m['title'].dropna().astype(str).head(5000)
    words=[]
    for title in titles:
        clean=re.sub(r'[^\u4e00-\u9fa5]',' ',title)
        words.extend(w for w in jieba.cut(clean) if len(w)>=2)
    freq=dict(Counter(words).most_common(300))
    if freq:
        wc=WordCloud(font_path=WC,width=1200,height=800,background_color='white',max_words=120,colormap='Blues').generate_from_frequencies(freq)
        fig,ax=plt.subplots(figsize=(12,8));ax.imshow(wc,interpolation='bilinear');ax.axis('off')
        plt.tight_layout()
        return sv(fig,'E06_关键词词云.png')
    return False

def e07():
    sf(); ld(); fig, ax = plt.subplots(figsize=(10, 6))
    cc=next((c for c in ['CAR','CAR_new'] if c in _m.columns),None)
    if not cc: return False
    v=pd.to_numeric(_m[cc],errors='coerce').dropna()
    _,bs,ps=ax.hist(v,bins=50,color='#3498DB',alpha=.65,edgecolor='w',density=True)
    for p,l in zip(ps,bs[:-1]): p.set_facecolor('#D73027' if l>=0 else '#1A9850'); p.set_alpha(.55)
    mu,sig=v.mean(),v.std(); xr=np.linspace(v.min()-1,v.max()+1,200)
    ax.plot(xr,stats.norm.pdf(xr,mu,sig),'--',color='#666',linewidth=1.8,label=f'N({mu:.2f},{sig:.2f})')
    ax.axvline(0,color='#333',linewidth=1); ax.legend(fontsize=9)
    ax.set_xlabel('CAR(%)'); ax.set_ylabel('密度')
    ax.text(.97,.97,f'N={len(v)} mu={mu:.3f} sig={sig:.3f}',transform=ax.transAxes,fontsize=8.5,ha='right',family='monospace',bbox=dict(fc='w',alpha=.92))
    plt.tight_layout()
    return sv(fig,'E07_CAR直方图.png')

def e08():
    sf(); ld(); fig, ax = plt.subplots(figsize=(11, 9))
    cols=[c for c in _m.columns if any(t in c.lower() for t in ['ma','ret_','volatility','rsi','macd','boll','atr','vol_','gap_','alpha','beta']) and str(_m[c].dtype) in ['float64','float32','int64','int32']][:25]
    if len(cols)>=5:
        cm=_m[cols].corr().iloc[:20,:20]; im=ax.imshow(cm.values,cmap='RdBu_r',vmin=-1,vmax=1)
        for i in range(min(20,len(cm))):
            for j in range(min(20,len(cm))):
                val=cm.iloc[i,j]
                ax.text(j,i,f'{val:.2f}',ha='center',va='center',fontsize=6.5,color='white' if abs(val)>.6 else 'k')
        sn=[c.replace('close_t1','C1')[:8] for c in cm.columns]
        ax.set_xticks(range(len(cm))); ax.set_xticklabels(sn,rotation=70,fontsize=7)
        ax.set_yticks(range(len(cm))); ax.set_yticklabels(sn,fontsize=7)
        plt.colorbar(im,ax=ax,shrink=.82).set_label('Pearson r')
    plt.tight_layout()
    return sv(fig,'E08_特征热力图.png')

def e09():
    sf(); ld(); fig, axes=plt.subplots(2,2,figsize=(13,9))
    g={'MA偏离':[c for c in _m.columns if c.startswith('dev_MA')][:4],'动量':['ret_1d','ret_5d','ret_10d','ret_20d'],'波动':['volatility_20d','RSI14','ATR14'],'布林':['BOLL_upper','BOLL_lower','BOLL_width','BOLL_pos']}
    for idx,(gn,cs) in enumerate(g.items()):
        a=axes[idx//2][idx%2]; vc=[c for c in cs if c in _m.columns]
        if vc:
            bp=a.boxplot([_m[c].dropna() for c in vc],tick_labels=[c.replace('dev_','') for c in vc],patch_artist=True,widths=.55)
            for bx in bp['boxes']: bx.set_facecolor('#BBDEFB'); bx.set_alpha(.72)
            a.axhline(0,color='#666',ls='--',alpha=.5); a.set_title(gn,fontsize=11,fontweight='bold')
    plt.tight_layout()
    return sv(fig,'E09_技术因子箱线图.png')

def e10():
    sf(); ld(); fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
    ft='RSI14' if 'RSI14' in _m.columns else list(_m.select_dtypes(include=[np.number]).columns)[min(10,len(_m.columns)-1)]
    for ax,(nm,df) in zip([a1,a2],[('训',_tr),('测',_te)]):
        df[ft].dropna().hist(bins=40,ax=ax,alpha=.62,color='#2166AC',edgecolor='w',density=True)
        ax.set_title(f'{nm}{ft}',fontsize=11)
    ks,pv=stats.ks_2samp(_tr[ft].dropna(),_te[ft].dropna())
    a2.text(.95,.95,f'KS={ks:.3f} p={pv:.4f}',transform=a2.transAxes,fontsize=9,ha='right',bbox=dict(fc='w',alpha=.9))
    plt.tight_layout()
    return sv(fig,'E10_特征对比.png')

def e11():
    sf(); ld(); fig, ax=plt.subplots(figsize=(10,5.5))
    s=_m.head(200); im=ax.imshow(s.isnull().astype(int).T,cmap='YlOrRd',aspect='auto',vmin=0,vmax=1)
    ax.set_yticks(range(len(s.columns))); ax.set_yticklabels(s.columns,fontsize=7); ax.set_xlabel('样本(前200行)')
    plt.colorbar(im,ax=ax,shrink=.6).set_label('缺失')
    mp=(_m.isnull().sum()/len(_m)*100).sort_values().tail(10)
    ax.text(.99,.01,'\n'.join([f'{k}:{v:.1f}%' for k,v in mp.items()]),transform=ax.transAxes,fontsize=7,ha='right',family='monospace',bbox=dict(fc='w',alpha=.92))
    plt.tight_layout()
    return sv(fig,'E11_缺失值矩阵.png')

def e12():
    sf(); ld(); fig, ax=plt.subplots(figsize=(10,5.5))
    ln=_m['title'].fillna('').str.len()
    ax.hist(ln,bins=40,color='#8E44AD',alpha=.65,edgecolor='w')
    for xv,lb,cl in zip([ln.mean(),ln.median()],['均值','中位'],['#D73027','#2980B9']):
        ax.axvline(x=xv,color=cl,linewidth=2,ls='--',label=f'{lb}{xv:.0f}')
    ax.set_xlabel('字符数'); ax.legend(fontsize=9)
    plt.tight_layout()
    return sv(fig,'E12_文本长度.png')

def e13():
    sf(); ld(); fig, ax=plt.subplots(figsize=(8,7))
    vc=_m['event_type'].value_counts().head(8)
    sp(ax,vc.index.tolist(),vc.values.tolist())
    ax.legend(loc='center left',bbox_to_anchor=(1.05,.5),fontsize=8)
    plt.tight_layout()
    return sv(fig,'E13_来源分布.png')

def e14():
    sf(); ld(); fig, ax=plt.subplots(figsize=(10,7))
    cc=next((c for c in ['ts_code','stock_code','code'] if c in _m.columns),None)
    if not cc: return False
    vc=_m[cc].value_counts().head(20); y=range(len(vc))[::-1]
    nm=[CN.get(c.split('.')[0] if '.' in str(c) else str(c),str(c)[:6]) for c in vc.index]
    ax.barh(y,vc.values,height=.65,color=plt.cm.Blues(np.linspace(.3,.85,20))[::-1],edgecolor='w')
    ax.set_yticks(y); ax.set_yticklabels(nm,fontsize=9); ax.set_xlabel('事件数')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return sv(fig,'E14_公司频次Top20.png')

def e15():
    sf(); ld()
    if not os.path.exists(WC): return False
    from wordcloud import WordCloud
    cc='ts_code' if 'ts_code' in _m.columns else None
    if not cc: return False
    fm_map=_m[cc].value_counts().head(50).to_dict()
    freq={
        CN.get(str(k).split('.')[0],str(k).split('.')[0]): int(v)
        for k,v in fm_map.items()
    }
    wc=WordCloud(font_path=WC,width=1200,height=800,background_color='white',max_words=80,colormap='Greens').generate_from_frequencies(freq)
    fig,ax=plt.subplots(figsize=(11,7)); ax.imshow(wc,interpolation='bilinear'); ax.axis('off')
    plt.tight_layout()
    return sv(fig,'E15_公司词云.png')

def e16():
    sf(); ld(); fig, ax=plt.subplots(figsize=(13,6))
    df=_m.copy(); df['_d']=pdt(df['event_date']); df=df[df['_d'].notna()]
    ct=pd.crosstab(df['_d'].dt.month,df['event_type'])
    im=ax.imshow(ct.values,cmap='YlOrRd',aspect='auto')
    ax.set_xticks(range(len(ct.columns))); ax.set_xticklabels(ct.columns,rotation=45,ha='right',fontsize=9)
    ax.set_yticks(range(len(ct.index))); ax.set_yticklabels([f'{m}月' for m in ct.index],fontsize=9)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v=int(ct.values[i,j]); ax.text(j,i,v,ha='center',va='center',fontsize=9,color='white' if v>ct.values.max()*.6 else 'k')
    plt.colorbar(im,ax=ax,shrink=.85); plt.tight_layout()
    return sv(fig,'E16_时间类型交叉.png')

def e17():
    sf(); ld(); fig, ax=plt.subplots(figsize=(8,7))
    cc=next((c for c in ['CAR','CAR_new'] if c in _m.columns),None)
    ac=next((c for c in ['AR_0','AR_pre'] if c in _m.columns),None)
    if cc and ac:
        cv=pd.to_numeric(_m[cc],errors='coerce').sample(n=min(2000,len(_m)),random_state=42).values
        av=pd.to_numeric(_m[ac],errors='coerce').sample(n=min(2000,len(_m)),random_state=42).values
        mk=np.isfinite(cv)&np.isfinite(av)
        ax.scatter(cv[mk],av[mk],alpha=.3,s=8,color='#2166AC'); r=np.corrcoef(cv[mk],av[mk])[0,1]
        ax.set_xlabel('CAR(%)'); ax.set_ylabel('AR(%)')
        ax.text(.97,.97,f'r={r:.3f}',transform=ax.transAxes,fontsize=9,ha='right',bbox=dict(fc='w',alpha=.92))
    plt.tight_layout()
    return sv(fig,'E17_CAR_AR散点图.png')

def e18():
    sf(); ld(); fig, ax=plt.subplots(figsize=(10,6))
    ic='event_sub_type' if 'event_sub_type' in _m.columns else 'event_type'
    vc=_m[ic].value_counts().head(15); y=range(len(vc))[::-1]
    ax.barh(y,vc.values,height=.6,color=plt.cm.Greens(np.linspace(.3,.85,len(vc)))[::-1],edgecolor='w')
    ax.set_yticks(y); ax.set_yticklabels(vc.index,fontsize=9.5); ax.set_xlabel('事件数')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return sv(fig,'E18_行业分布.png')

def e19():
    sf(); ld(); fig, ax=plt.subplots(figsize=(12,6))
    df=_m.copy(); df['_d']=pdt(df['event_date']); df=df[df['_d'].notna()]
    dw=['一','二','三','四','五','六','日']; ct=pd.crosstab(df['_d'].dt.dayofweek,df['_d'].dt.month)
    im=ax.imshow(ct.values,cmap='YlOrRd',aspect='auto')
    ax.set_xticks(range(len(ct.columns))); ax.set_xticklabels([f'{m}月' for m in ct.columns],fontsize=10)
    ax.set_yticks(range(len(ct.index))); ax.set_yticklabels(['周'+dw[d] for d in ct.index],fontsize=10)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v=int(ct.values[i,j]); ax.text(j,i,v,ha='center',va='center',fontsize=9,color='white' if v>ct.values.max()*.6 else 'k')
    plt.colorbar(im,ax=ax,shrink=.85); plt.tight_layout()
    return sv(fig,'E19_发布时间热力图.png')

def e20():
    sf()
    if not os.path.exists(WC): return False
    from wordcloud import WordCloud
    fig, axes=plt.subplots(1,3,figsize=(18,6))
    panels=[('多','大涨 牛市 涨停 反弹 突破 利好 买入 龙头 资金流入 业绩增长 并购重组 政策扶持 板块轮动','Reds'),
             ('空','大跌 熊市 跌停 下杀 利空 卖出 逃顶 套牢 资金流出 业绩暴雷 减持 风险 破位 回调 震荡下行','Greys'),
             ('中','震荡 盘整 观望 消息面 技术面 基本面 走势 成交量 换手率 市盈率 净资产 收盘 开盘','Wistia')]
    for ax,(ti,tx,cm) in zip(axes,panels):
        wc=WordCloud(font_path=WC,width=900,height=650,background_color='white',max_words=80,colormap=cm).generate(tx)
        ax.imshow(wc,interpolation='bilinear'); ax.axis('off'); ax.set_title(ti,fontsize=12,fontweight='bold')
    fig.suptitle('股吧情感词云',fontsize=14,fontweight='bold',y=1.02); plt.tight_layout()
    return sv(fig,'E20_股吧词云.png')

def e21():
    sf(); ld(); fig, ax=plt.subplots(figsize=(9,6))
    if 'title' in _m.columns and 'CAR' in _m.columns:
        ln=_m['title'].str.len(); cr=pd.to_numeric(_m['CAR'],errors='coerce')
        ax.scatter(ln,cr,alpha=.25,s=10,color='#2980B9'); r=np.corrcoef(ln.fillna(0),cr.fillna(0))[0,1]
        ax.set_xlabel('字符数'); ax.set_ylabel('CAR(%)')
        ax.text(.97,.97,f'r={r:.4f}',transform=ax.transAxes,fontsize=9,ha='right',bbox=dict(fc='w',alpha=.92))
    plt.tight_layout()
    return sv(fig,'E21_标题长度vsCAR.png')

def e22():
    sf(); ld(); fig, axes=plt.subplots(2,3,figsize=(16,10))
    mr=_m.isnull().sum()/len(_m)*100; tmr=mr[mr>0].sort_values().tail(15)
    axes[0][0].barh(range(len(tmr)),tmr.values,color='#E74C3C',alpha=.7)
    axes[0][0].set_yticks(range(len(tmr))); axes[0][0].set_yticklabels(tmr.index,fontsize=7)
    axes[0][0].set_title(f'缺失率Top15({_m.shape[1]}列)',fontsize=10)
    nc=_m.select_dtypes(include=[np.number]).columns.tolist()[:10]
    _m[nc].describe().T[['mean','std','50%']].plot(kind='bar',ax=axes[0][1],width=.8,alpha=.75)
    axes[0][1].set_title('数值统计Top10',fontsize=10); axes[0][1].legend(fontsize=7)
    szs={'主':len(_m),'训':len(_tr),'验':len(_va),'测':len(_te)}
    axes[0][2].bar(szs.keys(),szs.values(),color=['#3498DB','#2ECC71','#F39C12','#E74C3C'],alpha=.8)
    axes[0][2].set_title('各集规模',fontsize=10)
    for i,v in enumerate(szs.values()): axes[0][2].text(i,v+.02*max(szs.values()),str(v),ha='center',fontweight='bold')
    dtypes=_m.dtypes.value_counts()
    axes[1][0].pie(dtypes.values,labels=[str(x).split('.')[-1][:6] for x in dtypes.index],autopct=lambda p:f'{p:.1f}%',textprops={'fontsize':9})
    axes[1][0].set_title(f'字段类型({_m.shape[1]}列)',fontsize=10)
    dup=_m.duplicated().sum(); un=len(_m)-dup
    axes[1][1].bar(['唯一','重复'],[un,dup],color=['#27AE60','#C0392B'],alpha=.8)
    axes[1][1].set_title(f'重复{dup/len(_m)*100:.1f}%',fontsize=10)
    axes[1][1].text(0,un+.02*max(un,dup),str(un),ha='center',fontweight='bold')
    axes[1][1].text(1,dup+.02*max(un,dup),str(dup),ha='center',fontweight='bold')
    mem=_m.memory_usage(deep=True).sum()/1024/1024; cmem=_m.memory_usage(deep=True).sort_values(ascending=False).head(8)/1024
    axes[1][2].barh(range(len(cmem)),cmem.values,color='#9B59B6',alpha=.7)
    axes[1][2].set_yticks(range(len(cmem))); axes[1][2].set_yticklabels([str(c)[:10] for c in cmem.index],fontsize=7)
    axes[1][2].set_title(f'内存{mem:.1f}MB Top8',fontsize=10)
    fig.suptitle('数据质量概览',fontsize=13,fontweight='bold',y=1.02); plt.tight_layout()
    return sv(fig,'E22_数据质量概览.png')

def e23():
    sf(); ld(); fig, ax=plt.subplots(figsize=(12,6))
    cc=next((c for c in ['CAR','CAR_new'] if c in _m.columns),None)
    if not cc: return False
    sub_col='event_sub' if 'event_sub' in _m.columns else 'event_type'
    sub={k:_m.loc[_m[sub_col]==k,cc].dropna().values for k in _m[sub_col].unique() if (_m[sub_col]==k).sum()>5}
    top10=dict(sorted(sub.items(),key=lambda kv:abs(np.median(kv[1])) if len(kv[1])>0 else 0,reverse=True)[:10])
    bd=[v for v in top10.values() if len(v)>0]; bl=list(top10.keys())[:len(bd)]
    if bd:
        bp=ax.boxplot(bd,tick_labels=bl,patch_artist=True,widths=.6); clrs=plt.cm.Set3(np.linspace(0,1,len(bd)))
        for pc,c in zip(bp['boxes'],clrs): pc.set_facecolor(c); pc.set_alpha(.78)
        ax.axhline(0,color='#666',ls='--',linewidth=.8); ax.set_ylabel('CAR(%)')
    plt.xticks(rotation=35,ha='right',fontsize=8); plt.tight_layout()
    return sv(fig,'E23_子类CAR箱线图.png')

def e24():
    sf(); ld(); fig, ax=plt.subplots(figsize=(12,6))
    cc=next((c for c in ['CAR','CAR_new'] if c in _m.columns),None)
    if not cc: return False
    df=_m.copy(); df['_d']=pdt(df['event_date']); df=df[df['_d'].notna()].sort_values('_d')
    cv=pd.to_numeric(df[cc],errors='coerce'); rl=cv.rolling(window=min(50,len(df))).mean()
    ax.plot(range(len(rl)),rl.values,color='#2E86AB',linewidth=1.5,alpha=.82)
    ax.fill_between(range(len(rl)),rl.values,0,where=rl.values>=0,color='#27AE60',alpha=.2,interpolate=True)
    ax.fill_between(range(len(rl)),rl.values,0,where=rl.values<0,color='#E74C3C',alpha=.2,interpolate=True)
    ax.axhline(0,color='#333',ls='-',linewidth=.8); ax.set_xlabel('序号'); ax.set_ylabel('CAR MA(50)')
    ax.text(.97,.97,f'Mean={cv.mean():.3f}',transform=ax.transAxes,fontsize=8.5,ha='right',family='monospace',bbox=dict(fc='w',alpha=.9))
    plt.tight_layout()
    return sv(fig,'E24_滚动CAR趋势.png')

def e25():
    sf(); ld(); fig, ax=plt.subplots(figsize=(14,5))
    df=_m.copy(); df['_d']=pdt(df['event_date']); df=df[df['_d'].notna()]
    daily=df.groupby(df['_d'].dt.date).size()
    ax.plot(range(len(daily)),daily.values,color='#3498DB',linewidth=1,alpha=.7)
    ma7=daily.rolling(7,min_periods=1).mean()
    ax.plot(range(len(ma7)),ma7.values,color='#E74C3C',linewidth=1.8,label='MA7')
    ax.legend(fontsize=9); ax.set_xlabel('日期序号'); ax.set_ylabel('日事件数')
    ax.text(.97,.97,f'Days={len(daily)} Avg={daily.mean():.1f}',transform=ax.transAxes,fontsize=8.5,ha='right',bbox=dict(fc='w',alpha=.9))
    plt.tight_layout()
    return sv(fig,'E25_事件频率时序.png')

def e26():
    sf(); ld(); fig, axes=plt.subplots(2,2,figsize=(16,10))
    pairs=[('CAR','RSI14'),('CAR','volatility_20d'),('ret_1d','ret_5d'),('dev_MA5','dev_MA20')]
    for ax,(fx,fy) in zip(axes.flat,pairs):
        if fx in _m.columns and fy in _m.columns:
            x=pd.to_numeric(_m[fx],errors='coerce').sample(n=min(1500,len(_m)),random_state=42).values
            y=pd.to_numeric(_m[fy],errors='coerce').sample(n=min(1500,len(_m)),random_state=42).values
            mk=np.isfinite(x)&np.isfinite(y)
            ax.scatter(x[mk],y[mk],alpha=.25,s=6,color='#2166AC'); r=np.corrcoef(x[mk],y[mk])[0,1] if mk.sum()>5 else 0
            ax.set_xlabel(fx[:10],fontsize=9); ax.set_ylabel(fy[:10],fontsize=9)
            ax.text(.95,.95,f'r={r:.3f}',transform=ax.transAxes,fontsize=8,ha='right',bbox=dict(fc='w',alpha=.85))
    fig.suptitle('变量联合分布',fontsize=12,fontweight='bold'); plt.tight_layout()
    return sv(fig,'E26_联合散点阵.png')

def e27():
    sf(); ld(); fig, ax=plt.subplots(figsize=(10,5.5))
    raw=17266; stages=['原始\n17266','去AI\n114','治理\n741','低价值\n134','常规\n185','最终\n14431']
    vals=[raw,raw-114,raw-114-741,raw-114-741-134,raw-114-741-134-185,14431]
    bars=ax.bar(stages,vals,color=plt.cm.RdYlGn(np.linspace(.15,.85,len(stages))),edgecolor='white',linewidth=.8)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+100,str(v),ha='center',fontsize=8,fontweight='bold')
    ax.set_ylabel('样本数'); ax.set_title('清洗流程',fontsize=11,fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return sv(fig,'E27_清洗流程.png')






def bt_f1():
    sf()
    tp = _find_data(['交易明细_3个月'])
    wp = _find_data(['周度收益_3个月'])
    if not tp or not wp:
        print('  SKIP bt_f1'); return False
    df_t = pd.read_csv(tp)
    df_w = pd.read_csv(wp)
    weeks = df_w['周'].values
    rets = [float(str(r).strip('%')) for r in df_w['周收益'].values]
    caps = [float(c) for c in df_w['累计资金'].values]
    fig, ax1 = plt.subplots(figsize=(13, 6))
    colors = ['#E74C3C' if r >= 0 else '#27AE60' for r in rets]
    ax1.bar(range(len(rets)), rets, color=colors, edgecolor='white', width=0.65, alpha=0.85)
    ax1.set_xticks(range(len(weeks))); ax1.set_xticklabels(['W%d'%w for w in weeks], fontsize=10)
    ax1.set_ylabel('周度收益率 (%)', fontsize=12); ax1.axhline(0, color='#666', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax2 = ax1.twinx()
    ax2.plot(range(len(caps)), caps, color='#2980B9', marker='o', markersize=5, linewidth=2)
    ax2.set_ylabel('累计资金（元）', fontsize=12, color='#2980B9'); ax2.tick_params(axis='y', labelcolor='#2980B9')
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor='#E74C3C', label='正收益'), Patch(facecolor='#27AE60', label='负收益'),
        plt.Line2D([0],[0], color='#2980B9', marker='o', label='累计资金')
    ], loc='upper left', fontsize=9)
    plt.tight_layout()
    return sv(fig, 'F1_周度收益率与累计权益曲线.png', 't3')


def bt_f2():
    sf()
    tp = _find_data(['交易明细_3个月'])
    if not tp: print('  SKIP bt_f2'); return False
    df = pd.read_csv(tp)
    fig, ax = plt.subplots(figsize=(14, 7))
    vals, colors, codes = [], [], []
    for _, r in df.iterrows():
        ret = float(r['扣费收益'].strip('%'))
        vals.append(ret); colors.append('#E74C3C' if ret >= 0 else '#27AE60')
        codes.append(str(r['股票代码']).split('.')[0])
    bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor='white', width=0.75, alpha=0.85)
    for b, v, cd in zip(bars, vals, codes):
        off = 0.45 if v >= 0 else -0.65; va = 'bottom' if v >= 0 else 'top'; cl = '#C0392B' if v >= 0 else '#1E8449'
        ax.text(b.get_x()+b.get_width()/2, v+off, '%s\n%+.2f%%'%(cd,v), ha='center', va=va, fontsize=7, fontweight='bold', color=cl)
    ax.axhline(0, color='#666', linewidth=0.8); ax.set_ylabel('个股收益率 (%)', fontsize=12)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    return sv(fig, 'F2_每周个股收益率明细.png', 't3')


def bt_f3():
    sf()
    tp = _find_data(['交易明细_3个月'])
    if not tp: print('  SKIP bt_f3'); return False
    df = pd.read_csv(tp)
    from collections import defaultdict
    ev_rets = defaultdict(list)
    for _, r in df.iterrows():
        ev_rets[str(r['事件子类'])].append(float(str(r['收益率']).strip('%')))
    cats = list(ev_rets.keys())
    wr = [sum(1 for v in ev_rets[c] if v>0)/len(ev_rets[c]) for c in cats]
    cnt = [len(ev_rets[c]) for c in cats]
    si = sorted(range(len(cats)), key=lambda i: cnt[i], reverse=True)
    cats = [cats[i] for i in si]; wr = [wr[i] for i in si]; cnt = [cnt[i] for i in si]
    fig, ax = plt.subplots(figsize=(11, 6))
    clrs = ['#3498DB' if w>=0.5 else '#E74C3C' for w in wr]
    bars = ax.barh(cats, [w*100 for w in wr], color=clrs, edgecolor='white', height=0.55, alpha=0.85)
    for b, wv, n in zip(bars, wr, cnt):
        ax.text(wv*100+1.5, b.get_y()+b.get_height()/2, '%d%% (%d笔)'%(round(wv*100), n), va='center', ha='left', fontsize=10, fontweight='bold')
    all_r = [v for vs in ev_rets.values() for v in vs]
    owr = sum(1 for r in all_r if r>0)/len(all_r)
    ax.axvline(owr*100, color='#8E44AD', ls='--', linewidth=2, label='整体胜率 %.1f%%'%(owr*100))
    ax.set_xlim(0, 105); ax.set_xlabel('胜率 (%)', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(axis='x', alpha=0.15)
    plt.tight_layout()
    return sv(fig, 'F3_事件子类胜率统计.png', 't3')


def bt_f4():
    sf()
    tp = _find_data(['交易明细_3个月'])
    wp = _find_data(['周度收益_3个月'])
    if not tp or not wp: print('  SKIP bt_f4'); return False
    df_t = pd.read_csv(tp); df_w = pd.read_csv(wp)
    cp, cl, pcnt, lcnt, ratios, wkkeys = 0, 0, [], [], [], []
    tw = sorted(df_w[df_w['买入数']>0]['周'].values)
    for w in tw:
        wt = df_t[df_t['周']==w]
        cp += sum(1 for _, r in wt.iterrows() if float(str(r['扣费收益']).strip('%')) > 0)
        cl += sum(1 for _, r in wt.iterrows() if float(str(r['扣费收益']).strip('%')) <= 0)
        pcnt.append(cp); lcnt.append(cl)
        ratios.append(cp/cl if cl>0 else cp)
        wkkeys.append('W%d'%int(w))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = range(len(wkkeys)); wd = 0.35
    b1 = ax1.bar([p-wd/2 for p in x], pcnt, wd, color='#E74C3C', label='累计盈利笔数', edgecolor='white', alpha=0.85)
    b2 = ax1.bar([p+wd/2 for p in x], lcnt, wd, color='#27AE60', label='累计亏损笔数', edgecolor='white', alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(wkkeys, fontsize=10); ax1.set_ylabel('累计交易笔数', fontsize=12)
    ax1.spines['top'].set_visible(False)
    ax2 = ax1.twinx()
    ax2.plot(x, ratios, color='#8E44AD', marker='D', markersize=6, linewidth=2)
    ax2.set_ylabel('盈亏比', fontsize=11, color='#8E44AD'); ax2.tick_params(axis='y', labelcolor='#8E44AD')
    if ratios: ax2.set_ylim(0, max(ratios)*1.25)
    fp = ratios[-1] if ratios else 0
    ax2.annotate('最终: %.2f'%fp, xy=(len(ratios)-1, fp), xytext=(max(0,len(ratios)-3), fp+0.5),
                 fontsize=11, fontweight='bold', color='#8E44AD',
                 arrowprops=dict(arrowstyle='->', color='#8E44AD', lw=1.5))
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, loc='upper left', fontsize=9)
    plt.tight_layout()
    return sv(fig, 'F4_盈亏比分析.png', 't3')


def bt_f5():
    sf()
    wp = _find_data(['周度收益_3个月'])
    if not wp: print('  SKIP bt_f5'); return False
    df_w = pd.read_csv(wp)
    mm = {43:'10月',44:'10月',45:'10月',46:'11月',47:'11月',48:'11月',
          49:'12月',50:'12月',51:'12月',52:'12月',1:'1月'}
    mo_order = ['10月','11月','12月','1月']
    ws_all = sorted(set(int(r['周']) for _, r in df_w.iterrows()))
    mat = np.full((len(mo_order), len(ws_all)), np.nan)
    for _, r in df_w.iterrows():
        wi = int(r['周']); mi = mm.get(wi)
        if mi in mo_order:
            mat[mo_order.index(mi), ws_all.index(wi)] = float(str(r['周收益']).strip('%'))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('rg',['#27AE60','#FFFFFF','#E74C3C'], N=256)
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, aspect='auto', vmin=-4, vmax=8)
    for mi_idx, mo in enumerate(mo_order):
        for wi_idx, wo in enumerate(ws_all):
            v = mat[mi_idx, wi_idx]
            if not np.isnan(v):
                ax.text(wi_idx, mi_idx, '%+.2f%%'%v, ha='center', va='center', fontsize=9, fontweight='bold',
                       color='white' if abs(v)>4 else '#2C3E50')
    ax.set_xticks(range(len(ws_all))); ax.set_xticklabels(['W%d'%w for w in ws_all], fontsize=10)
    ax.set_yticks(range(len(mo_order))); ax.set_yticklabels(mo_order, fontsize=11)
    ax.set_xlabel('周', fontsize=12); ax.set_ylabel('月份', fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label('收益率 (%)', fontsize=10)
    plt.tight_layout()
    return sv(fig, 'F5_月度周度热力图.png', 't3')


ABLATION_MODELS = [
    ('E2', '纯技术因子', '#3498DB'),
    ('E3', '+事件 one-hot', '#2ECC71'),
    ('E4', '+交互/排名/先验', '#E74C3C'),
]


def load_ablation_metrics():
    path = _find_data(['消融实验结果'])
    if not path:
        public_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'docs', 'results', 'ablation_metrics.csv',
        )
        if os.path.exists(public_path):
            path = public_path
    if not path:
        raise FileNotFoundError('未找到消融实验结果.csv')

    df = pd.read_csv(path, encoding='utf-8-sig')
    required = {'实验', '分类F1', '回归Rank_IC'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'消融实验结果缺少字段: {", ".join(sorted(missing))}')

    experiment = df['实验'].astype(str)
    rows = []
    for code, label, color in ABLATION_MODELS:
        match = df.loc[experiment.str.startswith(code)]
        if len(match) != 1:
            raise ValueError(f'消融实验结果中 {code} 记录数应为 1，实际为 {len(match)}')
        row = match.iloc[0]
        rows.append({
            'code': code,
            'label': label,
            'color': color,
            'f1': pd.to_numeric(row['分类F1'], errors='coerce'),
            'rank_ic': pd.to_numeric(row['回归Rank_IC'], errors='coerce'),
        })
    return pd.DataFrame(rows)


def abl_f1():
    sf(); rows = load_ablation_metrics()
    if rows['f1'].isna().any():
        raise ValueError('E2/E3/E4 的 F1 指标不完整')
    fig, ax = plt.subplots(figsize=(9, 6))
    names = [f"{row.code}\n{row.label}" for row in rows.itertuples()]
    vals = rows['f1'].tolist()
    bars = ax.bar(names, vals, color=rows['color'].tolist(), edgecolor='white', width=0.62, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.006, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold', color='#333')
    ax.set_ylabel('Weighted F1', fontsize=12); ax.set_ylim(0, max(vals) * 1.18)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return sv(fig, 'E30_消融_F1对比.png', 't3')


def abl_ic():
    sf(); rows = load_ablation_metrics()
    if rows['rank_ic'].isna().any():
        raise ValueError('E2/E3/E4 的 Rank IC 指标不完整')
    fig, ax = plt.subplots(figsize=(9, 6))
    names = [f"{row.code}\n{row.label}" for row in rows.itertuples()]
    vals = rows['rank_ic'].tolist()
    bars = ax.bar(names, vals, color=rows['color'].tolist(), edgecolor='white', width=0.62, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.004, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold', color='#333')
    ax.set_ylabel('Rank IC', fontsize=12); ax.set_ylim(0, max(vals) * 1.18)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return sv(fig, 'E31_消融_RankIC对比.png', 't3')


def t4_pie():
    sf()
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    labels = ['顺丰控股\n002352', '立讯精密\n002475', '中国人寿\n601628']
    sizes = [33.45, 33.33, 33.22]
    colors = ['#E74C3C', '#3498DB', '#27AE60']
    explode = (0.03, 0.01, 0.01)
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90, pctdistance=0.55, labeldistance=1.12,
        wedgeprops={'edgecolor':'white','linewidth':1.5}, textprops={'fontsize':13})
    for t in autotexts: t.set_fontsize(14); t.set_fontweight('bold'); t.set_color('white')
    ax.set_title('第1周实测持仓权重分配\n(2026-04-21 买入)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return sv(fig, 'F_T1_实测持仓权重分配.png', 't3')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--eda', action='store_true')
    parser.add_argument('--t3', action='store_true')
    parser.add_argument('--t4', action='store_true')
    parser.add_argument('--abl', action='store_true')
    args = parser.parse_args()

    EDA_LIST = [('e'+str(i).zfill(2), globals()['e'+str(i).zfill(2)]) for i in range(1,28)]
    BT_LIST  = [('bt_f'+str(i), globals()['bt_f'+str(i)]) for i in range(1,6)]
    ABL_LIST = [('abl_f1',abl_f1),('abl_ic',abl_ic)]
    T4_LIST  = [('t4_pie',t4_pie)]

    run_all = args.all or not any([args.eda,args.t3,args.t4,args.abl])
    failures = []

    if run_all or args.eda:
        for nm,fn in EDA_LIST:
            try:
                if fn() is False:
                    failures.append(nm)
            except Exception as e:
                failures.append(nm)
                print(f'  FAIL {nm}: {e}')

    if run_all or args.t3:
        for nm,fn in BT_LIST:
            try:
                if fn() is False:
                    failures.append(nm)
            except Exception as e:
                failures.append(nm)
                print(f'  FAIL {nm}: {e}')

    if run_all or args.t4:
        for nm,fn in T4_LIST:
            try:
                if fn() is False:
                    failures.append(nm)
            except Exception as e:
                failures.append(nm)
                print(f'  FAIL {nm}: {e}')

    if run_all or args.abl:
        for nm,fn in ABL_LIST:
            try:
                if fn() is False:
                    failures.append(nm)
            except Exception as e:
                failures.append(nm)
                print(f'  FAIL {nm}: {e}')

    print('\nDone.')
    if failures:
        print(f'Failed: {", ".join(failures)}')
        sys.exit(1)
