# -*- coding: utf-8 -*-
"""
BC카드 공모전 · 핵심 수치 재현 스크립트
==========================================================================
목적  제출 자료(PPT)에 실린 핵심 수치를 원본 CSV에서 처음부터 다시 계산해 확인한다.
      (노트북에 없는 계산 — H2 비교·강건성, AMT 효과, H3 전국 재검증 — 도 여기서 재현)

입력  ABP_CONTEST_DATA.csv  (BC카드 제공, 2026.1~6월, 시군구×업종×성별×연령 집계)
      · 외국인 방문자수(한국관광 데이터랩 TourAPI, touDivCd=3)는 5단계 수집 결과의
        월 합계를 아래 VISITORS에 옮겨 두었다. → tour_datalab_foreign_visitors_monthly.csv 와 대조 가능
      · API 키는 필요 없다.

실행  python BC카드_수치재현.py  [CSV 경로]        (pandas · numpy · scipy)
      Colab에서는 CSV 경로를 인자 대신 아래 CSV 변수에 직접 적어도 된다.

출력  항목별로  ✓/✗  재계산값  (기대값)  을 출력한다. ✗가 있으면 마지막 줄에 개수를 알려준다.
==========================================================================
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu

CSV = next((a for a in sys.argv[1:] if a.lower().endswith('.csv')), 'ABP_CONTEST_DATA.csv')

MIN_CUM_CNT, MIN_LATEST_CNT = 200, 30      # 1단계 노트북과 같은 최소 거래건수 필터

# ------------------------------------------------------------------ 후보·비교군 정의
CANDS = [('경상북도', '청송군'), ('경상북도', '봉화군'), ('강원특별자치도', '고성군'), ('강원특별자치도', '영월군'),
         ('경상북도', '영양군'), ('경상북도', '의성군'), ('강원특별자치도', '양양군'), ('인천광역시', '옹진군')]
BORYEONG = ('충청남도', '보령시')                         # 대조군 (이미 알려진 곳)
TOURIST = [('서울특별시', '중구'), ('서울특별시', '종로구'), ('서울특별시', '강남구'), ('부산광역시', '해운대구'),
           ('인천광역시', '중구'), ('경상북도', '경주시'), ('강원특별자치도', '속초시'), ('강원특별자치도', '강릉시'),
           ('제주특별자치도', '제주시'), ('제주특별자치도', '서귀포시')]
INDUS = [('경기도', '안산시 단원구'), ('경기도', '시흥시'), ('경기도', '화성시 만세구'), ('경기도', '평택시'),
         ('광주광역시', '광산구'), ('충청남도', '아산시'), ('경기도', '김포시'), ('경상남도', '김해시'),
         ('경상북도', '경산시'), ('대구광역시', '달서구')]
# 육안 검수에서 아티팩트로 본 셀 (앞 2개: 1단계 노트북의 EXCLUDED_ARTIFACTS / 뒤 2개: 본 자료가 추가로 뺀 셀)
ARTIFACTS_NOTEBOOK = [('전북특별자치도', '무주군', '스넥'), ('전라남도', '곡성군', '서양음식')]
ARTIFACTS_EXTRA = [('경상북도', '봉화군', '제과점'), ('강원특별자치도', '양양군', '일식회집')]
# 2단계 EDA의 H3 홀드아웃 항목 20개
GOOD_ITEMS = [
    ('경상북도', '청송군', '편의점'), ('경상북도', '청송군', '서양음식'), ('경상북도', '청송군', '슈퍼마켓'),
    ('경상북도', '봉화군', '편의점'), ('경상북도', '봉화군', '슈퍼마켓'), ('경상북도', '봉화군', '서양음식'), ('경상북도', '봉화군', '일반한식'),
    ('강원특별자치도', '고성군', '서양음식'), ('강원특별자치도', '고성군', '편의점'), ('강원특별자치도', '고성군', '슈퍼마켓'),
    ('강원특별자치도', '영월군', '편의점'), ('강원특별자치도', '영월군', '일반한식'), ('강원특별자치도', '영월군', '슈퍼마켓'),
    ('경상북도', '영양군', '슈퍼마켓'), ('경상북도', '영양군', '편의점'),
    ('경상북도', '의성군', '서양음식'), ('경상북도', '의성군', '슈퍼마켓'),
    ('강원특별자치도', '양양군', '서양음식'), ('강원특별자치도', '양양군', '편의점'), ('강원특별자치도', '양양군', '일반한식'),
]
# 외국인 방문자수 (명, 한국관광 데이터랩 touDivCd=3, 2026.1~6월)
VISITORS = {
    '청송군': [2991, 2800, 2537, 12514, 21601, 18218], '봉화군': [1080, 1085, 3097, 12111, 24476, 22962],
    '고성군': [5032, 5921, 10444, 18282, 17684, 14979], '영월군': [2752, 2943, 2764, 6228, 8447, 7697],
    '영양군': [1101, 1109, 1905, 7530, 9270, 9980], '의성군': [2833, 2832, 4309, 9164, 19008, 17829],
    '양양군': [10641, 8710, 10742, 20766, 20009, 18510], '옹진군': [166906, 176009, 229536, 377775, 332677, 277484],
}

# ------------------------------------------------------------------ 출력 도우미
N_FAIL = 0


def head(t):
    print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)


def check(name, got, want, tol=0.0, fmt='{:,.2f}'):
    """재계산값(got)이 기대값(want)과 tol 이내로 같으면 ✓"""
    global N_FAIL
    ok = abs(got - want) <= tol
    N_FAIL += 0 if ok else 1
    print(f"  {'✓' if ok else '✗'} {name:<44} {fmt.format(got):>16}   (기대 {fmt.format(want)})")


# ------------------------------------------------------------------ 0. 로드 · 전처리
head('0. 데이터 로드')
cols = ['STRD_YYMM', 'SIDO_NM', 'CCG_NM', 'GENDER_CD', 'AGE_CD', 'TP_BUZ_NO', 'TP_BUZ_NM']
df = pd.read_csv(CSV, encoding='utf-8', dtype={c: str for c in cols}).rename(columns={'amt': 'AMT', 'cnt': 'CNT'})
df['TP_BUZ_NM_CLEAN'] = df['TP_BUZ_NM'].str.replace(' ', '', regex=False)        # 업종명 공백 정규화
foreign = df[df.GENDER_CD == '3']                                                 # 외국인 세그먼트
months = sorted(foreign.STRD_YYMM.unique())
idx = ['SIDO_NM', 'CCG_NM', 'TP_BUZ_NM_CLEAN']
pv_cnt = foreign.pivot_table(index=idx, columns='STRD_YYMM', values='CNT', aggfunc='sum', fill_value=0)[months]
pv_amt = foreign.pivot_table(index=idx, columns='STRD_YYMM', values='AMT', aggfunc='sum', fill_value=0)[months]
nat_cnt, nat_amt = pv_cnt.groupby(level='TP_BUZ_NM_CLEAN').sum(), pv_amt.groupby(level='TP_BUZ_NM_CLEAN').sum()

check('전체 행 수', len(df), 242_574, fmt='{:,.0f}')
check('시도+시군구 키 (동명 시군구 분리)', len(df[['SIDO_NM', 'CCG_NM']].drop_duplicates()), 255, fmt='{:,.0f}')
check('시군구명만 세면 (동명 병합 시)', df.CCG_NM.nunique(), 233, fmt='{:,.0f}')
check('외국인 이용 건수 합계', foreign.CNT.sum(), 75_509_726, fmt='{:,.0f}')
check('외국인 이용 금액 합계 (억 원)', foreign.AMT.sum() / 1e8, 12_780.90, tol=0.01)

# ------------------------------------------------------------------ 1. 최소건수 필터 · 신호 점수
head('1. 최소건수 필터 · 상대 초과성장률 점수 (3단계 신호 탐지)')
ok_mask = (pv_cnt.sum(axis=1) >= MIN_CUM_CNT) & (pv_cnt[months[-1]] >= MIN_LATEST_CNT)
pv_rel = pv_cnt[ok_mask]
check('지역×업종 조합 (필터 전)', len(pv_cnt), 2_199, fmt='{:,.0f}')
check('필터 통과 조합', ok_mask.sum(), 2_111, fmt='{:,.0f}')


def relative_growth_score(pivot, national, mons, n_recent=2):
    """지역 MoM 증가율 − 전국 동종업종 MoM 증가율 을 최근 n_recent개월 평균 (1단계 노트북과 동일)"""
    growth, nat_g = pivot.pct_change(axis=1), national.pct_change(axis=1)
    rel = growth.copy()
    for buz in national.index:
        m = rel.index.get_level_values('TP_BUZ_NM_CLEAN') == buz
        rel.loc[m] = growth.loc[m].values - nat_g.loc[buz].values
    return rel[mons[-n_recent:]].mean(axis=1)


scores = relative_growth_score(pv_rel, nat_cnt, months)          # 5~6월 평균 초과성장률 (건수 기준)
region_total = pv_cnt.groupby(level=['SIDO_NM', 'CCG_NM']).sum().sum(axis=1)


def region_frame(drop=()):
    s = scores.drop(index=[k for k in drop if k in scores.index]).replace([np.inf, -np.inf], np.nan).dropna()
    d = s.reset_index(); d.columns = ['SIDO_NM', 'CCG_NM', 'ind', 'v']
    return pd.DataFrame({'base': region_total, 'g': d.groupby(['SIDO_NM', 'CCG_NM'])['v'].mean()}).dropna()


# ------------------------------------------------------------------ 2. H2 사각지대
head('2. H2 사각지대 — 후보 8곳 vs 유명 관광지 · 산업단지형')


def grp(pdf, keys):
    sub = pdf.loc[[k for k in keys if k in pdf.index]]
    return sub.base.mean(), sub.g.mean() * 100


pdf = region_frame()
b8, g8 = grp(pdf, CANDS); bt, gt = grp(pdf, TOURIST); bi, gi = grp(pdf, INDUS); b9, g9 = grp(pdf, CANDS + [BORYEONG])
check('후보 8곳 평균 baseline (건)', b8, 24_145, fmt='{:,.0f}')
check('후보 8곳 평균 초과성장률 (%p)', g8, 21.03, tol=0.01)
check('유명 관광지 10곳 평균 초과성장률 (%p)', gt, 0.92, tol=0.01)
check('산업단지형 10곳 평균 초과성장률 (%p)', gi, 0.67, tol=0.01)
check('후보 ÷ 관광지 (배)', g8 / gt, 22.8, tol=0.1, fmt='{:.1f}')
check('후보 ÷ 산업단지형 (배)', g8 / gi, 31.4, tol=0.1, fmt='{:.1f}')
check('관광지 baseline ÷ 후보 baseline (배)', bt / b8, 47.8, tol=0.1, fmt='{:.1f}')
check('(보령 포함 9곳) baseline (건)', b9, 34_133, tol=0.5, fmt='{:,.0f}')
check('(보령 포함 9곳) 초과성장률 (%p)', g9, 19.74, tol=0.01)
pdf2 = region_frame(ARTIFACTS_NOTEBOOK + ARTIFACTS_EXTRA)          # 강건성: 아티팩트 셀 4개 제외
others = [i for i in pdf2.index if i not in CANDS + [BORYEONG] + TOURIST + INDUS]
_, g8b = grp(pdf2, CANDS); _, gtb = grp(pdf2, TOURIST); _, gib = grp(pdf2, INDUS)
check('강건성: 아티팩트 4셀 제외 후 후보 (%p)', g8b, 17.45, tol=0.01)
check('강건성: 후보 ÷ 관광지 (배)', g8b / gtb, 18.9, tol=0.1, fmt='{:.1f}')
check('강건성: 후보 ÷ 산업단지형 (배)', g8b / gib, 26.1, tol=0.1, fmt='{:.1f}')
p = mannwhitneyu(pdf2.loc[CANDS, 'g'], pdf2.loc[TOURIST, 'g'], alternative='greater').pvalue
check('강건성: Mann-Whitney p (후보 > 관광지)', p * 1e5, 2.29, tol=0.01)   # ×1e-5

# 크기 일치 대조군 — 관광지·산업단지형은 후보보다 규모가 수십 배 크므로, 후보(baseline 최대 67,590건)와 비슷한
# 규모(7만 건 미만)의 '비후보' 지역과도 비교한다. 후보는 신호 상위로 골랐으므로 이 격차에도 선정 효과가 섞여 있다.
SIZE_CAP = 70_000
def small_controls(pdf_):
    return [i for i in pdf_.index if i not in CANDS + [BORYEONG] + TOURIST + INDUS and pdf_.loc[i, 'base'] < SIZE_CAP]
sm_raw, sm_cln = small_controls(pdf), small_controls(pdf2)
check('크기 일치 대조군 지역 수 (7만 건 미만 비후보)', len(sm_cln), 63, fmt='{:.0f}')
check('크기 일치 대조군 평균 baseline (건)', pdf2.loc[sm_cln, 'base'].mean(), 34_630, tol=0.5, fmt='{:,.0f}')
check('크기 일치 대조군 평균 초과성장률 (%p, 원 산출)', pdf.loc[sm_raw, 'g'].mean() * 100, 4.76, tol=0.01)
check('크기 일치 대조군 평균 초과성장률 (%p, 4셀 제외)', pdf2.loc[sm_cln, 'g'].mean() * 100, 4.58, tol=0.01)
check('후보 ÷ 크기 일치 대조군 (배, 원 산출)', g8 / (pdf.loc[sm_raw, 'g'].mean() * 100), 4.4, tol=0.1, fmt='{:.1f}')
check('후보 ÷ 크기 일치 대조군 (배, 4셀 제외)', g8b / (pdf2.loc[sm_cln, 'g'].mean() * 100), 3.8, tol=0.1, fmt='{:.1f}')
p2 = mannwhitneyu(pdf2.loc[CANDS, 'g'], pdf2.loc[sm_cln, 'g'], alternative='greater').pvalue
check('크기 일치: Mann-Whitney p (후보 > 대조군)', p2 * 1e5, 1.54, tol=0.01)   # ×1e-5
check('크기 일치 대조군 중 +15%p 이상인 지역 수', int((pdf2.loc[sm_cln, 'g'] >= 0.15).sum()), 5, fmt='{:.0f}')

# ------------------------------------------------------------------ 3. AMT 기대효과
head('3. 후보 8곳 외국인 소비 금액(AMT) 성장')
cand_amt = sum(pv_amt.loc[k].sum(axis=0).values.astype(float) for k in CANDS)
nat_tot_amt = nat_amt.sum(axis=0)
check('후보 8곳 1월 AMT (억 원)', cand_amt[0] / 1e8, 4.14, tol=0.01)
check('후보 8곳 6월 AMT (억 원)', cand_amt[-1] / 1e8, 6.55, tol=0.01)
check('후보 8곳 1→6월 성장률 (%)', (cand_amt[-1] / cand_amt[0] - 1) * 100, 58.23, tol=0.01)
check('전국 외국인 1→6월 성장률 (%)', (nat_tot_amt.iloc[-1] / nat_tot_amt.iloc[0] - 1) * 100, 3.28, tol=0.01)
check('후보 8곳 6개월 합계 (억 원)', cand_amt.sum() / 1e8, 30.35, tol=0.01)
check('전국 외국인 AMT 대비 비중 (%)', cand_amt.sum() / foreign.AMT.sum() * 100, 0.2375, tol=0.0005, fmt='{:.4f}')

# ------------------------------------------------------------------ 4. H1 시차상관
head('4. H1 — 카드 초과성장률 ↔ TourAPI 외국인 방문자수 시차상관 (n=32, 8곳×4개월)')
nat_mom = (nat_cnt.sum(axis=0).pct_change().dropna() * 100).values


def mom(v):
    v = np.asarray(v, float); return (v[1:] / v[:-1] - 1) * 100


def card_excess(pivot):                          # 지역 카드 건수 MoM − 전국 MoM (2~6월)
    return {k[1]: mom(pivot.xs(k, level=('SIDO_NM', 'CCG_NM')).sum(axis=0).values) - nat_mom for k in CANDS}


vis_mom = {k: mom(v) for k, v in VISITORS.items()}


def lag_stat(cx, demean=False):
    xs, ys = [], []
    for k in cx:
        c, v = np.array(cx[k]), vis_mom[k]
        if demean: c, v = c - c.mean(), v - v.mean()
        xs += list(c[:-1]); ys += list(v[1:])      # 카드[m] ↔ 방문자[m+1]
    return spearmanr(xs, ys), pearsonr(xs, ys), len(xs)


cx_B = card_excess(pv_rel)                       # 필터 통과 조합만 합산 (본 자료 · pivot_cnt_reliable)
cx_A = card_excess(pv_cnt)                       # 필터 미적용 (시차상관 노트북이 그대로 쓰는 pivot_cnt)
(spB, peB, n) = lag_stat(cx_B)
(spA, peA, _) = lag_stat(cx_A)
(spD, _, _) = lag_stat(cx_B, demean=True)
check('표본 수 n', n, 32, fmt='{:.0f}')
check('Spearman ρ  [필터 통과 합산 · 본 자료]', spB[0], -0.591, tol=0.001, fmt='{:.3f}')
check('  p-value', spB[1], 0.0004, tol=0.00005, fmt='{:.4f}')
check('Pearson r', peB[0], -0.434, tol=0.001, fmt='{:.3f}')
check('지역 평균 제거 후 Spearman ρ', spD[0], -0.474, tol=0.001, fmt='{:.3f}')
check('  p-value', spD[1], 0.0061, tol=0.00005, fmt='{:.4f}')
check('Spearman ρ  [필터 미적용 — 노트북 그대로]', spA[0], -0.555, tol=0.001, fmt='{:.3f}')
check('  p-value', spA[1], 0.0010, tol=0.00005, fmt='{:.4f}')
first = {'방문자 먼저': 0, '동시': 0, '카드 먼저': 0}
for k in cx_B:                                    # 정점월: 카드=초과성장률 최대월, 방문자=전월 대비 증가율 최대월
    cm, vm = int(np.argmax(cx_B[k])), int(np.argmax(vis_mom[k]))
    first['카드 먼저' if cm < vm else '방문자 먼저' if vm < cm else '동시'] += 1
check('정점: 방문자가 먼저인 지역 수', first['방문자 먼저'], 6, fmt='{:.0f}')
check('정점: 같은 달인 지역 수 (영월군)', first['동시'], 1, fmt='{:.0f}')
check('정점: 카드가 먼저인 지역 수 (옹진군)', first['카드 먼저'], 1, fmt='{:.0f}')
print('  ※ ρ가 음수인 이유: 방문자 증가 정점이 카드 정점보다 먼저 와서 서로 반대로 움직이는 것처럼 보이기 때문이다.')

# ------------------------------------------------------------------ 5. H3 지속성 (사후 확인 · 전국 재검증)
head('5. H3 — 후보 20개 항목의 지속성(사후 확인)과 전국 재검증')


def rel_score_months(pivot, national, mons, n_recent=2):
    """월별 상대 초과성장률을 마지막 n_recent개월 평균 (2단계 EDA의 relative_growth_score와 동일한 값)"""
    buz = pivot.index.get_level_values('TP_BUZ_NM_CLEAN')
    rows = []
    for m0, m1 in list(zip(mons[:-1], mons[1:]))[-n_recent:]:
        reg_g = pivot[m1] / pivot[m0].replace(0, np.nan) - 1
        nat_g = national[m1] / national[m0].replace(0, np.nan) - 1
        rows.append(reg_g.values - nat_g.reindex(buz).values)
    return pd.Series(np.nanmean(np.vstack(rows), axis=0), index=pivot.index)


train = rel_score_months(pv_rel, nat_cnt, months[:4])                          # 1~4월 점수
nat_g46 = nat_cnt[months[5]] / nat_cnt[months[3]] - 1
buz_all = pv_rel.index.get_level_values('TP_BUZ_NM_CLEAN')
actual = (pv_rel[months[5]] / pv_rel[months[3]].replace(0, np.nan) - 1) - nat_g46.reindex(buz_all).values   # 4→6월 실측
persisted = sum(actual.loc[k] > 0 for k in GOOD_ITEMS)
neg_train = sum(train.loc[k] < 0 for k in GOOD_ITEMS)
check('후보 20개 중 4→6월 초과성장이 양(+)인 항목', persisted, 20, fmt='{:.0f}')
check('그중 1~4월 점수가 음수였던 항목', neg_train, 4, fmt='{:.0f}')
print('  ※ 이 20개는 6개월 전체 신호로 골랐기 때문에 “사후 확인”이다 (순환 선택).')

dfh = pd.DataFrame({'train': train, 'actual': actual}).replace([np.inf, -np.inf], np.nan).dropna()
check('전체 2,111개 조합 중 실측 양(+) 비율 (%)', (dfh.actual > 0).mean() * 100, 54.4, tol=0.1, fmt='{:.1f}')
sel = dfh[dfh.train >= 0.10]
check('1~4월 점수 ≥ +10%p 조합 수', len(sel), 275, fmt='{:.0f}')
check('  그중 실측 양(+) 비율 (%)', (sel.actual > 0).mean() * 100, 36.7, tol=0.1, fmt='{:.1f}')
check('점수 ↔ 실측 Spearman ρ (조합 단위)', spearmanr(dfh.train, dfh.actual)[0], -0.275, tol=0.001, fmt='{:.3f}')
d = dfh.reset_index()
reg = d.groupby(['SIDO_NM', 'CCG_NM']).agg(train=('train', 'mean'), actual=('actual', 'mean'))
check('점수 ↔ 실측 Spearman ρ (시군구 단위)', spearmanr(reg.train, reg.actual)[0], -0.363, tol=0.001, fmt='{:.3f}')
rank = reg.train.rank(ascending=False)
want_rank = {'청송군': 250, '양양군': 248, '봉화군': 58, '영월군': 30, '고성군': 25, '의성군': 44, '영양군': 9, '옹진군': 6}
for k in CANDS:
    check(f'후보 {k[1]}의 1~4월 점수 순위 (255곳 중)', rank.loc[k], want_rank[k[1]], fmt='{:.0f}')

# 수준 지속 — 지역×업종의 전국 동종업종 대비 비중을 1~2월 평균으로 정규화 (모든 달 건수>0인 조합)
X = pv_rel.values.astype(float) / nat_cnt.reindex(buz_all).values.astype(float)
X = X[(X > 0).all(axis=1)]


def level_stat(X, thr):
    b = X[:, :2].mean(1); ltr = X[:, 2:4].mean(1) / b; lte = X[:, 4:6].mean(1) / b
    s = ltr >= thr
    return s.sum(), ((lte[s] > 1).mean() if s.sum() else np.nan), (lte > 1).mean()


rng = np.random.default_rng(0)
res = {}
for thr in [1.2, 1.3, 1.5]:                         # 각 셀의 6개월 값 순서를 무작위로 섞은 300회 시뮬레이션
    n_obs, p_obs, base = level_stat(X, thr)
    nulls = [level_stat(np.array([rng.permutation(r) for r in X]), thr)[1] for _ in range(300)]
    res[thr] = (n_obs, p_obs, np.nanmean(nulls), np.nanpercentile(nulls, 95))
n_obs, p_obs, null_mean, null_p95 = res[1.3]
check('상대 수준 ≥1.3배(3~4월) 조합 수', n_obs, 106, fmt='{:.0f}')
check('  그중 5~6월 수준 >1 유지 비율 (%)', p_obs * 100, 100.0, tol=0.1, fmt='{:.1f}')
check('  무작위 기준선 평균 (%)', null_mean * 100, 83.5, tol=0.5, fmt='{:.1f}')
check('  무작위 기준선 95번째 백분위 (%)', null_p95 * 100, 91.5, tol=0.5, fmt='{:.1f}')

# ------------------------------------------------------------------ 요약
head('요약')
print('  모든 항목이 기대값과 일치합니다.' if N_FAIL == 0 else f'  ✗ 항목이 {N_FAIL}개 있습니다 — 위 출력에서 확인하세요.')
