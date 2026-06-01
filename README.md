# Parkinson's Disease Progression Prediction

기계학습 2026-1 Machine Learning Term Project 소스코드 <br>
프로젝트에서는 파킨슨병 환자의 telemonitoring voice measurement를 이용해 질병 진행 정도를 나타내는 total_UPDRS 점수를 예측


## Dataset

이 프로젝트는 UCI Machine Learning Repository의 Parkinsons Telemonitoring 데이터셋을 사용

- UCI page: https://archive.ics.uci.edu/dataset/189/parkinsons%2Btelemonitoring
- DOI: https://doi.org/10.24432/C5ZS3N
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Recordings: 5,875
- Subjects: 42
- Main target: total_UPDRS
- Secondary target: motor_UPDRS
- Missing values: 0

데이터는 42명의 초기 파킨슨병 환자에게서 약 6개월 동안 수집한 5,875개 음성 기록이다. 각 행은 하나의 음성 recording이고, subject#는 환자 ID이다.

주요 target은 다음 두 개이다.

- total_UPDRS: 전체 UPDRS score로 main target
- motor_UPDRS: motor UPDRS score로 보조 target

주의할 점은 같은 환자의 recording이 여러 번 반복되어 있다는 것이다. 그래서 본 프로젝트에서는 subject#를 feature로 사용하지 않고 train/validation/test에 같은 subject가 섞이지 않도록 group split을 사용하였다.


## Repository Structure

| 경로 | 설명 |
|---|---|
| datasets/ | Parkinsons telemonitoring CSV |
| src/data.py | 데이터 로드, feature/target 구성, subject group split 저장/로드 |
| src/preprocess.py | EDA 표/그림 생성, train/validation/test split 생성 |
| src/modeling.py | baseline, regression model, GroupKFold tuning, error summary |
| src/evaluate.py | MAE, RMSE, R2, Pearson, Spearman 계산 |
| src/plots.py | EDA, 모델 비교, 예측/잔차, 오류 분석, feature importance 그림 |
| src/experiments.py | 모델 비교, 튜닝, final test, ablation, leakage diagnostic 실행 |
| results/ | 실행 후 생성되는 결과물 |

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run

아래의 두 파일을 실행

```bash
python3 src/preprocess.py
python3 src/experiments.py
```


## Models and Metrics

- Mean Baseline
- Linear Regression
- kNN Regressor
- Decision Tree
- Random Forest
- AdaBoost

주요 평가지표:
- MAE: UPDRS 점수 단위의 평균 절대 오차
- RMSE: 큰 오차를 더 크게 반영
- R2: 평균 예측 대비 설명력
- Pearson/Spearman correlation: 실제 점수와 예측 점수의 순위/선형 관계

## Output

| 파일 | 설명 |
|---|---|
| results/tables/eda/dataset_summary.csv | 데이터 크기, subject 수, split 요약 |
| results/tables/eda/split_summary.csv | subject group split 요약 |
| results/tables/modeling/model_comparison_total_updrs.csv | validation set 모델 비교 |
| results/tables/modeling/best_params_total_updrs.csv | GridSearchCV 튜닝 결과 |
| results/tables/modeling/test_baseline_comparison_total_updrs.csv | baseline과 최종 모델 test 비교 |
| results/tables/analysis/feature_set_ablation.csv | voice_only와 voice_plus_context 비교 |
| results/tables/analysis/group_split_vs_random_split.csv | subject group split과 random row split 비교 |
| results/tables/analysis/error_by_subject_total_updrs.csv | subject별 error 분석 |
| results/tables/analysis/error_by_severity_bin_total_updrs.csv | severity 구간별 error 분석 |
| results/tables/analysis/error_by_test_time_bin_total_updrs.csv | test_time 구간별 error 분석 |
| results/tables/analysis/permutation_importance_total_updrs.csv | permutation feature importance |
| results/figures/eda/target_distribution_total_updrs.png | target 분포 |
| results/figures/eda/selected_voice_features_distribution.png | 주요 voice feature 분포 |
| results/figures/eda/feature_correlation_heatmap.png | feature correlation heatmap |
| results/figures/eda/recordings_per_subject.png | subject별 recording 수 |
| results/figures/modeling/model_comparison_total_updrs.png | validation MAE 모델 비교 |
| results/figures/modeling/predicted_vs_actual_total_updrs.png | 예측값 vs 실제값 |
| results/figures/modeling/residual_plot_total_updrs.png | residual plot |
| results/figures/analysis/feature_importance_total_updrs.png | feature importance 그림 |
| results/figures/analysis/error_by_subject_total_updrs.png | subject별 MAE 그림 |


## Reproducibility

Random seed: 42로 고정
