# re:vision — Doerig et al. 2025 복제/일반화 계획

## 0. 한 줄
Doerig et al. (Nat Mach Intell 2025)를 **LAION-fMRI**로 복제하고, COCO 밖 이미지 분포로 **일반화 검증**한다.

---

## 1. 논문 요약

**High-level visual representations in the human brain are aligned with large language models**
Doerig, Kietzmann, Allen, Wu, Naselaris, Kay, Charest.
*Nature Machine Intelligence* 7:1220–1234 (2025). doi:10.1038/s42256-025-01072-0

### 핵심 주장
뇌 고차 시각영역의 표상 = **장면 캡션의 LLM 문장 임베딩**으로 근사 가능. LLM은 시각 입력을 본 적 없음.

### 세팅
- 데이터: **NSD**. 7T fMRI, 피험자 8명, 각 9~10k COCO 이미지, 총 73k, 이미지당 3회 반복. 1.8mm, `betas_fithrf_GLMdenoise_RR`.
- 자극 표상: COCO 이미지당 사람 캡션 5개 → **MPNet** (`all-mpnet-base-v2`, 768-d) → 5개 평균.
- 비교 방법 3종:
  - **RSA** (모델 RDM ↔ 뇌 RDM, searchlight + ROI, noise-ceiling 보정)
  - **인코딩 모델** (LLM 임베딩 → 복셀, fractional ridge regression)
  - **디코딩 모델** (복셀 → LLM 임베딩, + 3.1M Google Conceptual Captions 사전 최근접 탐색)

### 결과 4개

**R1 — LLM 임베딩이 뇌 활동 예측**
ventral / lateral / parietal 스트림 전역. 인코딩 성능이 피험자 간 일치도(노이즈 실링)에 근접.
텍스트만으로 알려진 선택성 재현: 얼굴·신체(FFA/OFA/EBA) vs 장소(PPA/OPA) vs 음식.

**R2 — 뇌에서 캡션 복원**
복셀 → MPNet 임베딩 선형 디코딩 → 310만 캡션 사전 최근접 → 본 장면의 정확한 문장 생성.

**R3 — 성능 근원 = 단어가 아니라 *통합***
- 카테고리 라벨 LLM 임베딩 > multi-hot / fasttext / GloVe
- 전체 캡션 > 명사만 / 동사만
- 전체 캡션 > 개별 단어 임베딩 평균
- 단, 어순 섞은 문장도 원문과 상관 0.91 → MPNet은 어순 둔감. "통합" = 문맥적 관계지 통사구조 아님.

**R4 — LLM-학습 RCNN이 기존 모델 다 이김**
vNet 기반 순환 CNN을 `이미지 → 캡션 LLM 임베딩` 회귀로 학습 (COCO 4.8만 장, NSD 이미지 제외).
CLIP / ResNet101_32x8d(수억 장), ImageNet(100만+) 대비 **자릿수 단위로 적은 데이터**.
그럼에도 13개 SOTA 뇌예측 모델(CORnet-S, CLIP, SimCLR, Places365, Taskonomy 등) 대비 ventral·parietal 유의 우세.
비대칭 증거: LLM-학습 RCNN에서 카테고리 라벨 **디코딩 가능** / 카테고리-학습 RCNN에서 LLM 임베딩 **디코딩 불가** → LLM 목적함수가 카테고리를 **포함하고 더 넓음**.

### 저자 인정 한계 (= 복제/일반화가 노릴 지점)
- NSD 과제 = 연속 재인 과제. 피험자가 속으로 캡션 생성했을 가능성 배제 불가 → **과제 의존적 효과일 수 있음**.
- LLM 임베딩의 *어떤 축*이 뇌와 매칭되는지 해석 불가.
- 자극이 전부 COCO → **분포 일반화 미검증**.

---

## 2. LAION-fMRI 적합성

| Doerig 요구사항 | LAION-fMRI 보유 | 판정 |
|---|---|---|
| 7T fMRI, 고차 시각 ROI | 7T, BIDS, retinotopy derivatives | OK |
| 단일시행 베타 | GLMsingle `(n_trials × n_voxels)` | OK |
| 자연 장면 이미지 | 25,052장 @ 1000×1000 | OK |
| 이미지당 사람 캡션 | human + AI 캡션 | OK |
| 객체 카테고리 라벨 (R3 통제모델용) | 자동 객체 세그멘테이션 | OK |
| COCO 자극 (직접 복제용) | MSCOCO 포함 | OK |
| **COCO 밖 분포** | LAION-natural, THINGS, OOD | **일반화 기회** |

결론: **직접 복제 + 일반화 둘 다 한 데이터셋에서 가능.** 상금 트랙 2개($2,500 + $2,500) 동시 노림 가능.

---

## 3. 복제 범위 결정

### 복제(Replication) — 필수
- **R1**: MSCOCO 하위집합만 사용. 캡션 → MPNet → RSA + 인코딩. ventral/lateral/parietal에서 유의한 정합 재현.
- **R3**: 통제 모델 사다리 (multi-hot / GloVe / fasttext / 명사만 / 동사만 / 단어평균 vs 전체 캡션). 계산 싸고 결론 강함.
- **R2**: 디코딩 + 사전 최근접. 중간 비용.
- **R4**: RCNN 학습. **비용 최고.** 시간 남으면.

### 일반화(Generalization) — 차별점
Doerig가 못 한 것:
1. **분포 이동**: LLM-캡션 정합이 LAION-natural / THINGS / OOD 이미지에서도 유지되나? THINGS는 물체 중심 (장면 문맥 희박) → "통합" 가설이 옳다면 **THINGS에서 정합이 떨어져야 함**. 강한 예측, 반증 가능.
2. **AI 캡션 vs 사람 캡션**: 임베딩 소스 교체해도 뇌 정합 유지되나? 유지된다면 사람 라벨링 병목 제거.
3. **과제 교란 검증**: LAION-fMRI 과제가 NSD 재인 과제와 다르면 → 저자가 인정한 한계를 **직접 반증/확증** 가능. ★ 가장 값어치 큰 각도.
4. **비전 임베딩 대조**: 데이터셋에 CLIP/DINOv2/PEcore/SigLIP2 임베딩 이미 포함 → LLM 캡션 임베딩 vs 순수 비전 임베딩 정면 비교, 추가 계산 0.

---

## 4. re:vision 일정 (사이트 기준)

| Phase | 내용 | 마감 |
|---|---|---|
| 1 | 복제할 발견 선택 | — |
| 2 | 등록 (정원 마감 주의) | **2026-09-15** |
| 3 | 제안서 제출 (템플릿) + 원저자 통보 | 등록 후 4주 |
| 4 | 보드 피드백 수령 | — |
| 5 | 데이터 다운로드 + 분석 | — |
| 6 | 보고서 초안 + GitHub 코드 공개 | **2027-02-15** |
| 7 | 보드 피드백 반영 | — |
| 8 | 최종본 제출 (저자권 + 상금 대상) | **2027-04-30** |

보상: 컨소시엄 논문 공저자 + 복제 $2,500 / 일반화 $2,500 / facilitator $1,000.

원저자 통보 대상: Ian Charest (`ian.charest@umontreal.ca`), Adrien Doerig.

---

## 5. 지금 할 일 (순서 고정)

### 완료됨 (2026-07-12)
- [x] venv 생성 `.venv/` (Python 3.12.9)
- [x] `laion-fmri` 0.1.0 설치 완료 (+ awscli, nibabel, h5py, pandas, numpy)
- [x] API 확인. 핵심 함수:
  ```python
  import laion_fmri as lf
  lf.set_data_dir('./laion_data')      # 데이터 디렉토리 (미리 mkdir 필요)
  lf.get_subjects()                    # S3의 피험자 BIDS ID
  lf.get_rois(subject=None, category=None)
  lf.download_captions()               # 캡션 CSV — CC0, DUA 불필요
  lf.download_embeddings()             # CLIP/DINOv2/PEcore/SigLIP2 사전계산 임베딩
  lf.download_segmentations()
  lf.request_stimulus_access()         # ★ 원본 이미지 DUA 신청 (별도)
  lf.load_subject() / lf.load_subjects()
  ```
- [x] 라이선스 구조 확인:
  - **CC0 (즉시 사용)**: fMRI 데이터, 캡션, 임베딩, 세그멘테이션, 자극 메타데이터
  - **DUA 필요**: 원본 자극 이미지(JPEG)만

### ★ 다음 액션 — 본인이 직접 해야 함 (2건)

**(a) 라이선스 동의.** `download_captions()`가 대화형 프롬프트에서 `I AGREE` 입력을 요구함. 법적 동의라 대리 수락 안 함. 터미널에서 직접:
```bash
.venv/bin/python -c "
import laion_fmri as lf
lf.set_data_dir('./laion_data')
lf.download_captions()"      # 프롬프트에 I AGREE 입력
```
동의하면 그 뒤부터 캡션·임베딩·fMRI 전부 비대화형으로 받아짐.

**(b) DUA 신청.** `lf.request_stimulus_access()` — 원본 이미지용. **승인 지연이 최대 리스크. 오늘 넣을 것.**
단, R1/R2/R3는 캡션+베타만 있으면 되므로 **이미지 없이도 착수 가능**. 이미지는 R4(RCNN 학습)에서만 필수.

### 그 다음
1. **캡션 CSV 감사** — 이미지당 사람 캡션 몇 개? (Doerig는 5개 평균. 1개뿐이면 캡션 간 평균화 불가 → 방법 조정 필요.) MSCOCO / THINGS / NSD-겹침 각각 몇 장? → **검정력 결정. 전부의 상류.**
2. **Phase 2 등록** — 정원제. 마감 기다리지 말 것. 제안서 중심 문장은 **과제 교란(§6)** — Doerig 저자가 명시적으로 요청한 후속 연구임을 앞세울 것.
3. 파이프라인 스켈레톤: `GLMsingle 베타 + 캡션 → MPNet 임베딩 → RDM → RSA`. R1 최소 재현.
4. 통제 모델 사다리 (R3) → 싸고 결정적.
5. 일반화 축: THINGS 대비 대조 + 과제 교란. 여기가 논문의 알맹이.
6. R4(RCNN)는 DUA 승인 후. 시간 남으면.

### 공짜 보너스
`download_embeddings()`에 CLIP/DINOv2/PEcore/SigLIP2 임베딩이 **이미 계산돼 있음**. → "LLM 캡션 임베딩 vs 순수 비전 임베딩" 정면 비교를 **추가 계산 0으로** 붙일 수 있음. 제안서에 넣을 것.

---

## 6. LAION-fMRI 확인된 사양 (laion-fmri.hebartlab.com, 2026-07 기준)

| 항목 | LAION-fMRI | Doerig/NSD | 함의 |
|---|---|---|---|
| 피험자 수 | **5명** (런치 릴리스) | 8명 | 검정력 ↓. 단 반복 ↑로 상쇄 |
| 스캔 과제 | **수동 관찰 (passive viewing)** | 연속 재인 과제 | ★ **최대 기회** — 아래 참조 |
| 세션 | 각 30회 메인 세션 | 30~40회 | 동급 |
| 공유 이미지 | 1,492장, **최대 12회 반복** | 515장, 3회 반복 | **노이즈 실링 훨씬 높음** |
| 자극 출처 | LAION-natural(120M 코퍼스), MSCOCO, THINGS, **NSD**, OOD, 추상도형·착시 | COCO 전용 | 분포 일반화 가능 |
| 총 자극 | 25,052장 @ 1000×1000 | 73,000장 | 규모 ↓ |
| 캡션 | human + AI | human ×5 | 개수 확인 필요 |

### ★ 결정적 발견 — 과제 교란(task confound)
Doerig 저자 원문 인정:
> "The task of the NSD participants was to report if they had previously seen each presented image... participants were internally captioning the scenes, and this may have benefitted the LLM caption embeddings... it will be interesting for future work to investigate LLM-based codes under different tasks."

**LAION-fMRI = 수동 관찰.** 내적 캡션 생성을 유도할 과제 요구가 없음.
→ **이 복제 자체가 저자가 요청한 후속 연구다.**
- LLM-캡션 정합이 **유지되면**: 과제 교란 배제, 원 주장 강화. 복제 성공 + 이론적 기여.
- **사라지면**: 원 결과가 과제 산물임을 입증. 반증. 더 큰 뉴스.
둘 다 발표 가치 있음. **실패할 수 없는 설계.**
이 한 문장이 제안서의 중심이어야 함.

### 주의 — NSD 자극 겹침
LAION-fMRI 자극에 **NSD 이미지가 포함**됨. 두 갈래:
- 활용: 동일 이미지 × 다른 과제 × 다른 피험자 → **과제 대비를 이미지 매칭 상태로** 수행 가능. 교란 통제 최강.
- 위험: "독립 복제"라는 주장이 약해질 수 있음. 제안서에서 **먼저 밝히고 설계 이점으로 전환**할 것.

## 7. 남은 미해결 질문

- 이미지당 사람 캡션 개수 (Doerig는 5개 평균. 1개면 캡션 간 평균화 불가 → 방법 조정 필요)
- MSCOCO / THINGS / NSD 하위집합 각각 몇 장인지 정확한 수치 (문서 페이지에 미기재 → 데이터 받아서 직접 집계)
- GLMsingle 베타의 ROI 정의 — NSD 'streams' ROI(ventral/lateral/parietal)에 대응하는 게 있나? 없으면 retinotopy derivative에서 직접 정의
- ~~suggested studies에 Doerig 있나~~ → **있음.** 태그: Brain-model alignment / Representational geometry / Decoding-Reconstruction. 공식 승인 = 제안서 반려 리스크 낮음. 단 경쟁자 가능 → 과제 교란 축이 차별점.
