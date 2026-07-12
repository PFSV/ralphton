# 문헌 조사: TFM 사전학습 prior를 위한 "LLM 에이전트 기반 실세계 모사 Tabular 데이터 생성기"

## TL;DR (핵심 요약)

- **당신의 정확한 교차점을 차지한 논문은 없습니다** — "LLM 에이전트가 의미론(semantics)·현실성(realism)을 반영한 tabular 생성기를 설계하고, 그 출력이 tabular foundation model(TFM)의 *사전학습 prior*가 되는" 연구는 아직 없어, 핵심 아이디어의 novelty는 방어 가능합니다. 다만 반드시 선제적으로 구분해야 할 "니어미스(nearest-miss)" 논문이 둘 있습니다: **Hod, Rosenblatt & Stoyanovich (arXiv:2504.14368)** — LLM 에이전트로 스키마 메타데이터에서 SCM을 구축하지만 용도가 DP(차등 프라이버시) 분류기 사전학습이고 TFM prior가 아님; **O'PRIOR (arXiv:2605.18971)** — TFM 사전학습용 현실성 지향 조합형 SCM prior를 설계하지만 LLM을 전혀 쓰지 않음.
- 가장 강한 포지셔닝은 **역할 (b): "LLM 에이전트 = 생성기 설계자/구성자"** (SCM/DAG 구조, 컬럼 의미론, 분포 파라미터를 에이전트가 결정하고, 실제 샘플링은 프로그램이 수행)입니다. 역할 (a: LLM이 직접 행 생성)의 치명적 반론들 — OpenML 데이터셋에 대한 LLM 암기/오염(Bordt et al.), 프라이버시 누출(Ward et al.), 사전학습 규모에서의 감당 불가능한 토큰 비용 — 을 구조적으로 회피하기 때문입니다.
- 리뷰어는 TabPFN v2, TabICL/TabICLv2, Mitra, 그리고 ("현실성 논쟁"용) TabForestPFN과의 비교를 요구할 것이고, **"현실성이 중요하지 않을 수 있다"**는 반론을 제기할 것입니다. TabForestPFN은 "fine-tuned TabForest가 WhyTrees에서 최고, fine-tuned TabPFN이 TabZilla에서 우세" — 즉 비현실적이지만 복잡한 데이터가 fine-tuning 후 이길 수 있음을 보였습니다. 논문은 Mitra의 "다양성/구별성(distinctiveness)/성능" prior 품질 프레임워크와 Real-TabPFN/TabDPT의 "real 데이터가 도움된다" 증거를 반드시 다뤄야 합니다.

## 핵심 발견

1. **필드의 초점이 아키텍처에서 prior 설계로 이동했습니다.** Mitra가 이를 명시적으로 선언합니다 — "합성 데이터셋을 생성하는 분포, 즉 데이터 prior의 설계에 더 큰 관심을 기울여야 한다" — 그리고 최초의 체계적 prior 품질 방법론(Generalizability Matrix G, Performance Vector P)을 제공합니다. 당신 논문이 진입하는 지적 공간이 바로 여기입니다.
2. **역할 (b)의 직접적 방법론 선행 연구가 존재합니다.** Hod et al.(2504.14368)은 이미 스키마 메타데이터만으로 SCM(변수 → DAG → 구조방정식 → Pyro 코드)을 도출하는 LLM 에이전트를 갖고 있습니다. novelty는 이 논문과의 대비로 조각해야 합니다: 그들은 타깃 스키마당 하나의 surrogate 데이터를 만들어 *DP 분류기* 사전학습에 쓰지만, 당신은 *TFM prior* 역할을 할 *데이터셋들의 분포*를 생성합니다.
3. **타깃(TFM prior) 쪽의 직접 선행 연구도 존재합니다.** O'PRIOR는 TFM 사전학습 전용으로 현실성 지향 조합형 SCM prior를 설계했지만 — 전적으로 프로그램 방식이고, LLM도 의미론도 없습니다. 당신의 "prior 설계" 쪽 경쟁자입니다.
4. **의미론 공백은 실재하며 문서화되어 있습니다.** ConTextTab은 컬럼 이름과 의미 있는 값들이 TabPFN/TabICL에서 사용되지 않음을 명시합니다. CARTE/TabSTAR/TARTE는 의미론을 *real* 데이터로 주입할 뿐, LLM이 설계한 *합성* prior로 주입한 적은 없습니다. 이 교차점은 비어 있습니다.
5. **"현실성이 중요한가?" 논쟁은 미결이며 핵심 쟁점입니다.** TabForestPFN은 비현실적이지만 복잡한 결정경계가 (fine-tuning 후) 도움이 된다고 주장하고, Real-TabPFN과 TabDPT는 real 데이터가 도움이 된다고 주장하며, Mitra는 충실도보다 다양성+구별성이 중요하다고 주장합니다. 논문은 여기서 입장을 취해야 합니다.

## 상세 내용

### 카테고리 1 — 직접 관련 / 가장 가까운 연구 (novelty에 결정적)

**Do You Really Need Public Data? Surrogate Public Data for Differential Privacy on Tabular Data** — Shlomi Hod, Lucas Rosenblatt, Julia Stoyanovich, 2025, arXiv:2504.14368.
다단계 LLM **에이전트**(GPT-4o, Claude 3.5 Sonnet, Llama 3.3 70B)가 스키마 수준 메타데이터만으로 구조적 인과 모델을 도출합니다: 변수 나열 → 제약 제안 → 외생/루트 노드 선택 → DAG 간선 제안(비순환성 검증) → 각 변수를 부모에 대한 구조방정식으로 매핑 → 분포 파라미터 할당 → 실행 가능한 Pyro 코드로 자동 컴파일해 데이터를 샘플링. "panel of experts" 앙상블로 여러 실행을 혼합합니다.
*관련성:* 당신의 역할 (b)에 가장 가까운 기존 구현체입니다. 다음처럼 명확히 구분해야 합니다 — 그들은 타깃 스키마당 하나의 surrogate를 만들어 DP FTTransformer를 DP-SGD fine-tuning 전에 사전학습시키는 용도이며, TFM prior도 아니고, 다수 데이터셋에 걸친 분포도 아니며, 범용 prior가 아니라 per-dataset 방식입니다. **related work에서 가장 먼저 선제 처리해야 할 단일 최중요 논문입니다.**

**LLM-Driven Performance-Space Augmentation for Meta-Learning-Based Algorithm Selection** — Darren Zhu, Daren Ler, 2026, arXiv:2605.09518.
두 landmarker 알고리즘의 교차검증 R²로 정의된 2차원 성능 공간의 타깃 영역을 겨냥하도록 LLM에 프롬프트를 줘 통째로 합성 회귀 데이터셋을 생성하고, 알고리즘 선택용 메타 데이터셋을 증강합니다. 42개 UCI 회귀 데이터셋 + 730개 합성 데이터셋으로 평가했고 uniform 증강이 subset accuracy를 크게 개선했습니다.
*관련성:* "LLM이 통째 합성 tabular 데이터셋을 생성해 meta-learning 코퍼스를 풍부하게 한다"는 점에서 겹치지만, 목표가 성능 공간 커버리지(실세계 의미론 모사가 아님)이고 소비자가 PFN이 아닌 meta-learner입니다. 구분이 쉽습니다.

**Incorporating LLM Priors into Tabular Learners** — Max Zhu et al., 2023, arXiv:2311.11628.
LLM의 사전 지식을 tabular 분류기에 주입(해석 가능·제어 가능)하며, TabPFN 스타일 모델로의 LLM prior 통합을 future work로 언급합니다.
*관련성:* "LLM prior + tabular"를 향한 초기 시도이지만, 생성형이 아니고 사전학습용도 아닙니다.

**경쟁자 판정:** arXiv/OpenReview/Semantic Scholar(2024–2026)를 조사한 결과, (A) LLM 에이전트로 의미론적/현실적 tabular 생성기를 설계하면서 동시에 (B) TFM/PFN 사전학습 prior를 타깃으로 하는 단일 논문은 발견되지 않았습니다. Hod et al.은 (A)만, O'PRIOR와 Mitra는 (B)만 차지합니다. 이 교차점이 당신의 방어 가능한 novelty입니다.

### 카테고리 2 — TFM용 합성 prior 설계 (비교 대상이 될 기존 전통)

- **TabPFN v1** — Hollmann, Müller, Eggensperger, Hutter, ICLR 2023. SCM prior에서 뽑은 수백만 합성 데이터셋으로 사전학습한 트랜스포머; 작은 tabular 분류에 대해 단일 forward pass의 Bayesian in-context 추론.
- **TabPFN v2** — Hollmann et al., Nature 637(8045):319–326, 2025 (doi:10.1038/s41586-024-08328-6). 회귀로 확장; 타깃 레짐은 소·중형 테이블(Nature 논문 기준 최대 샘플 1만·feature 500 — TabArena 51개 중 36개가 이 "소형 데이터셋" 레짐); 더 정교한 DAG 구성으로 prior 강화. **TabPFN-2.5**(arXiv:2511.08667)가 최신 버전이자 현재 SOTA 주장.
- **TabICL** — Qu, Holzmüller, Varoquaux, Le Morvan, ICML 2025, arXiv:2502.05564. 2단계 컬럼→행 attention 아키텍처로 대형 테이블로 ICL 확장(최대 6만 샘플 합성 데이터로 사전학습, 추론 시 50만 처리); prior에 트리 기반 구성요소 추가(아이디어는 Léo Grinsztajn). **TabICLv2** — arXiv:2602.11139 — 더 좋고 빠르고 확장 가능한 오픈 TFM으로, PFN 스타일 prior의 유용한 분류 체계(SCM, 트리, 혼합, 계층적)를 related work에 정리.
- **Mitra** — Zhang et al. (Amazon/AutoGluon), 2025, arXiv:2510.21204, NeurIPS 2025. SCM + 트리 기반(gradient boosting, random forest, decision tree) prior의 큐레이션된 혼합; prior 품질 기준 도입: real 데이터 성능, 다양성, 구별성(Generalizability Matrix G, Performance Vector P).
  *관련성:* 당신 논문이 딛고 설 평가 어휘와 동기를 정확히 제공합니다.
- **LimiX** — Zhang et al., 2025, arXiv:2509.03505. 3단계 DAG 구성(국소 인과 구조 → 전체 DAG → 샘플링 → task 적응)의 계층적 SCM prior, 난이도/solvability 제어 가능; 분류·회귀·결측 대치·feature/sample 선택·인과추론을 하나의 모델로. **LimiX-2M**(arXiv:2606.04485)은 low-rank collapse/attention 병목을 다룸.
- **TabForestPFN / den Breejen et al.** — "Fine-tuned In-Context Learning Transformers are Excellent Tabular Data Classifiers," 2024, arXiv:2405.13396 (OpenReview pE0UM18TQh). forest 생성기(과적합 결정트리를 데이터 생성 과정으로 사용)가 *비현실적이지만 복잡한 결정경계*의 데이터셋을 생성. 핵심 발견: "TabForest는 비현실적 사전학습 데이터 탓에 zero-shot 성능이 낮음에도 fine-tuning 시 일부 real 데이터셋에서 TabPFN을 능가"; "fine-tuned TabForest가 WhyTrees에서 최고, fine-tuned TabPFN이 TabZilla에서 우세." TabForestPFN은 두 생성기를 혼합.
  *관련성:* 반드시 다뤄야 할 "현실성은 (크게) 중요하지 않다"는 핵심 반대 증거입니다.
- **TabDPT** — Ma et al., 2024. in-context retrieval + self-supervised masked-column 모델링으로 *real* 데이터에 사전학습; 대규모 real 데이터 사전학습이 경쟁력 있음을 보이고 회귀로 확장.
- **Real-TabPFN** — Garg, …, Hollmann, Müller, Hutter, 2025, arXiv:2507.03971. OpenML/Kaggle에서 큐레이션한 71개 대형 real 테이블로 TabPFNv2를 계속 사전학습(continued pretraining); OpenML AutoML Benchmark 29개 데이터셋에서 상당한 개선을 달성했고, 이 큐레이션된 세트가 "CommonCrawl이나 GitTables 같은 더 넓지만 잠재적으로 노이즈가 많은 코퍼스보다 우수한 다운스트림 예측 정확도"를 냄을 보임.
- **O'PRIOR (Shaping the Prior: How Synthetic Task Distributions Determine Tabular Foundation Model Quality)** — Bouadi et al. (Lexsi Labs), 2026, arXiv:2605.18971. TFM 사전학습용 조합형 현실성 prior: 5개 메커니즘 계열(MLP, 트리, 1-D convolution, RBF GP, VAR)의 계층적 SCM, 현실성 엔진(이질적 주변분포, MCAR/MAR/MNAR 결측, 타깃 변환), 스트레스 모듈(교란, 허위 상관/shortcut, 공변량 이동), 누수 방지 커리큘럼; 아키텍처/옵티마이저/컴퓨트를 고정하고 prior만 변화.
  *관련성:* 당신의 "TFM용 현실적 prior 설계" 쪽 최근접 경쟁자 — 그러나 프로그램 방식이며 LLM도 의미론도 없음.
- **APT (Adversarially Pre-trained Transformer)** — 2025, arXiv:2502.04573. 적대적 합성 데이터 "에이전트들"이 모델을 어렵게 하도록 자신의 DGP를 이동시켜 TabPFN보다 다양한 prior를 만듦.
  *관련성:* 여기서 "agents"는 적대적 데이터 생성기를 뜻하며 LLM 에이전트가 **아님** — 용어상 혼동을 정리해둘 것.
- **관련 PFN-prior 분석들:** Drift-Resilient TabPFN(시간에 따라 SCM 파라미터를 변조하는 2단계 생성 prior); FairPFN(arXiv:2506.07049, 인과 공정성 prior); "Causal Pre-training Under the Fairness Lens"(arXiv:2601.17912); "Does TabPFN Understand Causal Structures?"(arXiv:2511.07236). 이들은 SCM prior가 *왜* 작동하는지를 탐구하며, "의미론적/현실적 prior가 도움이 될 수 있는 이유" 논증에 유용합니다.

### 카테고리 3 — LLM 기반 tabular 데이터 생성 (일반, TFM 사전학습 목적 아님)

- **GReaT** — Borisov et al., ICLR 2023, arXiv:2210.06280. 직렬화된 행("Col is Val…")에 autoregressive LLM(GPT-2/DistilGPT-2)을 fine-tuning, 임의 조건화를 위한 feature 순서 랜덤 셔플. LLM-tabular 생성의 기초 논문.
- **GReaTER** — 2025, arXiv:2503.15564. GReaT을 데이터 강화/축소로 확장해 행 인코딩 개선.
- **CLLM (Curated LLM)** — Seedat, Huynh, van Breugel, van der Schaar, ICML 2024, arXiv:2312.12112. 저데이터 레짐에서 LLM 프롬프팅으로 tabular 증강 후, learning-dynamics + 신뢰도/불확실성 지표로 생성된 행을 큐레이션.
  *관련성:* LLM tabular 데이터에 대한 생성기→큐레이터 루프를 직접 모델링 — 당신 파이프라인의 verifier 단계 템플릿.
- **Tabby** — 2025, arXiv:2503.02152. tabular/구조화 합성을 위한 LLM 아키텍처 수정; 컬럼별 loss 추적; 미학습 컬럼 값의 의미 구조 포착.
- **LLM-TabFlow / LLM-TabLogic** — 2025, arXiv:2503.02161. LLM이 직렬화된 컬럼 이름+설명에서 컬럼 간 논리/인과 그룹을 추론하고, latent diffusion이 의존성을 보존하며 데이터 생성. 관련: TAPTAP, HARMONIC, AIGT (메타데이터/스키마 프롬프트 생성).
- **TabuLa-8B / T4** — Gardner et al., 2024, arXiv:2406.12031. 약 300만 테이블(T4, TabLib 파생)에 대한 대규모 LLM fine-tuning으로 tabular ICL; 초저데이터 성능 우수.
- **서베이:** "LLMs on Tabular Data: Prediction, Generation, Understanding"(arXiv:2402.17944); "Language Modeling on Tabular Data"(arXiv:2408.10548); "A Comprehensive Survey of Synthetic Tabular Data Generation"(arXiv:2504.16506); "A Survey on Evaluating Quality and Trustworthiness in LLM-Generated Data"(arXiv:2601.17717).

### 카테고리 4 — 생성기/시뮬레이터의 설계자·구성자로서의 LLM 에이전트 (역할 (b) ideation용)

- **Eureka** — Ma et al., ICLR 2024, arXiv:2310.12931. GPT-4가 환경 소스 + task 설명으로부터 보상 함수 *코드*를 작성하고 진화적으로 정제: "10개 로봇 형태를 포함한 29개 오픈소스 RL 환경에서 83%의 task에서 인간 전문가를 능가, 평균 정규화 개선 52%."
  *관련성:* "LLM이 생성기/목적함수 코드를 작성 + in-context 진화적 정제"의 정범(canonical) 패턴 — LLM이 SCM config를 작성하고 critic 루프를 도는 구조로 직접 이식 가능.
- **Eurekaverse** — 2024, arXiv:2411.01775. LLM이 4족 보행 파쿠르를 위한 적응형 커리큘럼으로 *환경 프로그램*을 생성.
  *관련성:* LLM이 끊임없이 변하는 task 분포를 설계 — 합성 데이터셋 분포 설계와 유사 구조.
- **Hod et al. 2504.14368** (카테고리 1 참조) — LLM 에이전트가 Pyro SCM 코드를 작성. tabular/SCM 설정에서 LLM-as-generator-designer의 가장 직접적인 선례.
- **SD-SCM / Bynum & Cho 2024**, "Causal Inference with Hybrid LLM Synthetic Data" (arXiv:2511.00318) — 위상 순서가 주어지면 LLM이 구조방정식을 명세해 (반사실) 데이터를 시뮬레이션.
  *관련성:* LLM이 SCM 메커니즘을 저작할 수 있음을 확인 — 다만 인과추론용이지 TFM prior용은 아님.
- **MachineLearningLM** — 2025, arXiv:2509.06806. TabICL 방식의 SCM 기반 합성 tabular task 수백만 개로 LLM을 LoRA continued-pretraining해 LLM이 many-shot ICL을 수행.
  *관련성:* 여기서는 LLM이 프로그램적 SCM prior로 훈련되는 *학생* — 당신 설정의 정반대라 유용한 대조점.

### 카테고리 5 — Tabular 학습에서의 의미론과 실세계 지식

- **TabLLM** — Hegselmann et al., AISTATS 2023, arXiv:2210.10723. 행을 텍스트로 직렬화("The [column] is [value]"); 초소수샷 분류가 컬럼 이름/값의 의미론적 prior를 활용; 이득은 의미 있는 컬럼/값 텍스트에 의존(LLM이 본 적 없는 유전자 이름 등에서는 약함).
- **LIFT** — Dinh et al., NeurIPS 2022. tabular task용 GPT-3/GPT-J의 언어 인터페이스 fine-tuning; 고전적 방법과 대략 비슷.
- **CARTE** — Kim, Grinsztajn, Varoquaux, ICML 2024. 테이블 의미론을 모델링하는 다양한 소스 간 그래프-attentional 사전학습; 의미론적으로 풍부한 CARTE 벤치마크에서 SOTA(단, task별 fine-tuning 필요).
- **ConTextTab** — Spinaci et al., 2025, arXiv:2506.10707 (OpenReview MmKuX9ZvM3). 모달리티별 임베딩(텍스트/날짜/수치)과 컬럼 헤더 인코딩을 갖춘 의미론 인식 table-native ICL 학습기, real 데이터로 훈련; CARTE 벤치마크의 새 기준.
  *관련성:* 당신의 동기가 되는 공백을 그대로 진술 — 컬럼 이름과 의미 있는 값들이 TabPFN/TabICL에서 사용되지 않음.
- **TabSTAR** — 2025, arXiv:2505.18125 — 텍스트 필드를 가진 TFM; **TARTE / Table Foundation Models** — arXiv:2505.14415 — 문자열/컬럼 이름을 이용한 knowledge pre-training; **TabGemma** — arXiv:2511.03570 — LLM continued pretraining + retrieval 기반 텍스트 tabular ICL. 모두 의미론을 real 데이터 또는 추론 시점의 real 컬럼 이름으로 주입하며, 합성 prior로 주입하지 않음.

### 카테고리 6 — 평가 방법론

- **다운스트림 벤치마크:** Grinsztajn et al.(WhyTrees, 45개), NeurIPS 2022; TabZilla(McElfresh et al., 176개), NeurIPS 2023; TALENT(약 300개, Liu et al.); OpenML-CC18; TabRepo; **TabArena**(Erickson et al., living benchmark, arXiv:2506.16791) — tabular 연구에 쓰인 1053개 데이터셋을 조사해 51개를 수동 큐레이션, 16개 모델 계열(TFM 3종 포함)을 약 2,500만 개 학습 모델 인스턴스로 평가; TableShift(분포 이동); TabReD(시간적); TabArena의 소형 데이터셋 레짐(51개 중 36개, ≤1만 샘플)은 TabPFN Nature 타깃과 일치.
  *관련성:* TFM prior에 대해 기대되는 transfer 평가 프로토콜.
- **합성 데이터 충실도 지표:** α-precision / β-recall / authenticity (Alaa et al., ICML 2022); 주변분포(KS/TVD "Shape"), 결합(상관 "Trend"), Wasserstein 거리, Jensen-Shannon divergence, C2ST; TSTR vs TRTR 유틸리티. 서베이: "Systematic Assessment of Tabular Data Synthesis"(arXiv:2402.06806); ACM Computing Surveys 리뷰(10.1145/3704437); **TabStruct** — tabular 데이터의 구조적 충실도(arXiv:2509.11950) — 무작위 구성 SCM이 "실세계 tabular 데이터에 인코딩된 구조 정보를 효과적으로 반영한다"고 주장하므로 직접 관련.
- **prior 품질 → 다운스트림 연결:** Mitra의 G/P 방법론(arXiv:2510.21204)과 O'PRIOR의 통제된 prior-ablation 프로토콜(arXiv:2605.18971)이 "prior 품질이 TFM 성능을 좌우함"을 입증하는 두 가지 템플릿. ScoringBench(arXiv:2603.29928)는 TFM용 proper scoring rule 평가를 추가.
- **현실성 논쟁:** *현실성 반대 증거* — TabForestPFN(arXiv:2405.13396); *현실성/real 데이터 지지* — Real-TabPFN(arXiv:2507.03971), TabDPT, O'PRIOR; *절충(다양성 > 충실도)* — Mitra.

### 카테고리 7 — Ideation에 인접한 연구

- **Phi / "Textbooks Are All You Need"** — Gunasekar et al., 2023, arXiv:2306.11644; phi-1.5 arXiv:2309.05463. 교과서 품질 합성 데이터(GPT-3.5 생성)로 소형 모델이 스케일링 추세를 깨뜨림; 합성 데이터의 다양성 강조.
  *관련성:* "큐레이션/합성 데이터 품질 > 규모" 논증의 대표작으로, 당신의 prior 품질 명제와 직접 유사.
- **Demystifying Synthetic Data in LLM Pre-training** — 2025, arXiv:2510.01631 (EMNLP 2025). 순수 합성 데이터가 CommonCrawl보다 자동으로 우수하지는 *않음*을 밝힌 체계적 스케일링 법칙 연구.
  *관련성:* 냉정한 균형추이자, 합성 사전학습의 한계에 대해 리뷰어가 인용할 가능성이 높은 논문.
- **CLLM** (카테고리 3) — 생성기 + 원칙적 큐레이션은 tabular판 self-instruct/verifier 유사체.
- **Domain randomization / sim2real** 및 **PFN prior 오지정(misspecification) / prior-real 분포 이동** — prior의 *다양성*(충실도가 아니라)이 중요한 이유에 대한 이론적 프레임으로 개념적으로 원용; TabForestPFN과 Mitra가 그 프레임의 tabular 특화 증거를 제공.

### 암기/오염 문헌 (반론 대응에 결정적)

- **Elephants Never Forget: Memorization and Learning of Tabular Data in LLMs** — Bordt et al., COLM 2024, arXiv:2404.06209. GPT-3.5/GPT-4가 다수의 유명 tabular 데이터셋(Iris, Wine, Adult, Housing, OpenML Diabetes, Titanic 등)을 그대로 암기했음을 보임; 4가지 암기 검사와 interpretml/LLM-Tabular-Memorization-Checker 패키지 제공.
- **When Tables Leak: Attacking String Memorization in LLM-Based Tabular Data Generation** — Ward et al., 2025, arXiv:2512.08875. no-box 멤버십 추론 공격(LevAtt)이 fine-tuned 및 in-context LLM tabular 생성기 양쪽의 프라이버시 누출을 노출, 일부 경우 완벽한 멤버십 분류기로 작동.
- **Detection of LLM Contamination with Tabular Data** — Ronval et al., IDA 2025. GPT/Llama/Gemma/Phi 계열 전반의 오염 탐지.

## 권고 사항

**1단계 — 역할 (b) 중심으로 포지셔닝을 고정하고 두 니어미스를 선제 처리.**
기여를 이렇게 프레임하세요: *"LLM 에이전트가 프로그램 방식의 (SCM 기반) tabular 생성기를 구성한다 — DAG 구조, 컬럼 의미론, 분포 파라미터를 선택해 — TFM용 의미론·현실성 인식 사전학습 prior를 생산하기 위해."* related work에서 (i) **Hod et al. 2504.14368**(같은 LLM→SCM 코드 메커니즘이지만 per-schema DP 분류기 사전학습이며, 데이터셋 분포에 대한 TFM prior가 아님)과 (ii) **O'PRIOR 2605.18971** 및 **Mitra**(TFM용 현실적/혼합 prior이지만 LLM도 의미론도 없음)를 명시적으로 구분하세요. 방어 가능한 주장: *TFM의 합성 prior를 LLM 에이전트로 설계한 최초, 그리고 실세계 컬럼 의미론을 합성 prior에 심은 최초.*

**2단계 — 역할 (a) vs (b) 분담을 의도적으로 선택.**
헤드라인은 역할 (b)로 (저비용, 오염 내성, 제어 가능, 샘플링은 빠른 코드가 수행). 역할 (a) — LLM이 직접 행/스키마를 출력 — 는 소규모 ablation 또는 시드(seeding) 메커니즘으로만 포함하고, 암기/비용 문제를 정면으로 다루세요. 결정 변경 임계값: 역할 (a)의 생성 비용이 다운스트림 TFM 사전학습 컴퓨트에 근접하거나, 암기 검사(4단계)가 걸리면 → 역할 (a)를 스키마/시드 전용으로 강등.

**3단계 — 리뷰어가 기대할 평가 계획.**
(a) 고정된 TFM 아키텍처(TabPFNv2 또는 TabICL 코드 재사용)를 당신의 LLM 설계 prior로 사전학습; 아키텍처/컴퓨트를 고정하고 prior만 변화(O'PRIOR 프로토콜). (b) 다운스트림: WhyTrees/Grinsztajn(45) + TabZilla(176) + TALENT(~300) + TabArena(51 큐레이션); TabPFN v2, TabICL/TabICLv2, Mitra, TabForestPFN, 가능하면 Real-TabPFN/TabDPT 대비 보고. (c) Mitra의 다양성/구별성/성능 프레임워크(G 행렬, P 벡터)로 당신 prior를 특성화. (d) 합성 데이터 충실도(α-precision/β-recall, Wasserstein, C2ST)를 보고해 "현실성"을 정량화한 뒤, 현실성과 다운스트림 이득의 상관을 측정 — 당신의 prior에 대해 현실성 논쟁을 직접 판정하는 실험이 됩니다.

**4단계 — 오염/암기 감사를 실행하고 보고.**
interpretml LLM-Tabular-Memorization-Checker를 사용하고 Bordt et al.(arXiv:2404.06209)과 Ward et al.(arXiv:2512.08875)을 인용하세요. 임계값: LLM이 어떤 평가 데이터셋의 행을 그대로 재구성할 수 있으면(Iris/Wine/Adult/Diabetes/Titanic은 암기 확인됨) → 생성을 역할 (b) 경로(스키마+SCM만, 원문 값 없음)로 라우팅하고/하거나 해당 데이터셋을 평가에서 제외 — 그리고 이를 명시적으로 서술해 반론을 선제 무력화.

**5단계 — 현실성 논쟁에서 명시적 입장을 취하기.**
문헌에 근거할 때 방어 가능한 명제는 *조건부*일 가능성이 큽니다: 의미론적/현실적 prior는 저데이터·의미론적으로 풍부한·zero-shot 설정(세계 지식이 데이터를 대신하는 곳)에서 가장 도움이 되고, fine-tuning 후(TabForestPFN)이거나 의미론이 없는 곳(TabArena의 비의미론적 데이터셋)에서는 복잡도/다양성만으로 충분할 수 있다. 이 경계를 격리하는 실험을 최소 하나 설계하세요(예: CARTE-CLF 의미론 풍부 vs TabArena 비의미론 분할, zero-shot vs fine-tuned).

## 주의 사항

- **니어미스 위험:** Hod et al.(2504.14368, 2025년 4월)은 방법이 충분히 가까워 리뷰어가 인용할 것입니다; related work 첫머리에서 구분하고 묻어두지 마세요.
- **"현실성이 중요하지 않을 수 있다"는 반론은 살아 있고 증거가 있습니다**(TabForestPFN): 실세계 모사 prior가 더 낫다고 단언만 하지 말고 — 의미론/현실성이 *언제* 도움이 되고 언제 복잡도/다양성만으로 충분한지를 보이세요.
- **합성 사전학습 회의론**(Demystifying Synthetic Data, arXiv:2510.01631): 합성 데이터가 자동으로 우월하지 않습니다; 이득을 신중히 프레임하고 강한 real 데이터 baseline(Real-TabPFN/TabDPT)과 벤치마크하세요.
- **규모에서의 비용:** TFM 사전학습은 수백만 합성 데이터셋을 소비합니다; 그 규모에서 LLM 직접 행 생성은 경제적으로 실행 불가능할 가능성이 큽니다 — 역할 (b)(LLM이 구성, 프로그램이 샘플링)가 건전한 아키텍처인 추가 이유.
- **빠르게 움직이는 2026 프리프린트:** 인용된 여러 연구(O'PRIOR 2605.18971, TabICLv2 2602.11139, Zhu & Ler 2605.09518, Ward et al. 2512.08875, LimiX-2M 2606.04485)는 최근 arXiv 프리프린트입니다; 제출 전 최종 게재처/버전을 확인하세요.
- **잔여 조사 위험:** OpenReview 제출물 전수 크롤링은 수행하지 않았습니다; 이 아이디어에 더 가까운 미색인 워크숍 논문이 존재할 작은 가능성이 남습니다. "직접 경쟁자 없음" 판정은 2026년 7월 기준 arXiv/Semantic Scholar 전반에서 견고하지만 제출 시점에 재확인해야 합니다.
- **용어 함정:** APT의 "agents"는 적대적 데이터 생성기이고, "Causal Agent"(arXiv:2408.06849)는 기존 테이블을 *분석*하는 LLM입니다 — 어느 쪽도 LLM-에이전트-prior-설계 경쟁자가 아니므로, 리뷰어가 혼동하지 않도록 명확히 구분하세요.
