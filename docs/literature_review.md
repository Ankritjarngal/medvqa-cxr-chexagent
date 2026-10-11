# Literature Review: Medical Visual Question Answering for Chest X-rays

**Project:** Medical VQA system for chest X-ray analysis — baseline VLM + possible RAG/fine-tuning/uncertainty extension
**Scope of this document:** Deep-dive on 6 papers (3 found by teammate, 3 found below), a cross-paper comparison, a dataset catalogue, and a gap analysis pointing to concrete, small-but-real contributions this project can make.

---

## 0. The six papers at a glance

| # | Paper | Venue/Year | Found by | Core idea | VQA formulation |
|---|---|---|---|---|---|
| 1 | **PubMedCLIP** — *How Much Does CLIP Benefit VQA in the Medical Domain?* | EACL Findings 2023 | Teammate | Fine-tune CLIP on PubMed radiology images (ROCO) and plug it in as the **visual encoder** of existing classification-based MedVQA pipelines (MEVF, QCR) | Discriminative (classification over a fixed answer set) |
| 2 | **LLaVA-Med** — *Training a Large Language-and-Vision Assistant for Biomedicine in One Day* | arXiv 2023 (Microsoft) | Teammate | Curriculum-tune a general-domain LLaVA model on GPT-4–generated biomedical instruction data (from PMC figure captions) | Generative, instruction-tuned |
| 3 | **Med-Flamingo** — *A Multimodal Medical Few-Shot Learner* | arXiv 2023 (Stanford) | Teammate | Continue-pretrain Flamingo/OpenFlamingo on medical textbooks + PMC figure-caption pairs; answer via **few-shot in-context prompting**, no fine-tuning; evaluated by clinicians | Generative, few-shot, open-ended |
| 4 | **CheXagent** — *A Vision-Language Foundation Model to Enhance Efficiency of Chest X-ray Interpretation* | arXiv 2024 (Stanford AIMI) | **New** | Build a **chest-X-ray–only** instruction dataset (CheXinstruct, 8.5M triplets, 32 datasets, 35 tasks) and foundation model (3.1B params), with a dedicated benchmark (CheXbench) and a radiologist reader study | Generative + discriminative, multi-task, CXR-specific |
| 5 | **PMC-VQA / MedVInT** — *Visual Instruction Tuning for Medical Visual Question Answering* | arXiv 2023 | **New** | Reframe MedVQA as generation; build a 227k-pair auto-generated dataset (PMC-VQA) via ChatGPT-from-captions; two architectures (encoder-based / decoder-based) | Generative, pretrain-then-fine-tune |
| 6 | **RULE** — *Reliable Multimodal RAG for Factuality in Medical Vision-Language Models* | EMNLP 2024 | **New** | Add retrieval-augmented generation to a Med-LVLM (LLaVA-Med) but **fix RAG's failure modes**: calibrated selection of how many retrieved contexts to use, and preference-tuning the model to stop over-trusting retrieved text | Generative + RAG + factuality tuning |

Papers 1–3 establish *how MedVQA models are usually built* (discriminative → generative → few-shot). Papers 4–6 are the ones most relevant to *our specific decisions*: #4 is the only one specialized to chest X-rays end-to-end, #5 is the standard "scale up generative data" baseline that most follow-up papers (including RULE) compare against, and #6 is the only paper that actually operationalizes RAG for medical VLMs and — critically — documents **why naive RAG breaks** and how to fix it.

---

## 1. Deep dive per paper

### 1.1 PubMedCLIP (Eslami, Meinel & de Melo, EACL 2023)

- **Problem framing:** MedVQA-as-classification. A fixed-size answer vocabulary is predicted from (image, question) via bilinear attention (BAN) fusion, following the MEVF/QCR line of work.
- **Contribution:** Fine-tunes CLIP (ViT-B/32, RN50, RN50x4) on ROCO (~80K PubMed image–caption pairs spanning X-ray, CT, MRI, ultrasound, angiography, many body regions — **not CXR-specific**) and substitutes it for the previously-used MAML visual encoder.
- **Results:** +1–6% overall accuracy on VQA-RAD and SLAKE over MAML/CLIP baselines; new SOTA at publication (72.1% VQA-RAD, 80.1% SLAKE overall, with QCR+PubMedCLIP).
- **Key empirical insight:** The **visual encoder** matters far more than the **text encoder** for MedVQA — swapping in PubMedCLIP's text encoder for GloVe+RNN actually *hurts* performance, because PubMedCLIP's text embeddings are "organ-aware" but not "question-type-aware" (can't tell apart "is there a fracture" vs "describe the lung abnormality").
- **Limitations relevant to us:**
  - Still fundamentally **discriminative / closed-vocabulary** — not a chatbot, can't produce free-form explanations, can't handle questions whose answer never appeared verbatim in training data.
  - ROCO is multi-organ; chest-specific signal is diluted (chest is only one of ~9 body-location buckets in ROCO's proxy-label analysis).
  - No attempt at explanation, rationale, uncertainty, or retrieval.

### 1.2 LLaVA-Med (Li, Wong, Zhang et al., Microsoft 2023)

- **Problem framing:** Build an open-ended biomedical **visual chatbot**, not just a benchmark-accuracy model.
- **Data pipeline (the main contribution):** Sample image–caption pairs from PMC-15M; feed the caption text (plus sentences from the paper that reference the figure, "inline mentions") to **language-only GPT-4** and ask it to hallucinate a plausible multi-turn conversation *as if it could see the image*. 600K pairs → **Stage 1** concept-alignment (teach the model to associate CXR/CT/MRI/histology/gross-pathology visual features with vocabulary, projection layer only); 60K GPT-4-generated conversations (5 modalities, CXR is 1 of 5) → **Stage 2** end-to-end instruction tuning.
- **Results:** Outperforms general-domain LLaVA on a 193-question chat eval (judged by GPT-4) and, after dataset-specific fine-tuning, beats prior supervised SOTA on **closed**-set questions on VQA-RAD/SLAKE/PathVQA; weaker on **open**-set questions than specialized methods. Cheap to train: ~15 GPU-hours on 8×A100.
- **Key insight for us:** Domain-specific curriculum (generic captioning → instruction-following) transfers a general VLM into a usable biomedical assistant very cheaply — a template we could realistically replicate at small scale if we choose the fine-tuning route.
- **Limitations relevant to us:**
  - GPT-4 never sees the actual image when generating training conversations — only the caption/citances. This means the "ground truth" conversations can contain details that are *textually plausible but not visually verifiable*, baking a hallucination risk directly into the training data.
  - Evaluation of open-ended chat quality is **GPT-4 judging GPT-4-style outputs** — somewhat circular, and not validated against real clinicians (contrast with Med-Flamingo's physician study).
  - Chest X-ray is only 1/5 modalities in the instruction set — not CXR-specialized.

### 1.3 Med-Flamingo (Moor, Huang et al., Stanford, 2023)

- **Problem framing:** In-context **few-shot** generative VQA — no fine-tuning on the downstream VQA dataset at all; the model is prompted with a handful of (image, question, answer) exemplars and asked to answer a new case.
- **Data:** Continues pretraining OpenFlamingo-9B on **MTB** (a new interleaved corpus built from 4,721 medical textbooks, 0.8M images, 584M tokens) + PMC-OA (1.6M pairs). Pulmonology/radiology are a small slice of MTB's specialty distribution (dermatology and cardiology dominate).
- **Evaluation methodology — the standout contribution:** Introduces a **human clinician evaluation app** (3 MDs including a board-certified radiologist) that scores generated answers 0–10 for *clinical usefulness*, rather than relying only on BERTScore/exact-match. Also builds **Visual USMLE**, a new open-ended, cross-specialty, image+vignette benchmark. Explicitly finds and fixes **data leakage** in the official VQA-RAD train/test split.
- **Results:** Best average clinical-score rank (1.67) across VQA-RAD, PathVQA, and Visual USMLE, beating MedVINT and OpenFlamingo baselines by up to ~20% in clinician rating.
- **Key insight for us:** **Automated metrics are unreliable proxies for clinical quality.** BERT-sim and exact-match sometimes *disagree* with clinician scores (e.g., a fine-tuned MedVINT gets strong BERT-sim on PathVQA but a terrible 1.23/10 clinical score). Any serious evaluation of our prototype needs more than BLEU/accuracy — even a lightweight rubric-based or LLM-judge-with-image protocol is better than automated text-overlap metrics alone.
- **Limitations relevant to us:** Not CXR-specialized; relies on a 9B-parameter backbone (compute-heavy for a course project); few-shot prompting needs long context windows (expensive per-query); the paper itself names **retrieval-augmented few-shot prompting** as unrealized future work — this is exactly the gap RULE and our proposed extension address.

### 1.4 CheXagent (Chen, Varma, Delbrouck et al., Stanford AIMI, 2024) — *most relevant architecture reference*

- **Problem framing:** The first foundation model trained **exclusively and comprehensively for chest X-rays**, across 35 distinct tasks (not just VQA): view classification, disease ID, temporal progression, fine-grained reasoning, open/closed VQA, phrase grounding, findings generation, findings summarization.
- **Data:** **CheXinstruct** — 8.5M instruction triplets compiled from **32 public CXR datasets** (ChestXray14, CheXpert, MIMIC-CXR, PadChest, RSNA, VinDr-CXR, SIIM, OpenI, VQA-RAD, SLAKE, PMC-VQA, Rad-Restruct, MIMIC-CXR-VQA, MIMIC-Diff-VQA, RadGraph, etc.) — by far the largest CXR-specific instruction corpus assembled to date.
- **Architecture:** 3-stage training — (1) a 2.7B clinical-text LM (Phi-2) trained on 2.7B tokens of clinical notes/radiology reports/PubMed text; (2) a SigLIP image encoder adapted to CXRs via contrastive image-report training; (3) joint instruction tuning on CheXinstruct via a 2-layer MLP projector. Total 3.1B params — **far smaller** than Med-Flamingo (9B) or LLaVA-Med (7B LLM + vision tower), yet outperforms both.
- **Results:** Outperforms GPT-4V, LLaVA-Med, RadFM and QwenVL on nearly every CheXbench task: view classification 99.3% vs GPT-4V's 84.7%; VQA 96.7% (SLAKE) / 68.7% (Rad-Restruct, harder/unseen); **fine-grained reasoning only 77–79%** (side/region/severity disambiguation — e.g., "left-sided" vs "right-sided" pleural effusion) — markedly lower than the ~90%+ on coarse perception tasks. A clinical reader study with 8 radiologists shows CheXagent-drafted reports save residents **36% drafting time** with no quality loss.
- **Why this matters for our project:** This is direct evidence that **narrowing scope to one modality (chest X-ray) and building a dedicated instruction corpus beats general-purpose medical VLMs**, even with a much smaller model — validating our project's decision to scope to CXR only. It also pinpoints exactly the kind of question our project description calls out ("which side shows a finding") as the **hardest, least-solved task type** (fine-grained laterality/severity reasoning), i.e., a concrete target for improvement.
- **Limitations / open gaps:** Still evaluates VQA mostly as yes/no or multiple-choice (SLAKE, Rad-Restruct), not fully free-text open-ended; no retrieval augmentation; no uncertainty quantification; explicitly "not for clinical use."

### 1.5 PMC-VQA / MedVInT (Zhang, Wu, Zhao et al., 2023) — *standard generative baseline referenced by both RULE and LLaVA-Med*

- **Problem framing:** Scale up generative MedVQA training data by **auto-generating QA pairs from PMC figure captions with ChatGPT** (ask for 5 Q + 4 multiple-choice options per caption), then filter with two auxiliary models: (a) a text-only LLaMA to drop questions answerable *without* the image at all, (b) a classifier (81.8% test accuracy) to drop questions that need caption info *beyond* what's visible in the image.
- **Data:** 227k QA pairs over 149k images ("mixture" modality — radiology, pathology, microscopy, signals, illustrations), far larger than VQA-RAD (3.5k)/SLAKE (14k)/PathVQA (32.8k).
- **Architecture:** MedVInT-TE (encoder LM + mask-filling decoder) and MedVInT-TD (decoder-only LM), both bridging a PMC-CLIP visual encoder to the language model.
- **Results:** Pretraining on PMC-VQA before fine-tuning on VQA-RAD/SLAKE sets new SOTA (81.6% / 88.0% overall accuracy), beating PMC-CLIP's previous SOTA. On their own harder **PMC-VQA-test** benchmark, even their best model is only ~35–40% (vs ~25–30% random-guess floor) — showing current models are nowhere near solved on more diverse/compositional medical questions.
- **Key insight for us:** Even with an explicit **two-stage filtering pipeline** designed to guarantee "image is necessary to answer," the filtering is imperfect (manually-trained classifier at 82% accuracy) — this is a recurring theme (also seen in LLaVA-Med): **caption-derived QA datasets risk teaching/testing text-prior shortcuts rather than true visual grounding.** A good quality check for our own eval set is to verify a sample of questions truly require the image.
- **Limitations relevant to us:** Multi-modality mixture dilutes CXR-specific signal (radiology is one of several categories, not isolated to chest). No retrieval, no uncertainty, no CXR-only evaluation split reported.

### 1.6 RULE (Xia, Zhu, Li et al., UNC-Chapel Hill, EMNLP 2024) — *the key paper for our RAG decision*

- **Problem framing:** Takes an existing Med-LVLM (LLaVA-Med 1.5, 7B) and adds **retrieval-augmented generation**, but the paper's real contribution is diagnosing and fixing **why naive RAG for medical VLMs backfires**.
- **Two concrete, quantified failure modes:**
  1. **Context-count sensitivity:** too few retrieved reports miss relevant facts; too many introduce irrelevant/contradictory text that confuses the model. RULE fixes this with a **statistically-calibrated top-k selection** (a conformal-risk-control-style procedure) that is provably bounded in factuality risk, tuned per dataset on a validation split.
  2. **Over-reliance on retrieved context:** even when the model's own (no-retrieval) answer was *already correct*, adding retrieved context flips it to incorrect in **47.4% of IU-Xray errors, 47.4% of Harvard-FairVLMed errors, and 58.7% of MIMIC-CXR errors** — i.e., roughly half of all RAG-induced mistakes are the model trusting a retrieved report too much. RULE fixes this with **Knowledge-Balanced Preference Tuning (KBPT)**: a small DPO fine-tuning step using exactly these "retrieval broke a correct answer" cases as dispreferred examples.
- **Retriever design:** fine-tunes a ResNet-50 (vision) + BioClinicalBERT (text) CLIP-style contrastive retriever to match a query CXR to the most similar **training-set reports** (case-based/nearest-neighbor retrieval of real radiology reports, *not* retrieval from textbooks or clinical guidelines).
- **Datasets used:** **MIMIC-CXR** and **IU X-ray** (both chest X-ray!) plus Harvard-FairVLMed (ophthalmology, included to show cross-domain generality).
- **Results:** +47.4% average accuracy improvement on VQA (vs. MedVInT, the next-best open-source Med-LVLM) and large gains in report-generation factuality (BLEU/ROUGE-L/METEOR), beating decoding-time hallucination-mitigation baselines (DoLa, OPERA, VCD) by a wide margin.
- **Why this is the single most important paper for our "RAG vs fine-tuning" decision:** It proves (a) RAG *can* substantially improve factuality for CXR-focused Med-LVLMs, but (b) **naive RAG is not free** — if we add retrieval without calibrating how much to retrieve and without training the model to resist over-trusting bad retrievals, we could easily make the system *worse*, not better. Any RAG component we build **must** budget for both of RULE's fixes (or a lightweight approximation of them), not just "add a retriever and call it done."
- **Limitations / gap for us to fill:** RULE's retrieval corpus is *other patients' radiology reports* (case-based retrieval), **not external authoritative knowledge** (textbooks, clinical guidelines, finding-definition glossaries) — which is exactly the kind of RAG the project brief originally describes ("relevant information from reliable medical references"). This is an open, unaddressed direction: **knowledge-grounded RAG** (retrieving definitions/differential-diagnosis text for a *finding*, rather than retrieving a similar patient's report) has not been tried for CXR VQA in any of these six papers.

---

## 2. Cross-cutting comparison table

| Paper | Formulation | Backbone size | CXR-specific? | Training data scale | Evaluation | Uses retrieval? | Uses uncertainty/abstention? |
|---|---|---|---|---|---|---|---|
| PubMedCLIP | Discriminative (classification) | CLIP (ViT-B/32 / RN50) + BAN | No (multi-organ ROCO) | ~80K image-text pairs | Accuracy (VQA-RAD, SLAKE) | No | No |
| LLaVA-Med | Generative, instruction-tuned | 7B LLaMA/Vicuna + CLIP | No (1 of 5 modalities) | 600K align + 60K instruct | GPT-4-judged chat score + VQA-RAD/SLAKE/PathVQA accuracy | No | No |
| Med-Flamingo | Generative, few-shot (no fine-tune) | 9B (OpenFlamingo) | No (small slice of textbooks) | 0.8M img / 584M tok (MTB) + 1.3M (PMC-OA) | **Human clinician 0–10 score** + BERT-sim | No (named as future work) | No |
| **CheXagent** | Generative + discriminative, multi-task | **3.1B** (smallest!) | **Yes — fully CXR-specialized** | 8.5M triplets, 32 datasets | CheXbench (8 tasks) + **radiologist reader study** | No | No |
| PMC-VQA/MedVInT | Generative, pretrain+fine-tune | 7B (PMC-LLaMA) + PMC-CLIP | No (mixed modality) | 227K auto-generated pairs | Accuracy (VQA-RAD, SLAKE) + new harder PMC-VQA-test | No | No |
| **RULE** | Generative + **RAG** + preference tuning | 7B (LLaVA-Med 1.5) + ResNet50/BioClinicalBERT retriever | Partially (MIMIC-CXR, IU X-ray; + ophthalmology) | Fine-tune/retrieval corpus from MIMIC-CXR/IU-Xray reports | Accuracy/F1 (VQA) + BLEU/ROUGE/METEOR (report gen) | **Yes — case-based report retrieval, calibrated k** | No |

**Pattern:** every paper picks at most one axis to innovate on (bigger generative data, smaller CXR-only model, or RAG mechanics) — **none combines (a) a CXR-dedicated backbone, (b) knowledge-grounded retrieval from authoritative references, and (c) an explicit uncertainty/abstention layer.** That combination is open territory for us (see §4).

---

## 3. Dataset catalogue — what's out there and when to use what

| Dataset | Size (images / QA pairs) | Modality | Open vs closed QA | Notes / caveats |
|---|---|---|---|---|
| **VQA-RAD** | 315 img / 3.5k QA | Radiology (head, chest, abdomen mixed) | Both | Classic small benchmark; **official train/test split leaks images** (Med-Flamingo had to re-split); good for quick baseline sanity checks, not for a CXR-only story unless filtered to chest images only |
| **SLAKE (English)** | 642 img / ~7–14k QA | Radiology (brain, chest, abdomen, pelvis — multi-organ) | Both | Has semantic segmentation + knowledge-graph annotations; richer than VQA-RAD but still not CXR-exclusive; bilingual (EN/ZH) version exists |
| **PathVQA** | ~5k img / 32.8k QA | **Pathology**, not radiology | Both | Frequently used as a secondary MedVQA benchmark but off-topic for a CXR-only project — useful only as a cross-domain generalization check |
| **ROCO** | ~80K pairs | Mixed (X-ray/CT/MRI/US/angio, many body regions) | N/A (pretraining captions) | Used to train PubMedCLIP; chest is a minority sub-cluster |
| **PMC-OA / PMC-15M** | 1.6M / 15M pairs | Mixed biomedical figures | N/A (pretraining) | Backbone of LLaVA-Med and PMC-VQA; broad coverage, weak CXR density |
| **PMC-VQA** | 149k img / 227k QA | Mixed ("radiology, pathology, microscopy, signals, illustrations") | Both, auto-generated by ChatGPT | Largest auto-generated MedVQA set; includes a **manually-verified "test-clean" subset (2k)** — useful as a hard generalization probe, but not CXR-exclusive and has known caption-shortcut risk |
| **MTB (Med-Flamingo)** | 0.8M images / 584M tokens, 4,721 textbooks | All specialties (49 categories) | N/A (interleaved pretraining) | Pulmonology/radiology is a small slice relative to dermatology/cardiology |
| **Visual USMLE** | 618 problems | Cross-specialty, image + case vignette | Open-ended only | Great for testing *reasoning*, not useful for simple chest-finding QA |
| **CheXinstruct** | 8.5M triplets, 32 source datasets, ~1.08M unique images | **Chest X-ray only** | Both, 35 task types | **The single most relevant training/eval resource for this project** — rolls up MIMIC-CXR, CheXpert, ChestXray14, PadChest, RSNA, SIIM, VinDr-CXR, OpenI, VQA-RAD, SLAKE, PMC-VQA, Rad-Restruct, MIMIC-CXR-VQA, MIMIC-Diff-VQA, RadGraph and more |
| **MIMIC-CXR (+ MIMIC-CXR-VQA / MIMIC-Diff-VQA)** | 377k images / 227k studies (base); VQA variants derived from reports | Chest X-ray only | Both | Largest public CXR + report corpus (PhysioNet credentialed access); MIMIC-CXR-VQA and MIMIC-Diff-VQA auto-derive QA (incl. difference/progression questions) from report entities — directly usable as a CXR-only training/eval source |
| **CheXpert** | 224k images | Chest X-ray only | N/A (14-label classification, not QA) | Great for disease-presence labels/weak supervision, not natively QA-formatted |
| **Open-I / IU X-ray** | ~7.4k images / 3.9k reports | Chest X-ray only | N/A natively, convertible | Public, de-identified, **no PhysioNet credential required** — most practical real-image source for a course project with limited data-access timelines; used by RULE and CheXagent as held-out/retrieval corpus |
| **Rad-Restruct** | — | Chest X-ray (structured reports) | Closed (template QA) | Harder/more realistic held-out CXR VQA benchmark used by CheXagent |
| **RSNA Pneumonia / SIIM-ACR Pneumothorax / VinDr-CXR** | Various | Chest X-ray | N/A (detection/classification) | Good for building disease-specific sanity-check questions ("is there a pneumothorax?") with reliable ground truth |

**Practical recommendation for this project:** Prefer **Open-I (IU X-ray)** as the easiest-to-access, fully chest-X-ray, de-identified image+report source for building our own small QA eval set (no PhysioNet credentialing needed), and use **VQA-RAD/SLAKE filtered to chest-only images** plus a hand-curated set of 30–50 question/answer pairs (grounded in Open-I reports) for qualitative deep-dive evaluation. If PhysioNet credentialing is feasible within the timeline, **MIMIC-CXR-VQA / MIMIC-Diff-VQA** (used by CheXagent/RULE) are the more rigorous, larger-scale chest-only options.

---

## 4. Loopholes / gaps we can realistically target

Ranked roughly by how small-but-real the contribution is for a course-scale project:

1. **No paper does knowledge-grounded RAG for CXR VQA.** RULE retrieves *similar patients' reports* (case-based). Med-Flamingo explicitly names retrieval from verified medical literature as unrealized future work. **Gap:** retrieve short, authoritative passages (finding definitions, typical differential diagnoses, laterality/severity criteria) from a small curated knowledge base (e.g., a glossary of chest X-ray findings, Radiopaedia-style articles, or textbook excerpts) to ground the *generative* answer — not just paste in a similar patient's report. This is close to the project brief's original RAG idea and is genuinely unexplored in these six papers.

2. **RAG over-reliance is a known, quantified failure mode (RULE: ~47–59% of RAG errors) that a naive RAG implementation will reproduce if ignored.** Any RAG component we build needs at minimum a cheap approximation of RULE's two fixes: (a) tune/limit how many retrieved snippets are shown (don't just always use top-5), and (b) include some retrieved-context-wrong examples in a small amount of fine-tuning/prompt-engineering so the model learns not to blindly trust retrieval when it contradicts strong visual evidence.

3. **Fine-grained spatial/severity reasoning is the field's weakest point.** CheXagent — the strongest CXR-specific model — only hits ~77–79% on exactly the laterality/severity questions the project brief uses as its running example ("which side shows a suspected finding"), versus ~90%+ on coarse tasks like view classification. This is a concrete, benchmarked weak spot to target for improvement (e.g., targeted data augmentation, chain-of-thought prompting for side/region reasoning, or a small supervised fine-tune on side-labeled CXR data such as CheXpert's laterality annotations where available).

4. **Automated metrics systematically disagree with clinical usefulness** (Med-Flamingo shows cases where BERT-sim/exact-match and clinician scores diverge sharply). **Gap/contribution:** build a small but rigorous evaluation protocol beyond raw accuracy/BLEU — e.g., a rubric scoring (a) correctness, (b) relevance, (c) whether the answer contains *unsupported/hallucinated* claims not grounded in the image — exactly the axis the project brief asks us to pay attention to. This doesn't require real radiologists; a careful rubric + a secondary "does this claim appear in the image/report" check (possibly LLM-assisted but *verified against ground-truth reports*, not just LLM-judged in a vacuum like LLaVA-Med's evaluation) is a meaningful, feasible addition.

5. **Caption-derived training/eval data risks testing text-shortcut answering rather than real visual grounding** (both LLaVA-Med's self-instruct data and PMC-VQA's auto-generated QA have this risk, and PMC-VQA's own filter is only ~82% accurate). **Gap/contribution:** when we build our small hand-curated eval set, explicitly verify each question requires the image (not answerable from a report/caption alone) — a lightweight but genuine quality-control step most of these papers under-deliver on.

6. **No paper implements uncertainty communication or abstention.** This matches the project brief's third named direction. Even a lightweight approach — e.g., self-consistency across multiple sampled generations, or verbalized confidence calibrated against a held-out set, or reusing RULE's calibrated-risk idea but for "abstain if predicted factuality risk is high" rather than "pick k" — would be a novel addition none of these six papers attempt.

7. **General-purpose medical VLMs under-represent chest X-ray in their training mix** (Med-Flamingo's textbook corpus, LLaVA-Med's 5-modality split, PMC-VQA's broad modality mix). CheXagent's result (smaller model, CXR-only data, beats all the general ones) is strong evidence that **even a modest amount of additional CXR-only fine-tuning data on top of a general baseline** (rather than full retraining) could meaningfully help — a realistic, compute-light fine-tuning angle if the team leans that direction instead of/alongside RAG.

### Suggested project framing given all of the above

- **Baseline:** Use **CheXagent** (already open-sourced, CXR-specific, 3.1B params — far cheaper to run than Med-Flamingo/LLaVA-Med) as the pretrained baseline VLM, rather than a general-purpose medical VLM. This directly matches the "suitable pretrained vision-language model" requirement and sidesteps re-deriving CXR specialization from scratch.
- **Improvement direction (recommended): knowledge-grounded RAG**, informed by RULE's documented pitfalls — i.e., retrieve from a small curated set of authoritative CXR-finding descriptions (not case reports), calibrate how much retrieved text to inject, and sanity-check that retrieval doesn't override strong visual evidence.
- **Secondary/stretch direction:** a lightweight uncertainty/abstention signal and/or targeted fine-tuning on laterality-labeled data to attack the fine-grained reasoning weak spot CheXagent itself exposes.
- **Evaluation:** combine (a) accuracy on a chest-filtered VQA-RAD/SLAKE subset + a small Open-I-derived hand-built set, with (b) a qualitative rubric for relevance/unsupported-claims, explicitly because papers #2 and #3 show raw accuracy/BLEU is not trustworthy on its own.

---

## 5. Reference list (for citation in report/slides)

1. Eslami, S., Meinel, C., & de Melo, G. (2023). *PubMedCLIP: How Much Does CLIP Benefit Visual Question Answering in the Medical Domain?* Findings of EACL 2023.
2. Li, C., Wong, C., Zhang, S., et al. (2023). *LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day.* arXiv:2306.00890.
3. Moor, M., Huang, Q., Wu, S., et al. (2023). *Med-Flamingo: A Multimodal Medical Few-Shot Learner.* arXiv:2307.15189.
4. Chen, Z., Varma, M., Delbrouck, J.-B., et al. (2024). *A Vision-Language Foundation Model to Enhance Efficiency of Chest X-ray Interpretation (CheXagent).* arXiv:2401.12208.
5. Zhang, X., Wu, C., Zhao, Z., et al. (2023). *PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering.* arXiv:2305.10415.
6. Xia, P., Zhu, K., Li, H., et al. (2024). *RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models.* EMNLP 2024 / arXiv:2407.05131.

All six PDFs are stored in `docs/papers/` for reference.
