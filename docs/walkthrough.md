# Medical VQA for Chest X-rays — Literature Review Walkthrough

Here's the full walkthrough, paper by paper, building up to what we can actually do about it.

## The starting point

Your team was asked: find 5-6 papers on Medical VQA, figure out what's already been done, and find gaps we can improve on. Your teammate found 3 papers, I found 3 more to round it out to 6. Together they trace the field's evolution from "classify the answer" → "generate a chatbot-style answer" → "specialize for chest X-rays" → "add retrieval to reduce hallucination."

---

## Paper 1: PubMedCLIP (your teammate's find)

**What it does:** Takes CLIP (the model that learns to match images with text) and fine-tunes it on ~80K medical image-caption pairs from PubMed (a dataset called ROCO — covers X-ray, CT, MRI, ultrasound, many body parts, not just chest). Then plugs this fine-tuned CLIP in as the "eyes" of an older-style MedVQA pipeline that works like multiple-choice: given an image+question, pick the most likely answer from a fixed list seen during training.

**Result:** +1-6% accuracy bump on two small benchmarks (VQA-RAD, SLAKE).

**Why it matters / why it's limited:** It proved that swapping in a *better visual encoder* matters more than a better *question encoder* for this task. But the whole approach is **discriminative** — it can only output an answer it already saw during training, like a multiple-choice test. It cannot write a free sentence, explain itself, or handle a brand-new phrasing of a question. Not useful as a "chatbot."

---

## Paper 2: LLaVA-Med (your teammate's find)

**What it does:** Takes a general-purpose chatbot-style vision-language model (LLaVA) and turns it into a biomedical chatbot — cheaply (~15 hours on 8 GPUs). The clever trick: they didn't pay humans to write training conversations. Instead they fed GPT-4 just the *caption* of a medical image (not the image itself) and asked it to invent a plausible multi-turn Q&A about it. They did this for ~60K images across 5 modalities (chest X-ray is only one of the five).

**Result:** Clearly better than a non-medical chatbot at answering biomedical questions; after further fine-tuning, beats prior state-of-the-art on "closed" (yes/no-style) questions on standard benchmarks, but weaker on truly open-ended questions.

**Why it matters / why it's limited:** This is the first time we see the **generative chatbot** approach. But there's a real problem baked in: **GPT-4 never saw the actual image** when writing the training answers — just the caption. So the model learns to produce answers that *sound* plausible but might describe things not actually visible in the image. That's a hallucination risk built into the training data itself. Also, its evaluation (GPT-4 judging GPT-4-generated-style answers) is a bit circular — no real doctor checked it.

---

## Paper 3: Med-Flamingo (your teammate's find)

**What it does:** Instead of fine-tuning on a specific VQA dataset, this model is trained to be good at **few-shot, in-context learning** — you show it a handful of example Q&A pairs in the prompt, and it learns the pattern on the fly, no retraining needed. It's pretrained on a huge new corpus built from 4,721 actual medical textbooks, plus PubMed image-caption pairs.

**Result:** When they had **real doctors** (including a radiologist) score the model's answers 0-10 for clinical usefulness, Med-Flamingo came out on top of the baselines they compared it to.

**Why it matters most:** This paper's biggest contribution isn't the model — it's proving that **automated metrics lie to you.** They show cases where a model scores *well* on text-similarity metrics (like BERT-score) but a doctor rates its actual answer as nearly useless (1.23 out of 10!). This is a critical warning for our own evaluation plan: accuracy/BLEU numbers alone won't tell us if our system is actually good. They also found and fixed a **data leakage bug** in the standard VQA-RAD benchmark (same images appearing in train and test).

**Limitation:** Chest X-ray is a tiny slice of its textbook training data (dermatology and cardiology dominate) — it's a generalist, not a chest-X-ray specialist. The paper even says "retrieval-augmented few-shot prompting" is something they didn't get to — a direct pointer toward RAG.

---

## Paper 4: CheXagent (I found this one) — *the most important paper for our baseline decision*

**What it does:** This is the first model trained **100% on chest X-rays**, nothing else. The authors built a massive instruction dataset (CheXinstruct: 8.5 million examples) by combining 32 different public chest X-ray datasets, covering 35 different tasks — not just VQA, but also "what view is this," "has this finding gotten worse since the last scan," "draw a box around the abnormality," "write the report," etc.

**Result:** Even though their model is much *smaller* (3.1 billion parameters) than Med-Flamingo (9B) or LLaVA-Med (7B), it **beats all of them**, and even beats GPT-4V, on almost every chest X-ray task. They also ran a real study with 8 radiologists: using CheXagent-drafted reports saved residents 36% of their report-writing time, with no drop in quality.

**The single most useful number for us:** on easy tasks (like "what view is this X-ray taken from"), it scores ~99%. But on **fine-grained reasoning — telling left from right, mild from severe** — it only scores **77-79%**. That is *exactly* the kind of question your project brief uses as its running example ("which side shows a suspected finding"). This tells us precisely where the field's weak spot is.

**Why this matters for us:** It proves that **narrowing scope to just chest X-rays beats being a generalist**, even with a smaller/cheaper model. Since our whole project is scoped to chest X-rays only, this is strong validation — and it's a ready-made, open-source model we could literally use as our baseline instead of building one from scratch.

---

## Paper 5: PMC-VQA / MedVInT (I found this one)

**What it does:** This is the "scale up the data" baseline that other papers (including RULE, below) compare themselves against. They used ChatGPT to auto-generate 227,000 question-answer pairs from medical image captions (again, broad mix of modalities — not chest-X-ray-only), then trained a generative model on it.

**Result:** New best-in-class accuracy on the standard small benchmarks (81.6% on VQA-RAD, 88% on SLAKE). But they also built a harder, manually-checked test set, and on that one even their best model only gets ~35-40% — showing there's still a lot of room to improve on realistically hard questions.

**Key limitation (same theme as LLaVA-Med):** Since the questions are generated from *captions* rather than real human-asked questions, some questions can technically be answered just from word patterns in the caption, without actually needing to look at the image. They built a filter to catch this, but it's only ~82% accurate — meaning some "shortcut" questions still slip through. This is a recurring theme across the generative-data papers: **caption-derived datasets risk testing text pattern-matching rather than real visual understanding.**

---

## Paper 6: RULE (I found this one) — *the most important paper for our RAG decision*

**What it does:** This is the only paper of the six that actually implements **retrieval-augmented generation (RAG)** for a medical vision-language model, and its real contribution is diagnosing *why naive RAG backfires*.

**Two critical findings:**

1. **How much you retrieve matters a lot.** Retrieve too little and you miss key facts; retrieve too much and irrelevant/conflicting text confuses the model. They fix this with a statistically-calibrated method to pick the "just right" number of retrieved documents per case.
2. **This is the big one: the model over-trusts retrieved information, even when it was already right on its own.** They measured this directly: **47-59% of all the mistakes the RAG-augmented model made were cases where the model's un-aided answer was correct, but adding retrieved text flipped it to wrong.** They fix this with a fine-tuning step that specifically teaches the model "don't blindly trust the retrieved text if it contradicts what you can clearly see."

**Result:** With both fixes, they get a 47% average improvement in factual accuracy over the best prior open-source medical model, tested on MIMIC-CXR and IU X-ray (both chest X-ray datasets!).

**The gap this leaves for us:** RULE's "retrieval" is actually retrieving *another patient's similar-looking X-ray report* from a big database — not retrieving from a medical textbook, glossary, or clinical guideline. The original idea in your project brief — "retrieve relevant information from reliable medical references" — hasn't actually been tried by anyone in these six papers. That's real white space.

---

## Putting it all together: the pattern across all 6 papers

| | Formulation | Chest-X-ray specific? | Uses retrieval? | Uses uncertainty/"I don't know"? |
|---|---|---|---|---|
| PubMedCLIP | Multiple-choice | No | No | No |
| LLaVA-Med | Chatbot | No (1 of 5) | No | No |
| Med-Flamingo | Few-shot chatbot | No | No (named, not built) | No |
| **CheXagent** | Chatbot + classification | **Yes, fully** | No | No |
| PMC-VQA | Chatbot | No | No | No |
| **RULE** | Chatbot + RAG | Partially | **Yes (case-based)** | No |

Every single paper picks at most one thing to improve. **Nobody combines** (a) a model dedicated purely to chest X-rays, (b) retrieval of real medical *knowledge* (not just similar case reports), and (c) a way to signal uncertainty. That three-way gap is exactly where a small, realistic project contribution lives.

---

## The datasets — quick map

- **VQA-RAD, SLAKE** — small, classic benchmarks, but mix chest/brain/abdomen together and VQA-RAD even has a data-leakage bug. Fine for a quick sanity check if filtered to chest-only images.
- **PathVQA** — pathology, not radiology. Off-topic for us.
- **PMC-VQA, ROCO, PMC-OA** — broad PubMed-sourced pretraining data, not chest-specific.
- **CheXinstruct** — the big one: 8.5M chest-X-ray-only examples pooled from 32 sources. Most relevant resource if we go the fine-tuning route.
- **MIMIC-CXR (+ its VQA variants)** — the largest, most rigorous chest X-ray + report dataset, but requires a PhysioNet data-access credential, which takes time to get approved.
- **Open-I / IU X-ray** — smaller (7.4K images) but chest-X-ray-only, de-identified, and **requires no special credentialing** — the easiest real option for us to build our own small evaluation set.

---

## So, concretely, what can we do?

1. **Baseline:** Don't build a general medical VLM from scratch — use **CheXagent**, since it's already open-source, already chest-X-ray-specific, and small enough to actually run on our hardware. This satisfies the "pretrained VLM baseline" requirement immediately.

2. **The improvement direction I'd recommend: knowledge-grounded RAG.** Specifically — build a small curated knowledge base (e.g., short definitions of common chest X-ray findings, typical differentials, laterality/severity criteria — could come from an open radiology glossary or textbook excerpts) and retrieve from *that*, not from other patients' case reports like RULE does. This is the genuinely unexplored gap. We'd bake in RULE's two lessons from day one: don't just always retrieve a fixed number of passages (calibrate it), and don't let the model blindly trust retrieved text over what it can see in the image.

3. **A good stretch goal:** some lightweight way for the model to say "I'm not confident about this" — nobody in these six papers does this at all, and it's one of your project brief's three named directions.

4. **Also a good stretch goal:** since CheXagent itself shows "which side / how severe" questions are the weak point (77-79% vs 90%+ elsewhere), we could specifically target that with a small amount of extra fine-tuning data (e.g. using CheXpert's laterality labels) — that's a very concrete, benchmarked weakness to go after.

5. **Evaluation:** Don't rely on accuracy/BLEU alone — Med-Flamingo proved those can be misleading. Use a chest-filtered slice of VQA-RAD/SLAKE for quick numbers, plus a small hand-built set of ~30-50 Q&A pairs from Open-I, scored with a rubric that explicitly checks for unsupported/hallucinated claims (exactly what your brief asks us to watch for) — not just "is this the right label."

---

## What's already been delivered

All of this is written up in more depth and already committed to the repo:

- `docs/literature_review.md` — the full deep-dive with all the per-paper detail, tables, and the ranked gap list
- `docs/papers/` — all 6 source PDFs
- `slides/Medical_VQA_CXR_Literature_Review.pptx` — the 5-slide deck for presenting to your supervisor

**Natural next step:** decide if you want to commit to the RAG direction (my recommendation, based on the gap analysis) or lean more toward fine-tuning, and I can start scaffolding the actual baseline (getting CheXagent running on a sample chest X-ray + question) and/or the retrieval knowledge base. Let me know which you want to tackle first.
