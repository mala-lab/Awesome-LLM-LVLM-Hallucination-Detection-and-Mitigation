 # Awesome-Hallucination-Detection-and-Mitigation


[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/mala-lab/Awesome-LLM-Hallucination-Detection-and-Mitigation)
[![Visits Badge](https://badges.pufler.dev/visits/mala-lab/Awesome-LLM-Hallucination-Detection-and-Mitigation)](https://badges.pufler.dev/visits/mala-lab/Awesome-LLM-Hallucination-Detection-and-Mitigation)


A collection of papers on LLM/LVLM hallucination evaluation benchmark, detection, and mitigation.

We will continue to update this list with the latest resources. If you find any missed resources (paper/code) or errors, please **feel free to open an issue or make a pull request**.

 

## Hallucinations Evaluation Benchmark 

- [Li2023] HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models in *EMNLP*, 2023. [\[paper\]](https://arxiv.org/abs/2305.11747)

- [Chen2024] FactCHD: Benchmarking Fact-Conficting Hallucination Detection in *IJCAI*, 2024. [\[paper\]](https://www.ijcai.org/proceedings/2024/0687.pdf)[\[code\]](https://github.com/zjunlp/FactCHD)  

- [Su2024] Unsupervised Real-Time Hallucination Detection based on the Internal States of Large Language Models in *Arxiv*, 2024.[\[paper\]](https://arxiv.org/pdf/2403.06448)[\[code\]](https://github.com/oneal2000/MIND/tree/main) 

- [Kossen2024] Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs in *Arxiv*, 2024.[\[paper\]](https://arxiv.org/pdf/2406.15927)[\[code\]](https://github.com/OATML/semantic-entropy-probes) 

- [Ji2024] ANAH: Analytical Annotation of Hallucinations in Large Language Models in *ACL*, 2024. [\[paper\]](https://arxiv.org/abs/2405.20315)

- [Simhi2024] Constructing Benchmarks and Interventions for Combating Hallucinations in LLMs in *Arxiv*, 2024. [\[paper\]](https://arxiv.org/abs/2404.09971)[\[code\]](https://github.com/technion-cs-nlp/hallucination-mitigation)  


## Causes of Hallucination 

- [Li2024] The Dawn After the Dark: An Empirical Study on Factuality Hallucination in Large Language Models in *Arxiv*, 2024. [\[paper\]](https://arxiv.org/abs/2401.03205)[\[code\]](https://github.com/RUCAIBox/HaluEval-2.0)  

- [Liu2025] More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models in *Arxiv*, 2025. [\[paper\]](http://arxiv.org/pdf/2505.21523)[\[code\]](https://github.com/MLRM-Halu/MLRM-Halu)  



## Hallucination Detection 

### Fact-checking 

- [Niu2024] RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models in *ACL*, 2024.  [\[paper\]](https://aclanthology.org/2024.acl-long.585.pdf)[\[code\]](https://github.com/ParticleMedia/RAGTruth)  

- [Chen2024] FactCHD: Benchmarking Fact-Conficting Hallucination Detection in *IJCAI*, 2024. [\[paper\]](https://www.ijcai.org/proceedings/2024/0687.pdf)[\[code\]](https://github.com/zjunlp/FactCHD)  
  
- [Zhang2024] KnowHalu: Hallucination Detection via Multi-Form Knowledge-Based Factual Checking in *Arxiv*, 2024. [\[paper\]](https://arxiv.org/pdf/2404.02935)[\[code\]](https://github.com/javyduck/KnowHalu)

-  [Rawte2024] FACTOID: FACtual enTailment fOr hallucInation Detection in *Arxiv*, 2024. [\[paper\]](https://arxiv.org/abs/2403.19113)[\[code\]]() 

-  [Es2024] RAGAs: Automated evaluation of retrieval augmented generation in *EACL*, 2024. [\[paper\]](https://aclanthology.org/2024.eacl-demo.16.pdf)[\[code\]](https://github.com/explodinggradients/ragas) 

- [Hu2024]  RefChecker: Reference-based Fine-grained Hallucination Checker and Benchmark for Large Language Models in *Arxiv*, 2024. [\[paper\]](https://arxiv.org/abs/2405.14486)[\[code\]](https://github.com/amazon-science/RefChecker) 

- [Zhang2025] CORRECT: Context- and Reference-Augmented Reasoning and Prompting for Fact-Checking, in *NAACL*, 2025. [\[paper\]](https://drive.google.com/file/d/1-2fieczt68O5SCFhhC4kOC7lp3D6oDrN/view)[\[code\]](https://github.com/cezhang01/correct) 

- [Lee2025] Enhancing Hallucination Detection via Future Context, in *Arxiv*, 2025. [\[paper\]](https://arxiv.org/abs/2507.20546)[\[code\]]() 

### Uncertainty Analysis 

- [Zhang2023] Enhancing Uncertainty-Based Hallucination Detection with Stronger Focus in *EMNLP*, 2023. [\[paper\]](https://aclanthology.org/2023.emnlp-main.58v2.pdf)[\[code\]](https://github.com/zthang/focus) 

- [Snyder2024] On Early Detection of Hallucinations in Factual Question Answering in *KDD*, 2024.[\[paper\]](https://arxiv.org/pdf/2312.14183)[\[code\]](https://github.com/amazon-science/llm-hallucinations-factual-qa)

- [Chuang2024] Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps in *EMNLP*, 2024.  [\[paper\]](https://arxiv.org/abs/2407.07071)[\[code\]](https://github.com/voidism/Lookback-Lens)

- [Ji2024]  LLM Internal States Reveal Hallucination Risk Faced With a Query in *Arxiv*, 2024. [\[paper\]](https://arxiv.org/pdf/2407.03282)[\[code\]](https://github.com/ziweiji/Internal_States_Reveal_Hallucination)

- [Bouchard2025] Uncertainty Quantification for Language Models: A Suite of Black-Box, White-Box, LLM Judge, and Ensemble Scorers in *Arxiv*, 2025.  [\[paper\]](https://arxiv.org/abs/2504.19254)[\[code\]](https://github.com/cvs-health/uqlm)

- [Ma2025] Semantic Energy: Detecting LLM Hallucination Beyond Entropy in *Arxiv*, 2025.  [\[paper\]](https://arxiv.org/pdf/2508.14496)[\[code\]](https://github.com/MaHuanAAA/SemanticEnergy/blob/main/semantic_energy.ipynb)

### Consistency Measure 

- [Cohen2023] LM vs LM: Detecting Factual Errors via Cross Examination in *Arxiv*, 2023. [\[paper\]](https://arxiv.org/abs/2305.13281)[\[code\]]() 

- [Manakul2023] SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models in *EMNLP*, 2023. [\[paper\]](https://arxiv.org/abs/2303.08896)[\[code\]](https://github.com/potsawee/selfcheckgpt) 

- [Chen2023] Hallucination Detection: Robustly Discerning Reliable Answers in Large Language Models in *CIKM*, 2023. [\[paper\]](https://arxiv.org/abs/2407.04121)[\[code\]]() 

- [Su2024] Unsupervised Real-Time Hallucination Detection based on the Internal States of Large Language Models in *Arxiv*, 2024.[\[paper\]](https://arxiv.org/pdf/2403.06448)[\[code\]](https://github.com/oneal2000/MIND/tree/main) 

- [Mündler2024] Self-Contradictory Hallucinations of LLMs: Evaluation, Detection and Mitigation in *ICLR*, 2024.[\[paper\]](https://arxiv.org/pdf/2305.15852)[\[code\]](https://chatprotect.ai/)

- [Kossen2024] Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs in *ICML*, 2024.  [\[paper\]](https://arxiv.org/pdf/2406.15927)[\[code\]](https://github.com/OATML/semantic-entropy-probes)

- [Xu2024] Hallucination is Inevitable:An Innate Limitation of Large Language Models in *Arxiv*, 2024. [\[paper\]](https://arxiv.org/pdf/2401.11817)[\[code\]]()

- [Niu2025] Robust Hallucination Detection in LLMs via Adaptive Token Selection in *NeurIPS*, 2025.[\[paper\]](https://arxiv.org/abs/2504.07863)[\[code\]](https://github.com/mala-lab/HaMI)

- [Sun2025] Why and How LLMs Hallucinate: Connecting the Dots with Subsequence Associations in *Arxiv*, 2025. [\[paper\]](https://arxiv.org/abs/2504.12691)[\[code\]](https://github.com/sunyiyou/SAT) 

- [Islam2025] How Much Do LLMs Hallucinate across Languages? On Multilingual Estimation of LLM Hallucination in the Wild in *Arxiv*, 2025. [\[paper\]](https://arxiv.org/pdf/2502.12769)[\[code\]]() 

- [Muhammed2025] SelfCheckAgent: Zero-Resource Hallucination Detection in Generative Large Language Models in *Arxiv*, 2025. [\[paper\]](https://arxiv.org/pdf/2502.01812)[\[code\]](https://github.com/potsawee/selfcheckgpt) 

- [Yang2025] Hallucination Detection in Large Language Models with Metamorphic Relations in *FSE*, 2025. [\[paper\]](https://arxiv.org/pdf/2502.15844)[\[code\]](https://github.com/zbybr/LLMhalu/tree/MetaQA-Open-Base) 


### Hidden States Analysis

- [Azaria2023] The internal state of an llm knows when it’s lying in *EMNLP findings*, 2023.  [\[paper\]](https://aclanthology.org/2023.findings-emnlp.68.pdf)[\[code\]](https://github.com/zthang/focus) 

- [Chen2024] INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection in *ICLR*, 2024. [\[paper\]](https://arxiv.org/abs/2402.03744)[\[code\]](https://github.com/D2I-ai/eigenscore)

- [Kuhn2023] Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation in *ICLR*, 2023. [\[paper\]](https://arxiv.org/pdf/2302.09664)[\[code\]]()

- [Farquhar2024] Detecting Hallucinations in Large Language Models Using  Semantic Entropy in *Nature*,2024. [\[paper\]](https://www.nature.com/articles/s41586-024-07421-0)[\[code\]]()

-  [Sriramanan2024] LLM-Check: Investigating Detection of Hallucinations in Large Language Models in *NeurIPS*, 2024. [\[paper\]](https://proceedings.neurips.cc/paper_files/paper/2024/file/3c1e1fdf305195cd620c118aaa9717ad-Paper-Conference.pdf)[\[code\]]()

- [Wang2025]  Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation in *ICLR*, 2025. [\[paper\]](https://arxiv.org/abs/2410.13640)[\[code\]](https://github.com/Alsace08/Chain-of-Embedding) 

- [Zhang2025] ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs in *ACL*, 2025. [\[paper\]](https://arxiv.org/pdf/2507.16488)[\[code\]]() 

- [Cheang2025] Large Language Models Do NOT Really Know What They Don't Know in *arXiv*, 2025. [\[paper\]](https://arxiv.org/abs/2510.09033)[\[code\]]() 


###  RL Reasoning

- [Su2025] Learning to Reason for Hallucination Span Detection in *Arxiv*, 2025. [\[paper\]](https://arxiv.org/abs/2510.02173)[\[code\]]() 



## Hallucination Mitigation 



### Model Calibration 

- [Li2023] Inference-Time Intervention: Eliciting Truthful Answers from a Language Model in *NeurIPS*, 2023. [\[paper\]](https://arxiv.org/abs/2306.03341)[\[code\]](https://github.com/likenneth/honest_llama)

- [Liu2023] LitCab: Lightweight Language Model Calibration over Short- and Long-form Responses in *ICLR*,2023.  [\[paper\]](https://arxiv.org/abs/2310.19208)[\[code\]](https://github.com/launchnlp/LitCab)

- [Ji2023] Towards Mitigating Hallucination in Large Language Models via Self-Reflection in *EMNLP findings*, 2023. [\[paper\]](https://arxiv.org/abs/2310.06271)

- [Chen2023] PURR: Efficiently Editing Language Model Hallucinations by Denoising Language Model Corruptions in *Arxiv*, 2023 [\[paper\]](https://arxiv.org/pdf/2305.14908)

- [Campbell2023] Localizing Lying in Llama: Understanding Instructed Dishonesty on True-False Questions Through Prompting, Probing, and Patching in *Arxiv*, 2023. [\[paper\]](https://arxiv.org/pdf/2311.15131)

- [Wan2023] Faithfulness-Aware Decoding Strategies for Abstractive Summarization in *EACL*, 2023.  [\[paper\]](https://aclanthology.org/2023.eacl-main.210.pdf)[\[code\]](https://github.com/amazon-science/faithful-summarization-generation)

- [Shi2023] Trusting Your Evidence: Hallucinate Less with Context-aware Decoding in *Arxiv*, 2023. [\[paper\]](https://arxiv.org/pdf/2305.14739)

- [Chen2024] Truth Forest: Toward Multi-Scale Truthfulness in Large Language Models through Intervention without Tuning in *AAAI*,2024.  [\[paper\]](https://arxiv.org/abs/2312.17484)[\[code\]]()

- [Zhang2024] R-Tuning: Instructing Large Language Models to Say `I Don't Know' in *NAACL*,  2023.  [\[paper\]](https://arxiv.org/abs/2311.09677)[\[code\]](https://github.com/shizhediao/R-Tuning)

- [Chuang2024] DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models in *ICLR*, 2024. [\[paper\]](https://arxiv.org/abs/2309.03883)[\[code\]](https://github.com/voidism/DoLa)

- [Kapoor2024] Calibration-Tuning: Teaching Large Language Models to Know What They Don’t Know in *UncertaiNLP*, 2024. [\[paper\]](https://aclanthology.org/2024.uncertainlp-1.1/)

- [Zhang2024] TruthX: Alleviating Hallucinations by Editing Large Language Models in Truthful Space in *ACL*, 2024. [\[paper\]](https://arxiv.org/abs/2402.17811)[\[code\]](https://ictnlp.github.io/TruthX-site/)

- [Zhou2025] HaDeMiF: Hallucination Detection and Mitigation in Large Language Models in *ICLR*, 2025. [\[paper\]](https://openreview.net/pdf?id=VwOYxPScxB)[\[code\]]()

- [Zhang2025] The Law of Knowledge Overshadowing: Towards Understanding, Predicting, and Preventing LLM Hallucination in *ACL*, 2025. [\[paper\]](https://arxiv.org/abs/2502.16143)[\[code\]]()

- [Wu2025] Mitigating Hallucinations in Large Vision-Language Models via Entity-Centric Multimodal Preference Optimization in *Arxiv*, 2025. [\[paper\]](https://arxiv.org/abs/2506.04039)[\[code\]](https://github.com/RobitsG/EMPO)

- [Cheng2025]  Integrative Decoding: Improving Factuality via Implicit Self-consistency in *ICLR*, 2025. [\[paper\]](https://openreview.net/pdf?id=gGWYecsK1U)[\[code\]](https://github.com/YiCheng98/IntegrativeDecoding) 

- [Yang2025] Nullu: Mitigating Object Hallucinations in Large Vision-Language Models via HalluSpace Projection in *CVPR*, 2025. [\[paper\]](https://arxiv.org/abs/2412.13817)[\[code\]](https://github.com/Ziwei-Zheng/Nullu) 

- [Wan2025] ONLY:One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Model in *ICCV*, 2025. [\[paper\]](https://arxiv.org/abs/2507.00898)[\[code\]]() 

- [Chang2025]  Monitoring Decoding: Mitigating Hallucination via Evaluating the  Factuality of Partial Response during Generation in *Arxiv*, 2025. [\[paper\]](https://arxiv.org/abs/2503.03106)[\[code\]]() 

- [Wang2025] Image Tokens Matter: Mitigating Hallucination in Discrete Tokenizer-based Large Vision-Language Models via Latent Editing  in *Arxiv*, 2025. [\[paper\]](https://arxiv.org/abs/2505.21547)[\[code\]](https://github.com/weixingW/CGC-VTD/tree/main) 



### External Knowledge 

- [Ji2022] RHO (ρ): Reducing Hallucination in Open-domain Dialogues with Knowledge Grounding in *ACL findings*, 2022. [\[paper\]](https://arxiv.org/abs/2212.01588)

- [Kang2024] Unfamiliar Finetuning Examples Control How Language Models Hallucinate in *Arxiv*, 2024. [\[paper\]](https://arxiv.org/abs/2403.05612)

- [Gekhman2024] Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? in *EMNLP*, 2024. [\[paper\]](https://arxiv.org/pdf/2405.05904)
  
- [Sun2025] Redeep: Detecting hallucination in retrieval-augmented generation via mechanistic interpretability  in *ICLR*, 2025. [\[paper\]](https://arxiv.org/pdf/2410.11414)[\[code\]](https://github.com/Jeryi-Sun/ReDEeP-ICLR)

- [Dey2025] Uncertainty-Aware Fusion: An Ensemble Framework for Mitigating Hallucinations in Large Language Models in *WebConf*, 2025. [\[paper\]](https://arxiv.org/abs/2503.05757)[\[code\]]()

- [Lavrinovics2025] MultiHal: Multilingual Dataset for Knowledge-Graph Grounded Evaluation of LLM Hallucinations in *Arxiv*, 2025. [\[paper\]](https://arxiv.org/abs/2505.14101)[\[code\]](https://github.com/ernlavr/multihal)

- [Sui2025] Bridging External and Parametric Knowledge: Mitigating Hallucination of LLMs with Shared-Private Semantic Synergy in Dual-Stream Knowledge in *Arxiv*, 2025. [\[paper\]](https://arxiv.org/pdf/2506.06240)[\[code\]]()

- [Ferrando2025]  Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models in *ICLR*, 2025.  [\[paper\]](https://arxiv.org/abs/2411.14257)[\[code\]](https://github.com/javiferran/sae_entities)

- [Xue2025] UALIGN: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models in *ACL*, 2025. [\[paper\]](https://arxiv.org/pdf/2412.11803)[\[code\]](https://github.com/AmourWaltz/UAlign)

- [Cheng2025]  Small Agent Can Also Rock! Empowering Small Language Models as Hallucination Detector in *EMNLP*, 2024. [\[paper\]](https://aclanthology.org/2024.emnlp-main.809.pdf)[\[code\]](https://github.com/RUCAIBox/HaluAgent)


### Alignment-Fine-tuning

- [Lee2022] Factuality Enhanced Language Models for Open-Ended Text Generation in *NeurIPS*, 2022. [\[paper\]](https://arxiv.org/abs/2206.04624)[\[code\]](https://github.com/nayeon7lee/FactualityPrompt)

- [Tian2023] Fine-tuning Language Models for Factuality in *ICLR*, 2023. [\[paper\]](https://arxiv.org/abs/2311.08401)[\[code\]](https://github.com/kttian/llm_factuality_tuning)

- [Lin2024] FLAME: Factuality-Aware Alignment for Large Language Models in *NeurIPS*, 2024.[\[paper\]](https://arxiv.org/abs/2405.01525)

- [Yang2024] V-DPO: Mitigating Hallucination in Large Vision Language Models via Vision-Guided Direct Preference Optimization in *EMNLP Findings*, 2024. [\[paper\]](https://arxiv.org/abs/2411.02712)[\[code\]](https://github.com/YuxiXie/V-DPO)

- [Kang2024] Unfamiliar finetuning examples control how language in *NAACL*, 2024. [\[paper\]](https://arxiv.org/pdf/2403.05612)[\[code\]](https://github.com/katiekang1998/llm_hallucinations)

- [Yang2025] Mitigating Hallucinations in Large Vision-Language Models via DPO: On-Policy Data Hold the Key in *CVPR*, 2025.  [\[paper\]](https://arxiv.org/abs/2501.09695)[\[code\]](https://github.com/zhyang2226/OPA-DPO)

- [Gu2025] Mask-DPO: Generalizable Fine-grained Factuality Alignment of LLMs in *ICLR*, 2025. [\[paper\]](https://arxiv.org/abs/2503.02846)[\[code\]](https://github.com/open-compass/ANAH)



# Related Survey 

- [Wang2023] Survey on Factuality in Large Language Models: Knowledge, Retrieval and Domain-Specificity in *Arxiv*,2023. [\[paper\]](https://arxiv.org/abs/2310.07521) 
  
- [Ye2023] Cognitive Mirage: A Review of Hallucinations in Large Language Models  in *Arxiv*,2023. [\[paper\]](https://arxiv.org/abs/2309.06794v1) 
  
- [Zhang2023] Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models in *Arxiv*,2023. [\[paper\]](https://arxiv.org/abs/2309.01219)

- [Gao2023] Retrieval-augmented generation for large language models: A survey in *Arxiv*, 2023. [\[paper\]](https://arxiv.org/pdf/2312.10997)

- [Huang2024] A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions in *TOIS*, 2024. [\[paper\]](https://arxiv.org/pdf/2311.05232)

- [Ji2024] Survey of Hallucination in Natural Language Generation in *CSUR*, 2024.  [\[paper\]](https://arxiv.org/abs/2202.03629)

- [Bai2024] Hallucination of Multimodal Large Language Models: A Survey in *Arxiv*, 2024. [\[paper\]](https://arxiv.org/abs/2404.18930)

- [Chen2025] A Survey of Multimodal Hallucination Evaluation and Detection in *Arxiv*, 2025. [\[paper\]](https://arxiv.org/pdf/2507.19024)

- [Lin2025] LLM-based Agents Suffer from Hallucinations: A Survey of Taxonomy, Methods, and Directions in *Arxiv*, 2025. [\[paper\]](https://arxiv.org/abs/2509.18970)
 

# Datasets


