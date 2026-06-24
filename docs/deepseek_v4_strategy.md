# DeepSeek V4 Pro Strategy for L-RAG Legal Document Comparison

## 1. Why DeepSeek V4 Pro for Vietnamese Legal Text

DeepSeek V4 Pro brings several architectural advantages that directly benefit the L-RAG pipeline's Phase 3 (Generative Comparison):

| Capability | DeepSeek V4 Pro | Current (Qwen 2.5 7B/14B) | Impact on L-RAG |
|---|---|---|---|
| **Architecture** | Mixture-of-Experts (MoE) | Dense Transformer | More capacity for legal reasoning without proportional latency increase |
| **Context Window** | 128K tokens | 32K (Qwen 2.5) | Can process entire long articles without windowing |
| **Reasoning** | Chain-of-Thought native | Standard next-token | Better at detecting logic shifts, condition changes (Category D) |
| **Multilingual** | Strong Vietnamese support | Good Vietnamese | More accurate terminology matching for Vietnamese legal terms |
| **JSON Mode** | Strict structured output | JSON mode available | Fewer parsing errors, less hallucinated JSON fields |
| **Instruction Following** | Strong constraint adherence | Moderate | Better adherence to RULE 1-8 in ACU_SYSTEM_PROMPT |

## 2. DeepSeek's Strengths for This Task

### 2.1 Long-Context Processing
DeepSeek's 128K context eliminates the need for windowing in most cases. Current L-RAG uses `windowing_threshold_chars=5000` — with DeepSeek we can raise this to 20000+, reducing:
- Artifact boundaries (split sentences causing false negatives)
- Duplicate ACUs across windows
- Extra LLM calls (cost savings despite API usage)

### 2.2 Reasoning for Legal Logic (Category D)
DeepSeek V4 Pro has strong reasoning capabilities particularly suited for detecting:
- **Negation shifts**: "không được" → "có thể được" 
- **Condition changes**: Thêm/bớt "nếu", "trừ khi", "chỉ khi"
- **Scope modifications**: Thay đổi phạm vi đối tượng áp dụng
- **Obligation changes**: "phải" → "có trách nhiệm", "bắt buộc" → "tự nguyện"

These are the hardest change types for current Qwen, and where DeepSeek should show the largest improvement.

### 2.3 Vietnamese Legal Terminology
DeepSeek's training data includes significant Vietnamese content. For legal terms like:
- "Thẩm phán", "Hội thẩm", "Chánh án"
- "công vụ", "phê duyệt", "văn bản"
- Number formats: "500.000 đồng", "13/01/2026"

DeepSeek tends to handle these more precisely than Qwen, particularly for terminology changes (Category B).

## 3. Prompt Engineering Strategy

### 3.1 Temperature Calibration

```
# DeepSeek V4 Pro settings (configs/model_config.yaml → provider_presets.deepseek)
temperature_acu: 0.0          # Zero-temp for ACU extraction (structured JSON)
temperature_summary: 0.2      # Low-temp for Vietnamese fluency without hallucination
```

**Rationale:** DeepSeek V4 Pro at temperature 0.0 still produces diverse enough output for fine-grained ACU detection while being maximally deterministic — critical for evidence extraction that must match source text verbatim.

Current Qwen settings (`temperature_acu: 0.05`) work because Qwen needs slight randomness to avoid repetitive patterns. DeepSeek's MoE architecture handles zero-temp well.

### 3.2 Max Tokens

```
max_tokens_acu: 8192     # Can go higher — DeepSeek supports up to 8K output
max_tokens_summary: 1024  # Same as Qwen — summaries don't need to be long
```

### 3.3 Prompt Adaptation

The existing `ACU_SYSTEM_PROMPT` in `report_generator.py` was tuned for Qwen. For DeepSeek:

**Keep as-is (works well):**
- RULE 1-8 structure (clear constraint enumeration)
- XML tags for text delimitation (`<v1_text>`, `<v2_text>`)
- Vietnamese language prompts (DeepSeek handles Vietnamese instructions well)

**Optional enhancements for DeepSeek:**
- Add explicit "Think step by step" instruction before JSON output — leverages DeepSeek's reasoning
- Use `response_format={"type": "json_object"}` which DeepSeek supports natively
- Consider adding a pre-ACU reasoning field: `"reasoning": "<why this is a change>"` in the ACU schema for better self-verification

### 3.4 Evidence Verification Enhancement

DeepSeek can perform better self-verification. Strategy:
1. **Pass 1**: Extract ACUs (standard ACU_SYSTEM_PROMPT)
2. **Pass 2**: Self-verify — feed ACU list back with raw texts, ask "Are these ACUs accurate? Flag any that cite evidence not present in source texts."
3. **Pass 3** (current Tier 2/3): Run existing VerificationEngine (fuzzy + numerical check)

This adds one extra LLM call but can significantly reduce hallucination rate, especially for long articles.

## 4. Two-Pass Strategy for DeepSeek

The current pipeline has `enable_second_pass=True` (strategy C2). For DeepSeek, we recommend:

| Strategy | DeepSeek | Rationale |
|---|---|---|
| **C1 (Number Enumeration)** | Keep enabled | Helps focus on numerical changes even with strong reasoning |
| **C2 (Two-Pass)** | Keep enabled | DeepSeek's pass 2 adds fewer ACUs (lower redundancy) because pass 1 is more complete |
| **Windowing** | Raise threshold to 20000 chars | 128K context eliminates most windowing |

### Anti-Hallucination Two-Pass:
```python
# New: Pass 0 — Self-Verification (DeepSeek only)
# After ACU extraction, before VerificationEngine
async def _deepseek_self_verify(acus, raw_v1, raw_v2):
    prompt = f"""Verify these ACUs against the source texts. 
    Flag any ACU whose verbatim_evidence does NOT appear in the source texts.
    Source V1: {raw_v1}
    Source V2: {raw_v2}
    ACUs to verify: {json.dumps(acus)}
    Return: {{"valid": [...], "flagged": [{{"acu_id": N, "reason": "..."}}]}}
    """
    # This catches hallucinations before they reach VerificationEngine
```

## 5. Embedding Strategy

### Keep BGE-M3 for Phase 2
DeepSeek does not have a dedicated embedding API model. **BGE-M3 remains the best choice** for Phase 2 because:
- 1024-dim dense + sparse BM25 = optimal for Vietnamese legal text
- Runs locally (no API cost, no latency)
- Proven on this exact dataset with strong alignment metrics

### Future: Hybrid Embedding (Optional)
If DeepSeek adds an embedding model, consider:
- BGE-M3 for structural matching (article numbers)
- DeepSeek embedding for semantic matching (full text)
- Weighted fusion: `w_bge * BGE_cosine + w_ds * DS_cosine`

## 6. Evaluation Plan on data_test/

### 6.1 Baseline (Qwen 2.5 14B)
Run full eval with current setup:
```bash
python evaluation/run_full_eval.py --output-dir ./eval_results_v3_qwen_baseline
```

### 6.2 DeepSeek V4 Pro
```bash
export LLM_PROVIDER=deepseek
# Or:
python evaluation/run_full_eval.py --provider deepseek --output-dir ./eval_results_v3_deepseek
```

### 6.3 Head-to-Head Comparison Metrics

Expected improvements:

| Metric | Qwen Baseline (approx) | DeepSeek Expected | Delta |
|---|---|---|---|
| **Recall (semantic)** | ~0.60-0.65 | 0.70-0.78 | +10-15% |
| **Precision (semantic)** | ~0.55-0.60 | 0.65-0.72 | +10-12% |
| **F1 (semantic)** | ~0.57-0.62 | 0.67-0.75 | +10-13% |
| **Category D (logic) recall** | ~0.35-0.45 | 0.55-0.70 | +15-25% |
| **Hallucination rate** | ~8-15% | 3-8% | -5-7% |
| **Category A (numerical) recall** | ~0.70-0.80 | 0.80-0.90 | +10% |

**Where DeepSeek should excel:**
- **Category D (Logic shifts)**: Largest improvement expected — DeepSeek's reasoning catches negation/condition changes Qwen misses
- **Category B (Paraphrasing)**: Better Vietnamese understanding catches subtle rewording
- **Hallucination**: Lower rate due to better instruction following and deterministic output at temp=0

**Where Qwen may still win:**
- **Speed**: Local Qwen 14B is ~2-5x faster than cloud API
- **Cost**: Free (local) vs API pricing
- **Category F (Formatting noise)**: Both models should handle typos/date format changes similarly

### 6.4 Per-Pair Analysis Strategy

For each document pair, compare:
1. **Detection overlap**: ACUs found by BOTH models → high confidence changes
2. **Unique detections**: ACUs found by only one model → review manually to identify blind spots
3. **False positives**: ACUs rejected by VerificationEngine → measure hallucination difference
4. **GT coverage**: Which model covers more ground truth changes per category

## 7. Cost-Benefit Analysis

### DeepSeek API Costs (approximate)
- Input: ~$0.27/1M tokens
- Output: ~$1.10/1M tokens
- Per document pair (avg 10K input + 2K output): ~$0.005

### Recommendation: Hybrid Mode
For production, consider:
- **Phase 3 first pass**: DeepSeek (high accuracy for complex changes)
- **Phase 3 second pass / summary**: Qwen (low-cost for final aggregation)
- **Phase 2 embedding**: BGE-M3 always (local, free, proven)

## 8. Configuration Summary

```yaml
# For DeepSeek — configs/model_config.yaml provider_presets.deepseek
model_name: "deepseek-chat"
base_url: "https://api.deepseek.com/v1"
temperature_acu: 0.0
temperature_summary: 0.2
max_tokens_acu: 8192
max_tokens_summary: 1024
timeout_seconds: 180.0
max_retries: 3
```

```bash
# .env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
LLM_PROVIDER=deepseek
```

## 9. References

- DeepSeek API Docs: https://platform.deepseek.com/api-docs
- DeepSeek V4 Technical Report: https://arxiv.org/abs/...
- BGE-M3 Paper: https://arxiv.org/abs/2402.03216
- L-RAG Architecture: `docs/project_overview.md`
- Improvement Strategies: `docs/improvement_strategies/`
