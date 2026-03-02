# Week 7 Blog: Hybrid Neutralization with Sentence + Word-Level Editing

## 1) Context and Goal

This week I focused on making the tone-neutralization pipeline more practical for short review-style texts. The project direction still follows the original proposal idea: detect tone, attribute what drives that tone, then perform local edits to neutralize language while keeping as much content as possible.

Compared with last week, the biggest changes are:

- I switched to a **hybrid attribution/editing policy**:
  - use **sentence-level** edits for longer clause-heavy inputs,
  - use **word-level** edits for short inputs.
- I upgraded the tone classifier base model to **`roberta-large`**.
- I manually reviewed and corrected part of the evaluation data to reduce ambiguous “impolite” labels and make the target standard more consistent.

---

## 2) What Changed in the Pipeline

### 2.1 Classifier and Evaluation

Using the updated model and curated evaluation file (`modified_ruder_dataset_v2.csv`), the batch attribution summary reports:

- **Accuracy: 0.80**
- **Macro-F1: 0.719**

These numbers come from:
- `artifacts/attribution_batch_extreme_full/summary.json`

| Metric | Value |
|---|---:|
| Accuracy | 0.8000 |
| Macro-F1 | 0.7193 |
| Labeled examples | 145 |
| Predicted impolite | 51 |
| Predicted neutral | 81 |
| Predicted polite | 13 |

Per-class metrics on the same run:

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| impolite | 0.7059 | 0.9231 | 0.8000 | 39 |
| neutral | 0.9136 | 0.7708 | 0.8362 | 96 |
| polite | 0.4615 | 0.6000 | 0.5217 | 10 |

### 2.2 Hybrid Neutralizer Strategy

I implemented a comparison script that keeps the old baseline intact and adds a new hybrid neutralization policy:

- **Old baseline (kept):** sentence-only iterative neutralizer  
  - `run_iterative_neutralizer_modal.py`
- **New hybrid (added):** sentence + word-level switch  
  - `run_iterative_neutralizer_hybrid_modal.py`

Hybrid routing rule:

- If comma-based clause count is **>= 3** → sentence-level deletion (Context-Cite path)
- If comma-based clause count is **< 3** → word/token deletion (attention token importance path)

---

## 3) Data Curation Note

I manually inspected examples and found that a non-trivial subset labeled as `impolite` were not clearly rude in normal human reading. To reduce label noise, I revised examples so that most `impolite` items in the final set are more unambiguous and clearly rude.

This improves evaluation consistency for the neutralization task (especially when success is measured as converting `impolite/polite` to `neutral`).

---

## 4) Neutralizer Outcomes (Hybrid Run)

From:
- `artifacts/neutralizer_hybrid_clause4/neutralizer_summary.json`

I got:

- total examples: **145**
- target examples attempted (`impolite` + `polite`): **64**
- successful neutralizations: **28**
- neutralization success rate over attempted: **43.75%**

The neutralization success rate is currently based on our tone evaluation model, the actual success rate might be different due to the limitation of our tone recognition accuracy.

| Neutralizer Metric | Value |
|---|---:|
| Total examples | 145 |
| Attempted (`impolite` + `polite`) | 64 |
| Successful neutralizations | 28 |
| Success rate over attempted | 43.75% |
| Skipped (already neutral) | 81 |
| Policy | clause count >= 3: sentence-level; else token-level |

---

## 5) Qualitative Examples

Below are representative cases from:
- `artifacts/neutralizer_hybrid_clause4/neutralized_results.jsonl`

### 5.1 Successful Cases

#### Success A (sentence-level, impolite → neutral)
- **Original (full):**  
  `To be blunt, this isn't very well done. In a nutshell, I am not sure if the problem is indeed a challenge or just a fact that we have to live with, such as gravity or death, for which no solutions exist. You need to explain this properly instead of glossing over it.`
- **Action:** removed the high-contribution directive sentence  
  `You need to explain this properly instead of glossing over it.`  
  and then removed  
  `To be blunt, this isn't very well done.`
- **Final (full):**  
  `In a nutshell, I am not sure if the problem is indeed a challenge or just a fact that we have to live with, such as gravity or death, for which no solutions exist.`
- **Result:** prediction flipped **impolite → neutral**


#### Success B (sentence-level, polite → neutral)
- **Original (full):**  
  `Thank you for your response to the reviewers comments and adequate addressing it in the revised manuscript. Avian rotavirus induce haemagglutination, so that you are right and it can be ruled out in the described case. Only minor comments: Line 3 correct domestica to not italic. Line 27 Escherichia coli should be italic. Line 49 "Avian" lower case "avian". Line 51, 67 lower case for circoviruses, aviadenoviruses.`
- **Action:** removed the leading explicitly polite sentence  
  `Thank you for your response to the reviewers comments and adequate addressing it in the revised manuscript.`
- **Final (full):**  
  `Avian rotavirus induce haemagglutination, so that you are right and it can be ruled out in the described case. Only minor comments: Line 3 correct domestica to not italic. Line 27 Escherichia coli should be italic. Line 49 "Avian" lower case "avian". Line 51, 67 lower case for circoviruses, aviadenoviruses.`
- **Result:** prediction flipped **polite → neutral**


#### Success C (sentence-level, polite → neutral)
- **Original (full):**  
  `Thank you for the opportunity to review this interesting paper. It can be accepted to publication; however, in your eFigure 1, in the 'Excluded' box, the text still seems to be cut ('Suicide attempt not recorded in ED or' is what I see in a pdf file and I cannot see what's after 'or'). Please make sure to check it.`
- **Action:** removed the opening gratitude sentence  
  `Thank you for the opportunity to review this interesting paper.`
- **Final (full):**  
  `It can be accepted to publication; however, in your eFigure 1, in the 'Excluded' box, the text still seems to be cut ('Suicide attempt not recorded in ED or' is what I see in a pdf file and I cannot see what's after 'or'). Please make sure to check it.`
- **Result:** prediction flipped **polite → neutral**


#### Success D (token-level, polite → neutral
- **Original (full):**  
  `Moreover , not having read Durand 2016 , I would appreciate a few more technical details or formal description here and there .`
- **Action:** removed token  
  `appreciate`
- **Final (full):**  
  `Moreover, not having read Durand 2016, I would a few more technical details or formal description here and there.`
- **Result:** prediction flipped **polite → neutral**




### 5.2 Unsuccessful but Representative Cases

#### Failure A (token-level, polite stays polite)
- **Original (full):**  
  `Thank you for your re-submission. This case report reads extremely well and will be a valuable addition tot he limited literature on the topic.`
- **Actions across rounds:** removed tokens like `This`, `case`, `report`, `reads`, `Thank`
- **Final (full):**  
  `you for your re-submission. extremely well and will be a valuable addition tot he limited literature on the topic.`
- **Result:** still **polite** after max rounds

In this sentence, praise is the message itself. Because tone carries most of the meaning, removing tone markers often leads to unnatural text.

#### Failure B (mixed strategy, impolite stays impolite)
- **Original (full):**  
  `Right now, this is embarrassingly underdeveloped. suggesting much but saying nothing of import, the sort of balderdash that is in the vernacular often compared to the waste of certain male ruminants. Clean it up, or don't submit it.`
- **Actions:** removed one harsh sentence + several tokens (`Clean`, `it`, `up`, `this`)
- **Final (full):**  
  `Right now, is embarrassingly underdeveloped., or don't submit it.`
- **Result:** still **impolite** after max rounds



### 5.3 Example Table (Quick Scan)

| Case | Initial → Final | Method | Key edit | Outcome |
|---|---|---|---|---|
| Success A | impolite → neutral | sentence | removed directive + blunt opener | flip achieved |
| Success B | polite → neutral | sentence | removed leading gratitude sentence | flip achieved |
| Success C | polite → neutral | sentence | removed opening gratitude sentence | flip achieved |
| Success D | polite → neutral | token | removed `appreciate` | flip achieved |
| Failure A | polite → polite | token | removed `This`, `case`, `report`, `reads`, `Thank` | no flip after max rounds |
| Failure B | impolite → impolite | mixed | removed 1 harsh sentence + tokens `Clean`, `it`, `up`, `this` | no flip after max rounds |

---

## 6) Key Takeaways

1. **sentence level is generally better, but not always applicable**: sentence-level generally produce better result, but if only couple sentences in the prompt, word-level will be useful, although deleting word might cause the mess up on grammar.
2. **Short texts benefit from word-level edits**, especially when tone is triggered by a few lexical markers.
3. **some failure are from the data**: some data is too short or itself not clear in tone label, so the actual accuracy are expected to be higher.
4. **Data clarity matters a lot**: manual de-ambiguation of rude labels makes both attribution analysis and neutralization evaluation more meaningful.

---

## 7) Next Steps

- Continue refining the dataset with a stronger focus on **extreme-tone** examples.
- Prioritize more reliable identification of clearly **overly impolite** and **overly polite** utterances.
- For `polite` vs `neutral` (and mildly polite language), explicit markers are often weak or ambiguous, even for human annotators.
- Therefore, our next-stage neutralization objective will focus on pulling clearly extreme queries back into a normal tone range.
- Improve stopping criteria to balance **neutralization success** and **readability preservation**.


---


