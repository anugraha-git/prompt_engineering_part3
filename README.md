# Part 3 — Prompt Engineering and Customer Review Analysis

## 1. Project Overview

This project uses Google Gemini through a Python-based LLM pipeline to analyze customer reviews from the Women's Clothing E-Commerce Reviews dataset.

The project demonstrates prompt engineering techniques, structured JSON outputs, aspect-based sentiment analysis, automated customer-response generation, and multi-turn conversation handling.

The main objective is to compare different prompting strategies and build a reusable pipeline for extracting useful customer insights from clothing reviews.

---

## 2. Objectives

The project covers the following tasks:

1. Create and compare zero-shot, few-shot, and role-prompted templates.
2. Run a prompt experiment across multiple customer reviews.
3. Evaluate the consistency and validity of structured model outputs.
4. Identify the best-performing prompt approach based on the available experiment results.
5. Perform aspect-based sentiment analysis.
6. Generate customer-service responses from structured customer insights.
7. Demonstrate a two-turn conversation using conversation history.

---

## 3. Dataset

The project uses the **Women's Clothing E-Commerce Reviews** dataset.

The dataset contains customer reviews and associated information about women's clothing products.

The analysis primarily uses the:

* `Review Text` column

Data preparation includes:

* Loading the CSV dataset using pandas.
* Removing the unnecessary `Unnamed: 0` index column.
* Removing rows where `Review Text` is missing.
* Using the cleaned review text for LLM-based analysis.

---

## 4. Project Structure

```text
part3/
│
├── data/
│   └── Womens Clothing E-Commerce Reviews.csv
│
├── prompts/
│   ├── __init__.py
│   └── prompt_templates.py
│
├── outputs/
|       └── prompt_comparison_results.json
│   
│
├── analysis.py
├── llm_utils.py
├── prompt_experiments.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

Output JSON files for Tasks 5–7 are generated when API execution is enabled and successful.

---

## 5. Prompt Templates

Three main prompt templates were implemented.

### Zero-Shot Prompt

The model receives a customer review without any labelled examples and is asked to classify its overall sentiment.

### Few-Shot Prompt

The model receives multiple labelled examples before classifying the new review.

The template contains five examples to demonstrate the expected classification behaviour.

### Role-Prompted Template

The model is given an explicit role/persona in addition to the classification instructions.

The role is designed to encourage the model to approach the review as a customer-insight/sentiment-analysis task.

### Structured Output

All three templates use the same JSON schema:

```json
{
    "label": "Positive | Negative | Mixed",
    "confidence": "High | Medium | Low",
    "reason": "Brief explanation"
}
```

This provides a consistent structure that can be parsed programmatically.

---

## 6. Aspect-Based Sentiment Analysis

An additional prompt template was implemented for aspect-based analysis.

The analysis focuses on at least two customer-review aspects:

* **Product Quality**
* **Fit & Comfort**

The aspect prompt also requires an actionable phrase of approximately 3–6 words, making the extracted insight more useful for downstream customer-service or product analysis.

The results are designed to be stored as structured JSON.

---

## 7. Prompt Experiment

The prompt experiment evaluates three templates across five customer reviews:

* 3 prompting strategies
* 5 reviews per strategy
* 15 total API calls attempted

The experiment programmatically attempts to parse each model response using Python's `json.loads()`.

### Experiment Result

The 15-call experiment was attempted successfully, but the Gemini free-tier API quota was exhausted during execution.

The recorded results were:

| Prompt Type   | Valid JSON Results |
| ------------- | -----------------: |
| Zero-Shot     |              3 / 5 |
| Few-Shot      |              1 / 5 |
| Role-Prompted |              0 / 5 |
| **Total**     |         **4 / 15** |

Some responses also failed JSON parsing, while subsequent requests returned `429 RESOURCE_EXHAUSTED` because the Gemini free-tier request quota had been reached.

Therefore, the results should be interpreted as an execution snapshot rather than a statistically reliable comparison of prompt quality.

The experiment code remains available in the project for re-execution when API access is available.

---

## 8. Task 5 — Aspect Analysis

Task 5 implements an API-driven aspect-analysis pipeline for the first 10 cleaned customer reviews.

For each review, the pipeline:

1. Creates an aspect-analysis prompt.
2. Sends the prompt to Gemini.
3. Parses the response as JSON.
4. Records whether the returned response is valid JSON.
5. Saves the results to:

```text
outputs/aspect_analysis_results.json
```

During the submitted experiment, the API became unavailable/quota-limited and the 10 aspect-analysis calls did not produce usable final JSON results.

The implementation itself remains complete and can be executed when Gemini API quota is available.

---

## 9. Task 6 — Auto-Drafted Customer Responses

Task 6 uses structured customer insights to generate concise and professional customer-service responses.

The response-generation prompt instructs the model to:

* Address the specific points raised by the customer.
* Acknowledge positive points where appropriate.
* Address negative points where appropriate.
* Avoid generic responses.
* Avoid inventing refunds, policies, or actions.
* Keep the response concise.

The pipeline is designed to use the first three valid aspect-analysis results and save the generated responses to:

```text
outputs/autodrafted_replies.json
```

Because Task 5 did not produce valid results after the API quota was exhausted, Task 6 could not be completed with fresh API-generated outputs during the final run.

---

## 10. Task 7 — Two-Turn Conversation

Task 7 demonstrates context-aware multi-turn interaction.

The conversation begins with a customer describing:

* Positive feedback about soft fabric.
* Negative feedback about a tight waist fit.

The first turn asks the model to identify the positive and negative points.

The second turn asks the model to generate a professional customer-service response using the previous conversation history.

Conversation messages are stored as structured objects containing:

```text
role
content
```

The complete conversation is designed to be saved to:

```text
outputs/multiturn_conversation.json
```

API execution was disabled after the Gemini quota was exhausted, but the complete multi-turn conversation implementation remains in `analysis.py`.

---

## 11. API and Environment Variables

The project uses the Google Gemini API.

The API key must **not** be stored directly in source code.

Create a local `.env` file containing:

```env
GEMINI_API_KEY=your_api_key_here
```

The `.env` file must remain local and must not be committed to GitHub.

The repository's `.gitignore` should include:

```text
.env
__pycache__/
*.pyc
```

---

## 12. Installation

Create and activate a Python virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

Create the `.env` file and add the required Gemini API key.

---

## 13. Running the Project

From the Part 3 project directory, run:

```bash
python analysis.py
```

The main analysis script loads and cleans the dataset and contains the implementations for Tasks 1–7.

### API execution setting

Because the Gemini free-tier quota was exhausted during experimentation, API-dependent Tasks 5–7 are currently disabled using:

```python
RUN_API_TASKS = False
```

When API quota is available, this can be changed to:

```python
RUN_API_TASKS = True
```

to execute the API-dependent tasks.

The prompt experimentation notebook can also be opened and executed when API access is available:

```text
prompt_experiments.ipynb
```

---

## 14. Dependencies

All project dependencies are pinned in:

```text
requirements.txt
```

The environment should be installed using:

```bash
pip install -r requirements.txt
```

This ensures reproducibility across environments.

---

## 15. Design Decisions

### Consistent JSON Schema

The three main sentiment prompts use the same output schema so that responses can be parsed and compared programmatically.

### Explicit Constraints

The prompts explicitly define:

* Required output format.
* Allowed sentiment labels.
* Allowed confidence levels.
* Required explanation field.

This reduces ambiguity in model responses.

### Few-Shot Examples

Five examples are included in the few-shot template to provide the model with concrete demonstrations of the expected classification behaviour.

### Separate Aspect Prompt

Aspect analysis is implemented separately from overall sentiment classification because product quality and fit/comfort provide more actionable customer insights than a single overall sentiment label.

### API Execution Flag

API-dependent tasks are controlled through:

```python
RUN_API_TASKS
```

This prevents unnecessary API requests when the quota is unavailable while keeping the complete implementation available for future execution.

---

## 16. API Limitation

During experimentation, the Gemini free-tier API returned:

```text
429 RESOURCE_EXHAUSTED
```

because the available request quota had been exceeded.

Some earlier responses also returned malformed JSON or temporary `503 UNAVAILABLE` errors.

The project therefore distinguishes between:

* **Implemented functionality** — Tasks 1–7 are implemented in the source code.
* **Successful API execution** — limited by the available Gemini free-tier quota during the final experiment.

No API keys or other secrets are included in the repository.

---

## 17. Summary

This project demonstrates a complete LLM-based customer-review analysis workflow using Google Gemini.

It includes:

* Prompt engineering
* Zero-shot prompting
* Few-shot prompting
* Role prompting
* Structured JSON output
* Prompt experimentation
* Aspect-based sentiment analysis
* Automated customer-response generation
* Multi-turn conversation handling
* Environment-variable based API authentication
* Reproducible dependency management

The implementation is designed so that API-dependent tasks can be rerun when Gemini API quota is available.
