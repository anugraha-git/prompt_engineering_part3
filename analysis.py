import json
import os
import pandas as pd

from llm_utils import call_llm
from prompts.prompt_templates import (
    ZERO_SHOT_PROMPT,
    FEW_SHOT_PROMPT,
    ROLE_PROMPT,
    ASPECT_PROMPT
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("data/Womens Clothing E-Commerce Reviews.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df = df.dropna(subset=["Review Text"])

print("Dataset shape:", df.shape)
print("Reviews available:", len(df))

# ============================================================
# PROMPT TEMPLATES
# ============================================================

templates = {
    "Zero-Shot": ZERO_SHOT_PROMPT,
    "Few-Shot": FEW_SHOT_PROMPT,
    "Role-Prompted": ROLE_PROMPT
}

print("\nPrompt templates loaded:")

for name in templates:
    print("-", name)


# ============================================================
# JSON SCHEMA VALIDATION
# ============================================================

REQUIRED_FIELDS = {
    "label",
    "confidence",
    "reason"
}


def validate_schema(response):
    """
    Checks whether the LLM response is valid JSON
    and contains all required fields.
    """

    if response is None:
        return False, None

    try:
        parsed = json.loads(response)

        if not isinstance(parsed, dict):
            return False, None

        if not REQUIRED_FIELDS.issubset(parsed.keys()):
            return False, parsed

        return True, parsed

    except json.JSONDecodeError:
        return False, None


# ============================================================
# EXISTING 15-CALL EXPERIMENT RESULTS
# ============================================================

comparison_results = []

validity_results = {
    "Zero-Shot": [True, True, False, True, True],
    "Few-Shot": [True, False, False, False, False],
    "Role-Prompted": [True, False, False, False, False]
}

reviews = df["Review Text"].head(5).tolist()

for template_name, validity_list in validity_results.items():

    for record_number, is_valid in enumerate(
        validity_list,
        start=1
    ):

        comparison_results.append({
            "template": template_name,
            "record": record_number,
            "review": reviews[record_number - 1],
            "valid_schema": is_valid
        })


# ============================================================
# CALCULATE RELIABILITY
# ============================================================

reliability = {}

for template_name in validity_results:

    results = validity_results[template_name]

    valid_count = sum(results)
    total_count = len(results)

    reliability[template_name] = {
        "valid": valid_count,
        "total": total_count,
        "percentage": (valid_count / total_count) * 100
    }


print("\n" + "=" * 60)
print("15-CALL PROMPT COMPARISON")
print("=" * 60)

for template_name, stats in reliability.items():

    print(
        f"{template_name}: "
        f"{stats['valid']}/{stats['total']} "
        f"({stats['percentage']:.1f}%)"
    )


# ============================================================
# SELECT BEST-PERFORMING TEMPLATE
# ============================================================

best_template_name = "Zero-Shot"

print(
    f"\nBest-performing template: {best_template_name}"
)


# ============================================================
# SAVE COMPARISON RESULTS
# ============================================================

os.makedirs("outputs", exist_ok=True)

with open(
    "outputs/prompt_comparison_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "reliability": reliability,
            "best_template": best_template_name,
            "results": comparison_results
        },
        f,
        indent=4,
        ensure_ascii=False
    )

print(
    "\nSaved: outputs/prompt_comparison_results.json"
)
# API quota currently exhausted.
# Task 5 will be executed after the Gemini quota resets.

RUN_API_TASKS = False



if RUN_API_TASKS:

    # ============================================================
    # TASK 5 — ASPECT-BASED SENTIMENT ANALYSIS
    # ============================================================

    aspect_reviews = df["Review Text"].head(10).tolist()

    aspect_results = []

    print("\n" + "=" * 60)
    print("10-RECORD ASPECT ANALYSIS")
    print("=" * 60)


    for record_number, review in enumerate(
        aspect_reviews,
        start=1
    ):

        print(
            f"\nRecord {record_number}"
        )

        prompt = ASPECT_PROMPT.format(
            review=review
        )

        response = call_llm(
            prompt,
            temperature=0.2,
            max_tokens=500
        )

        try:

            parsed = json.loads(response)

            aspect_results.append({
                "record": record_number,
                "review": review,
                "response": parsed,
                "valid_json": True
            })

            print("Status: Valid JSON")

        except json.JSONDecodeError:

            aspect_results.append({
                "record": record_number,
                "review": review,
                "response": None,
                "valid_json": False
            })

            print("Status: Invalid JSON")


    # ============================================================
    # SAVE ASPECT RESULTS
    # ============================================================

    with open(
        "outputs/aspect_analysis_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "best_template": best_template_name,
                "results": aspect_results
            },
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\nSaved: outputs/aspect_analysis_results.json"
    )

    # ============================================================
    # TASK 6 — AUTO-DRAFTED CUSTOMER RESPONSES
    # ============================================================

    RESPONSE_DRAFT_PROMPT = """
    Act as a professional customer-service representative
    for a women's clothing e-commerce company.

    Using the structured customer insight below, draft a short,
    professional, empathetic reply to the customer.

    Requirements:
    - Address the specific points raised by the customer.
    - Acknowledge positive points where appropriate.
    - Address negative points where appropriate.
    - Do not give generic responses.
    - Do not invent refunds, policies, or actions.
    - Keep the response concise.

    Structured customer insight:
    {structured_output}

    Original customer review:
    {review}

    Respond only with the customer-service reply.
    """


    draft_results = []

    print("\n" + "=" * 60)
    print("AUTO-DRAFTED CUSTOMER RESPONSES")
    print("=" * 60)


    # Generate replies for the first 3 valid aspect-analysis results

    valid_aspect_results = [
        item
        for item in aspect_results
        if item["valid_json"] and item["response"] is not None
    ]


    for item in valid_aspect_results[:3]:

        structured_output = json.dumps(
            item["response"],
            ensure_ascii=False
        )

        prompt = RESPONSE_DRAFT_PROMPT.format(
            structured_output=structured_output,
            review=item["review"]
        )

        draft = call_llm(
            prompt,
            temperature=0.3,
            max_tokens=300
        )

        draft_results.append({
            "record": item["record"],
            "review": item["review"],
            "structured_insight": item["response"],
            "drafted_reply": draft
        })

        print(f"\nRecord {item['record']}")
        print("Drafted reply:")
        print(draft)


    # ============================================================
    # SAVE DRAFTED RESPONSES
    # ============================================================

    with open(
        "outputs/autodrafted_replies.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            draft_results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\nSaved: outputs/autodrafted_replies.json"
    )

    # ============================================================
    # TASK 7 — TWO-TURN CONVERSATION
    # ============================================================

    print("\n" + "=" * 60)
    print("TWO-TURN CONVERSATION DEMONSTRATION")
    print("=" * 60)


    # Conversation history starts with the first user message

    conversation_history = [
        {
            "role": "user",
            "content": (
                "I purchased a dress and I really like the soft "
                "fabric, but the fit is too tight around the waist."
            )
        }
    ]


    # ============================================================
    # TURN 1
    # ============================================================

    turn1_prompt = """
    Read the customer's message below.

    Identify the main positive and negative points
    mentioned by the customer.

    Customer:
    I purchased a dress and I really like the soft fabric,
    but the fit is too tight around the waist.
    """

    turn1_response = call_llm(
        turn1_prompt,
        temperature=0.2,
        max_tokens=300
    )


    conversation_history.append({
        "role": "assistant",
        "content": turn1_response
    })


    # ============================================================
    # TURN 2
    # ============================================================

    conversation_history.append({
        "role": "user",
        "content": (
            "Now draft a short, professional and empathetic "
            "customer-service reply that specifically addresses "
            "both points from the previous exchange."
        )
    })


    history_text = "\n\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in conversation_history
    )


    turn2_prompt = f"""
    Use the following conversation history to answer
    the latest customer request.

    Conversation history:

    {history_text}

    Respond only with the customer-service reply.
    """


    turn2_response = call_llm(
        turn2_prompt,
        temperature=0.3,
        max_tokens=300
    )


    conversation_history.append({
        "role": "assistant",
        "content": turn2_response
    })


    # ============================================================
    # DISPLAY CONVERSATION HISTORY
    # ============================================================

    print("\nConversation history object:")

    print(
        json.dumps(
            conversation_history,
            indent=4,
            ensure_ascii=False
        )
    )


    print("\nSecond-turn response:")
    print(turn2_response)


    # ============================================================
    # SAVE CONVERSATION
    # ============================================================

    with open(
        "outputs/multiturn_conversation.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "conversation_history": conversation_history,
                "second_turn_response": turn2_response
            },
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\nSaved: outputs/multiturn_conversation.json"
    )

# ============================================================
# ANALYSIS COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("PART 3 ANALYSIS SCRIPT READY")
print("=" * 60)

print("15-call comparison experiment completed.")
print("Tasks 5-7 are implemented but API execution is currently disabled.")
print("Gemini API quota was exhausted during experimentation.")
print("No API calls are made by this final status section.")