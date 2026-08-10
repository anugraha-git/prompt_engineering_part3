ZERO_SHOT_PROMPT = """
You are given a customer clothing review.

Classify the overall sentiment of the review.

Respond ONLY in the following JSON format:

{{
  "label": "Positive | Negative | Mixed",
  "confidence": "High | Medium | Low",
  "reason": "Brief explanation"
}}

Review:
{review}
"""

FEW_SHOT_PROMPT = """
You are given customer clothing reviews.

Classify the overall sentiment of each review.

Respond ONLY in the following JSON format:

{{
  "label": "Positive | Negative | Mixed",
  "confidence": "High | Medium | Low",
  "reason": "Brief explanation"
}}

Example 1

Review:
"I absolutely love this dress. The fabric is soft and the fit is perfect."

Output:
{{
  "label": "Positive",
  "confidence": "High",
  "reason": "The customer expresses satisfaction with the dress quality and fit."
}}
Example 2

Review:
"The dress looks beautiful and the color is exactly as shown in the picture, but the fabric feels uncomfortable and the fitting is too tight."

Output:
{{
  "label": "Mixed",
  "confidence": "High",
  "reason": "The customer appreciates the appearance and color of the dress but is dissatisfied with the fabric and fit."
}}

Example 3

Review:
"I am not so satisfied with the model nor the fabric as it doesn't look like the picture on your website. Not a bad product, but it did not meet my expectations."

Output:
{{
  "label": "Negative",
  "confidence": "High",
  "reason": "The customer is dissatisfied with the design, fabric, and difference between the product received and the website image."
}}
Example 4

Review:
"I am really satisfied with the dress. The fabric feels so soft as mentioned in the description. Worth the price."

Output:
{{
  "label": "Positive",
  "confidence": "High",
  "reason": "The customer appreciates the soft fabric, accurate product description, and feels that the dress provides good value for the price."
}}
Example 5

Review:
"The dress looks beautiful and the color is exactly what I expected. However, the fabric feels cheap and the stitching started coming loose after wearing it once."

Output:
{{
  "label": "Negative",
  "confidence": "High",
  "reason": "The customer likes the appearance of the dress but is dissatisfied with the poor fabric quality and stitching issues."
}}
Now classify the following customer review using the examples above.

Review:
{review}
"""

ROLE_PROMPT = """
Act as a senior customer-insights analyst specializing in fashion e-commerce reviews.


Instruction:
Classify the overall sentiment of the customer clothing review.


Context:
The review dataset contains customer opinions about clothing products, including fabric quality, fit, design, comfort, and overall satisfaction.


Constraints:
- Analyze the customer's overall opinion.
- Consider both positive and negative statements.
- Respond ONLY in the following JSON format.


Output:
{{
  "label": "Positive | Negative | Mixed",
  "confidence": "High | Medium | Low",
  "reason": "Brief explanation"
}}


Review:
{review}
"""

ASPECT_PROMPT = """
Act as a senior customer-insights analyst specializing in customer clothing reviews.

Instruction:
Analyze the sentiment of the following clothing review for two aspects:
1. Product Quality
2. Fit & Comfort

Context:
The review is written by a customer about a clothing product.
Use only information that is actually present in the review.
If an aspect is not mentioned, use "Not Mentioned" as its sentiment.

Constraints:
- Product Quality sentiment must be Positive, Negative, Mixed, or Not Mentioned.
- Fit & Comfort sentiment must be Positive, Negative, Mixed, or Not Mentioned.
- Each actionable phrase must contain 3 to 6 words.
- The actionable phrase should briefly describe what the customer liked or disliked.
- Respond ONLY with the JSON object below.
- Do not add markdown or any text outside the JSON.

Output:
{{
  "product_quality": {{
    "sentiment": "Positive | Negative | Mixed | Not Mentioned",
    "actionable_phrase": "3 to 6 words"
  }},
  "fit_comfort": {{
    "sentiment": "Positive | Negative | Mixed | Not Mentioned",
    "actionable_phrase": "3 to 6 words"
  }}
}}

Review:
{review}
"""
print("ASPECT PROMPT LOADED")