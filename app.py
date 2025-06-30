import gradio as gr
import openai
import re
import os
from dotenv import load_dotenv
import gradio

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Model Configuration
MODELS = {
    "GPT-4o": "gpt-4o",
    "GPT-4.1-mini": "gpt-4o-mini",
    "O4-mini": "o4-mini"
}

# App Configuration
MAX_TOKENS = 500
TEMPERATURE = 0.7

def query_model_with_search(model_name: str, prompt: str, brand_name: str) -> dict:
    """
    Query a specific model with the new OpenAI web_search_preview tool and check for brand mentions.
    Uses the responses endpoint and sets user_location to Melbourne, Australia.
    """
    try:
        response = openai.responses.create(
            model=MODELS[model_name],
            tools=[{
                "type": "web_search_preview",
                "search_context_size": "low",
                "user_location": {
                    "type": "approximate",
                    "country": "AU",
                    "city": "Melbourne",
                    "region": "Victoria"
                }
            }],
            input=prompt,
        )
        response_text = response.output_text
        brand_mentioned = bool(re.search(rf'\b{re.escape(brand_name)}\b', response_text, re.IGNORECASE))
        return {
            "model": model_name,
            "response": response_text,
            "brand_mentioned": brand_mentioned,
            "success": True
        }
    except Exception as e:
        return {
            "model": model_name,
            "response": f"Error: {str(e)}",
            "brand_mentioned": False,
            "success": False
        }

def analyze_brand_score(prompt: str, brand_name: str) -> dict:
    if not prompt.strip() or not brand_name.strip():
        return {
            "score": 0,
            "total_models": 3,
            "mentions": 0,
            "results": [],
            "error": "Please provide both prompt and brand name."
        }
    results = []
    mentions = 0
    for model_name in MODELS.keys():
        result = query_model_with_search(model_name, prompt, brand_name)
        results.append(result)
        if result["brand_mentioned"]:
            mentions += 1
    return {
        "score": mentions,
        "total_models": len(MODELS),
        "mentions": mentions,
        "results": results,
        "error": None
    }

def create_detailed_output(analysis_result: dict) -> str:
    if analysis_result["error"]:
        return analysis_result["error"]
    output = f"**Brand Score: {analysis_result['mentions']} out of {analysis_result['total_models']}**\n\n"
    output += "**Detailed Results:**\n\n"
    for result in analysis_result["results"]:
        status = "✅" if result["brand_mentioned"] else "❌"
        output += f"**{result['model']}** {status}\n"
        output += f"Response: {result['response']}\n\n"
    return output

def sentiment_to_score(sentiment):
    sentiment = sentiment.lower()
    if "positive" in sentiment:
        return 1
    elif "negative" in sentiment:
        return -1
    else:
        return 0

def get_sentiment(text, brand_name, brand_mentioned):
    if not brand_mentioned:
        return "Not Mentioned"
    try:
        sentiment_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis assistant."},
                {"role": "user", "content": f"Classify the sentiment (Positive, Negative, Neutral) of the following text specifically about the brand '{brand_name}'. If the brand is not mentioned, respond with 'Not Mentioned'.\n\n{text}"}
            ],
            max_tokens=10,
            temperature=0
        )
        return sentiment_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def process_brand_analysis(prompt: str, brand_name: str):
    analysis = analyze_brand_score(prompt, brand_name)
    outputs = []
    sentiments = []
    responses = []
    for result in analysis["results"]:
        status = "✅" if result["brand_mentioned"] else "❌"
        sentiment = get_sentiment(result["response"], brand_name, result["brand_mentioned"])
        sentiments.append(sentiment)
        responses.append(result["response"])
        outputs.append(
            f"**{result['model']}** {status}\n\n**Sentiment:** {sentiment}\n\n{result['response']}"
        )
    # Calculate overall sentiment score
    sentiment_scores = [sentiment_to_score(s) for s in sentiments]
    if sentiment_scores:
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    else:
        avg_sentiment = 0
    # Summarize brand sentiment
    brand_sentiment_summary = summarize_perception(sentiments)
    # Return slider, summary, sentiment slider, sentiment summary, all model outputs
    return (
        analysis["score"],
        f"{analysis['mentions']} out of {analysis['total_models']}",
        avg_sentiment,
        brand_sentiment_summary,
        *outputs
    )

def summarize_perception(responses):
    try:
        joined = "\n\n".join(responses)
        summary_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a brand perception analyst."},
                {"role": "user", "content": f"Based on the following outputs, summarize how the brand is perceived overall:\n\n{joined}"}
            ],
            max_tokens=60,
            temperature=0.3
        )
        return summary_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

custom_css = """
body, .gradio-container, .gr-block, .gr-group, .gr-box, .gr-markdown, .gr-textbox, .gr-slider, .gr-button, .gr-row, .gr-column {
    font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
}
"""

with gr.Blocks(title="Brand Score Analyzer", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🏆 Brand Score Analyzer")
    gr.Markdown("Enter a prompt and brand name to analyze how often the brand appears across multiple AI models with real-time web search capabilities. Results are optimized for Melbourne, Australia location.")
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your question or prompt here...",
                lines=10
            )
            brand_input = gr.Textbox(
                label="Brand Name",
                placeholder="Enter the brand name to search for...",
                lines=1
            )
            analyze_btn = gr.Button("Analyze Brand Score", variant="primary", size="lg")
        with gr.Column(scale=1):
            with gr.Group():
                score_text = gr.Textbox(
                    value="0 out of 2",
                    label="Score Summary",
                    interactive=False
                )
                score_slider = gr.Slider(
                    minimum=0,
                    maximum=3,
                    value=0,
                    step=1,
                    label="Brand Score",
                    interactive=False
                )
                sentiment_slider = gr.Slider(
                    minimum=-1,
                    maximum=1,
                    value=0,
                    step=0.01,
                    label="Brand Sentiment",
                    interactive=False,
                    render=True,
                    show_label=True,
                    info="Negative (-1), Neutral (0), Positive (1)",
                )
                brand_sentiment_summary_box = gr.Textbox(
                    value="Brand sentiment summary will appear here.",
                    label="Brand Sentiment Summary",
                    interactive=False
                )
    # Side-by-side model outputs
    gr.Markdown("## OpenAI models (ChatGPT) responses")
    with gr.Row() as model_outputs_row:
        model_output_blocks = []
        for model_name in MODELS.keys():
            model_output_blocks.append(
                gr.Markdown(
                    value=f"Waiting for {model_name}...",
                    label=f"{model_name} Output"
                )
            )
    analyze_btn.click(
        fn=process_brand_analysis,
        inputs=[prompt_input, brand_input],
        outputs=[score_slider, score_text, sentiment_slider, brand_sentiment_summary_box, *model_output_blocks]
    )
    gr.Examples(
        examples=[
            ["What are the best video production companies in Melbourne?", "AngryChair"],
            ["Top-rated video production studios in Melbourne for corporate videos?", "AngryChair"],
            ["Can you recommend high-quality video production agencies in Melbourne?", "AngryChair"],
            ["List the most reputable video production companies in Melbourne.", "AngryChair"],
            ["Who are the leading commercial video production companies in Melbourne?", "AngryChair"]
        ],
        inputs=[prompt_input, brand_input]
    )

if __name__ == "__main__":
    demo.launch()

print(gradio.__file__)