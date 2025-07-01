import gradio as gr
import openai
import re
import os
from dotenv import load_dotenv
import pandas as pd
import concurrent.futures

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

def generate_prompts_for_keyword(keyword: str) -> list[str]:
    """
    Generates a list of diverse search queries for a given keyword using an LLM.
    """
    if not keyword.strip():
        return []
    try:
        system_prompt = """You are an expert in generating search queries. Your task is to create 5 diverse, high-quality search queries based on a user-provided keyword.

The queries should be designed to find the best companies or service providers related to the keyword. They should be phrased as natural questions or search terms that a real user would type into a search engine.

Return the queries as a numbered list, with each query on a new line. Do not include any other text, preamble, or explanation.

Example Input:
video production melbourne

Example Output:
1. What are the best video production companies in Melbourne?
2. Top-rated video production studios in Melbourne for corporate videos?
3. Can you recommend high-quality video production agencies in Melbourne?
4. List the most reputable video production companies in Melbourne.
5. Who are the leading commercial video production companies in Melbourne?
"""
        user_prompt = f"Generate 5 search queries for the keyword: '{keyword}'"

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.5,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content.strip()
        # Use regex to find all lines that start with a number and a dot.
        prompts = re.findall(r"^\d+\.\s*(.*)", content, re.MULTILINE)
        return [p.strip() for p in prompts if p.strip()]
    except Exception as e:
        print(f"Error generating prompts: {e}")
        return []


def query_model_with_search(model_name: str, prompt: str, brand_name: str) -> dict:
    """
    Query a specific model with the new OpenAI web_search_preview tool and check for brand mentions.
    Uses the responses endpoint and sets user_location to Australia.
    """
    try:
        response = openai.responses.create(
            model=MODELS[model_name],
            tools=[{
                "type": "web_search_preview",
                "search_context_size": "low",
                "user_location": {
                    "type": "approximate",
                    "country": "AU"
                }
            }],
            input=prompt,
        )
        response_text = response.output_text
        brand_mentioned = bool(re.search(rf'\b{re.escape(brand_name)}\b', response_text, re.IGNORECASE))
        return {
            "model": model_name,
            "prompt": prompt,
            "response": response_text,
            "brand_mentioned": brand_mentioned,
            "success": True
        }
    except Exception as e:
        return {
            "model": model_name,
            "prompt": prompt,
            "response": f"Error: {str(e)}",
            "brand_mentioned": False,
            "success": False
        }

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

def run_full_analysis(keyword: str, brand_name: str):
    if not brand_name.strip() or not keyword.strip():
        return gr.Slider(maximum=3, value=0), "0 out of 0", 0, "Please provide both a keyword and a brand name.", pd.DataFrame(), "Results will appear here."

    prompts = generate_prompts_for_keyword(keyword)
    if not prompts:
        return gr.Slider(maximum=3, value=0), "0 out of 0", 0, "Could not generate prompts for the given keyword.", pd.DataFrame(), "Results will appear here."

    all_results = []
    total_mentions = 0
    total_models_queried = len(prompts) * len(MODELS)
    summary_table_data = {prompt: {"Prompt": prompt, "Brand Name": brand_name, "Score": 0} for prompt in prompts}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_result = {executor.submit(query_model_with_search, model_name, prompt, brand_name): (prompt, model_name) for prompt in prompts for model_name in MODELS.keys()}
        
        for future in concurrent.futures.as_completed(future_to_result):
            prompt, model_name = future_to_result[future]
            try:
                result = future.result()
                all_results.append(result)
                summary_table_data[prompt][model_name] = "✅" if result["brand_mentioned"] else "❌"
                if result["brand_mentioned"]:
                    summary_table_data[prompt]["Score"] += 1
                    total_mentions += 1
            except Exception as exc:
                print(f'{prompt} generated an exception: {exc}')
                summary_table_data[prompt][model_name] = "Error"

    # Finalize summary table
    for prompt in prompts:
        summary_table_data[prompt]["Score"] = f'{summary_table_data[prompt]["Score"]}/{len(MODELS)}'
    summary_df = pd.DataFrame(list(summary_table_data.values()), columns=["Prompt", "Brand Name"] + list(MODELS.keys()) + ["Score"])

    # Process sentiment for mentioned responses
    all_sentiments = []
    all_responses_for_summary = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_sentiment = {executor.submit(get_sentiment, result["response"], brand_name, result["brand_mentioned"]): result for result in all_results if result["brand_mentioned"]}
        for future in concurrent.futures.as_completed(future_to_sentiment):
            result = future_to_sentiment[future]
            try:
                sentiment = future.result()
                all_sentiments.append(sentiment)
                all_responses_for_summary.append(result["response"])
            except Exception as exc:
                print(f'Sentiment analysis generated an exception: {exc}')

    # Calculate overall sentiment
    sentiment_scores = [sentiment_to_score(s) for s in all_sentiments]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

    # Summarize overall perception
    brand_sentiment_summary = "Not enough data for a summary."
    if all_responses_for_summary:
        brand_sentiment_summary = summarize_perception(all_responses_for_summary, brand_name)

    score_summary_text = f"{total_mentions} out of {total_models_queried}"
    score_slider_update = gr.Slider(maximum=total_models_queried, value=total_mentions)

    # Generate recommendations
    recommendations = generate_recommendations(brand_name, all_results, total_mentions, total_models_queried)

    return score_slider_update, score_summary_text, avg_sentiment, brand_sentiment_summary, summary_df, recommendations

def summarize_perception(responses, brand_name):
    try:
        joined = "\n\n".join(responses)
        summary_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a brand perception analyst."},
                {"role": "user", "content": f"Based on the following outputs, summarize how the brand '{brand_name}' is perceived overall:\n\n{joined}"}
            ],
            max_tokens=150,
            temperature=0.3
        )
        return summary_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def generate_recommendations(brand_name, results, score, total):
    try:
        # Basic analysis of where the brand was missed
        missed_prompts = []
        prompt_results = {}
        for r in results:
            if r["prompt"] not in prompt_results:
                prompt_results[r["prompt"]] = []
            prompt_results[r["prompt"]].append(r)

        for prompt, p_results in prompt_results.items():
            if not any(r["brand_mentioned"] for r in p_results):
                missed_prompts.append(prompt)

        # Basic competitor analysis (extracting other mentioned brands)
        competitors = set()
        for r in results:
            found_brands = re.findall(r'\b[A-Z][a-zA-Z-]+\b', r["response"])
            for b in found_brands:
                if b.lower() != brand_name.lower():
                    competitors.add(b)

        system_prompt = """You are a strategic marketing consultant specializing in AI and search engine optimization.
        Based on the user's brand score, provide actionable recommendations to improve their visibility in AI chat models.
        Structure your response in clear, easy-to-understand sections.
        
        Your response MUST include the following markdown headers:
        ### Overview
        ### Analysis of Missed Prompts
        ### Competitors Identified
        ### Strategic Recommendations

        - Under "Overview", provide a brief, encouraging summary of the brand's performance.
        - Under "Analysis of Missed Prompts", analyze why the brand might have been missed for the provided prompts.
        - Under "Competitors Identified", list the key competitors that were identified in the search results.
        - Under "Strategic Recommendations", provide a list of 3-5 concrete, strategic recommendations.
        - Keep the tone professional, insightful, and helpful.
        """

        user_prompt = f"**Brand:** {brand_name}\n        **Overall Score:** {score}/{total}\n        \n        **Prompts where the brand was NOT mentioned:**\n        - {"\n- ".join(missed_prompts) if missed_prompts else "None"}\n        \n        **Competitors mentioned in the results:**\n        - {", ".join(list(competitors)) if competitors else "None identified"}\n        \n        Please provide your analysis and strategic recommendations based on this data, following the required structure.\n        "

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=700,
            temperature=0.6
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error generating recommendations: {str(e)}"


custom_css = """
body, .gradio-container, .gr-block, .gr-group, .gr-box, .gr-markdown, .gr-textbox, .gr-slider, .gr-button, .gr-row, .gr-column {
    font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
}
"""

with gr.Blocks(title="Brand Score Analyzer", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🏆 Brand Score Analyzer")
    gr.Markdown("Enter a keyword and a brand name to analyze how often the brand appears across multiple AI models for a set of generated search queries. Results are optimized for Australia.")
    
    with gr.Row():
        with gr.Column(scale=2):
            keyword_input = gr.Textbox(
                label="Keyword",
                placeholder="e.g., video production melbourne, financial advisor sydney",
                lines=1,
                value="video production melbourne"
            )
            brand_input = gr.Textbox(
                label="Brand Name",
                placeholder="Enter the brand name to search for...",
                lines=1,
                value="AngryChair"
            )
            analyze_btn = gr.Button("Analyze Brand Score", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            with gr.Group():
                score_text = gr.Textbox(
                    value="0 out of 15",
                    label="Overall Score Summary",
                    interactive=False
                )
                score_slider = gr.Slider(
                    minimum=0,
                    maximum=15, # Initial max, will be updated
                    value=0,
                    step=1,
                    label="Total Brand Mentions",
                    interactive=False
                )
                sentiment_slider = gr.Slider(
                    minimum=-1,
                    maximum=1,
                    value=0,
                    step=0.01,
                    label="Average Brand Sentiment",
                    interactive=False,
                    info="Negative (-1), Neutral (0), Positive (1)",
                )
                brand_sentiment_summary_box = gr.Textbox(
                    value="Brand sentiment summary will appear here.",
                    label="Brand Sentiment Summary",
                    interactive=False,
                    lines=4
                )

    gr.Markdown("## Summary Table")
    summary_df = gr.DataFrame(headers=["Prompt", "Brand Name"] + list(MODELS.keys()) + ["Score"], interactive=False)

    gr.Markdown("## Strategic Recommendations")
    recommendations_output = gr.Markdown("Recommendations will appear here.")

    analyze_btn.click(
        fn=run_full_analysis,
        inputs=[keyword_input, brand_input],
        outputs=[score_slider, score_text, sentiment_slider, brand_sentiment_summary_box, summary_df, recommendations_output]
    )

if __name__ == "__main__":
    demo.launch()
