# Brand Score Analyzer

A Gradio web application that analyzes brand mentions across multiple OpenAI models with real-time web search capabilities.

## Features

- **Multi-Model Analysis**: Queries multiple OpenAI models (GPT-4o, GPT-4.1-mini, O4-mini)
- **Web Search Integration**: Each model uses OpenAI's web search tool, location set to Melbourne, Australia
- **Brand Scoring**: Counts how many times a brand appears across all model responses
- **Visual Display**: Shows results with a slider (e.g., "2 out of 3") and detailed breakdown, side by side
- **User-Friendly Interface**: Clean Gradio interface with relevant examples

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key**:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key in `.env`:
     ```
     OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
     ```

3. **Run the Application**:
   ```bash
   python gradio_brand_score.py
   ```

## Usage

1. Enter a prompt/question in the text area
2. Enter the brand name you want to search for
3. Click "Analyze Brand Score"
4. View the results:
   - Slider showing score (e.g., 2 out of 3)
   - Text summary
   - Side-by-side breakdown of each model's response

## Example Prompts

All examples use the brand **AngryChair**:

- What are the best video production companies in Melbourne?
- Top-rated video production studios in Melbourne for corporate videos?
- Can you recommend high-quality video production agencies in Melbourne?
- List the most reputable video production companies in Melbourne.
- Who are the leading commercial video production companies in Melbourne?

## Configuration

The app uses the following models:
- GPT-4o (`gpt-4o`)
- GPT-4.1-mini (`gpt-4o-mini`)
- O4-mini (`o4-mini`)

## Environment Variables

- Only one variable is required in your `.env` file:
  ```
  OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```
- Do **not** commit your real `.env` file. Use `.env.example` for sharing variable requirements.

## Notes

- You need a valid OpenAI API key to use this application
- The app will create a shareable link when launched (if enabled)
- All model outputs are shown side by side for easy comparison 