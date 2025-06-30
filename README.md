# Brand Score Analyzer

A Gradio web application that analyzes brand mentions and sentiment across multiple OpenAI models with real-time web search capabilities.

## Features

- **Multi-Model Analysis**: Queries multiple OpenAI models (GPT-4o, GPT-4.1-mini, O4-mini)
- **Web Search Integration**: Each model uses OpenAI's web search tool, location set to Melbourne, Australia
- **Brand Scoring**: Counts how many times a brand appears across all model responses
- **Brand Sentiment**: Calculates and displays the average sentiment (Negative, Neutral, Positive) across all model outputs, with a clear slider and summary
- **Visual Display**: Clean, modern UI with side-by-side model outputs and a grouped analytics panel
- **User-Friendly Interface**: Responsive Gradio interface with relevant examples and custom font for readability

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
   python app.py
   ```

## Usage

1. Enter a prompt/question in the text area
2. Enter the brand name you want to search for
3. Click "Analyze Brand Score"
4. View the results:
   - Brand Score slider and summary
   - Brand Sentiment slider and summary (Negative, Neutral, Positive)
   - Side-by-side breakdown of each model's response, with sentiment for each

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

## Design & UX

- Inputs and analytics are grouped for clarity and ease of use
- Brand Score, Brand Sentiment, and Brand Sentiment Summary are shown together in a single panel
- Custom font: `'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif`
- All model outputs are shown side by side for easy comparison

## Notes

- You need a valid OpenAI API key to use this application
- The app will create a shareable link when launched (if enabled)
- Only the Brand Sentiment Summary is shown (no bottom summary) 