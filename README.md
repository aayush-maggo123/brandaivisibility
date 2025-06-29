# Brand Score Analyzer

A Gradio web application that analyzes brand mentions across multiple OpenAI models with search capabilities.

## Features

- **Multi-Model Analysis**: Queries 4 different OpenAI models (GPT-4o, GPT-4.1, GPT-4.1-mini, GPT-3.5)
- **Search Integration**: Each model has access to web search capabilities
- **Brand Scoring**: Counts how many times a brand appears across all model responses
- **Visual Display**: Shows results with a slider (e.g., "3 out of 4") and detailed breakdown
- **User-Friendly Interface**: Clean Gradio interface with examples

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys**:
   - Edit `config.py` and add your OpenAI API key
   - Optionally add a search API key for web search functionality

3. **Run the Application**:
   ```bash
   python gradio_brand_score.py
   ```

## Usage

1. Enter a prompt/question in the text area
2. Enter the brand name you want to search for
3. Click "Analyze Brand Score"
4. View the results:
   - Slider showing score (e.g., 3 out of 4)
   - Text summary
   - Detailed breakdown of each model's response

## Example Prompts

- "What are the best smartphone brands in 2024?" → Brand: "Apple"
- "Which companies are leading in electric vehicles?" → Brand: "Tesla"
- "What are the top coffee brands worldwide?" → Brand: "Starbucks"

## Configuration

The app uses the following models:
- GPT-4o (`gpt-4o`)
- GPT-4.1 (`gpt-4-turbo`)
- GPT-4.1-mini (`gpt-4o-mini`)
- GPT-3.5 (`gpt-3.5-turbo`)

## Notes

- You need a valid OpenAI API key to use this application
- The search functionality is currently simulated - you can integrate with real search APIs like SerpAPI or Google Custom Search
- The app will create a shareable link when launched 