# Brand Score Analyzer

A dynamic and high-speed Gradio web application that analyzes brand visibility, sentiment, and competitive positioning across multiple AI models using real-time, location-aware web searches.

## Key Features

- **Dynamic Prompt Generation**: Enter any keyword (e.g., "video production melbourne", "best cafes in sydney") to automatically generate a diverse set of relevant search queries.
- **Concurrent API Requests**: All API calls to the AI models are made in parallel, delivering analysis results significantly faster than sequential requests.
- **Multi-Model Analysis**: Queries multiple leading OpenAI models (GPT-4o, GPT-4.1-mini, O4-mini) to provide a comprehensive view of your brand's visibility.
- **Location-Aware Search**: All web searches are optimized for Australia to provide locally relevant results.
- **Comprehensive Scoring**: Calculates a brand score based on how many times your brand is mentioned across all model responses.
- **Brand Sentiment Analysis**: Analyzes the sentiment (Positive, Negative, or Neutral) of the text specifically where your brand is mentioned.
- **Actionable Insights**:
  - **Summary Table**: A clear, concise table showing which models mentioned your brand for each generated prompt.
  - **Strategic Recommendations**: Automatically generates a report with an overview, analysis of missed opportunities, a list of identified competitors, and actionable recommendations to improve your brand's AI visibility.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure API Key**:
    -   Copy `.env.example` to `.env`.
    -   Add your OpenAI API key to the `.env` file:
        ```
        OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        ```

3.  **Run the Application**:
    ```bash
    python app.py
    ```

## Usage

1.  **Enter a Keyword**: Type a keyword or phrase that represents your industry or service (e.g., `financial advisor perth`).
2.  **Enter a Brand Name**: Provide the name of the brand you want to analyze.
3.  **Click "Analyze Brand Score"**.
4.  **Review the Results**:
    -   **Overall Score**: See the a total number of brand mentions in the top-right panel.
    -   **Average Sentiment**: View the average sentiment score for your brand's mentions.
    -   **Summary Table**: Quickly see which prompts and models featured your brand.
    -   **Strategic Recommendations**: Read the detailed analysis and actionable advice at the bottom to improve your score.

## Configuration

The app is configured to use the following models:
-   GPT-4o (`gpt-4o`)
-   GPT-4.1-mini (`gpt-4o-mini`)
-   O4-mini (`o4-mini`)

All API requests are sent with `user_location` set to Australia (`AU`).