# Playground for chatbots
### Project to work with LangChain
This project lets you play around with different settings in embedding models and LLMs to find something that works. Feed in your pdf data and using an LLM you are able to interact with that data.

## Openai
This project currently works with [Openai](https://platform.openai.com/docs/introduction) embedding models and LLMs. This requires that you have an Openai account and an Openai [API key](https://platform.openai.com/docs/quickstart/step-2-setup-your-api-key). Place your Openai key in a `.env` file at the root directory of this project with the following format, `OPENAI_API_KEY=...`

## Python environment
It's recommended to use a python virtual environment like [venv](https://docs.python.org/3/library/venv.html) or [conda](https://conda.io/projects/conda/en/latest/index.html).

## Dependencies
Run `pip install -r requirements.txt`

## Run
Streamlit will run a simple server on your machine and open the tool in your default browser.
<br>`streamlit run app.py`