from flask import Flask, request, jsonify, render_template, g, send_file
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import io
import requests
from bs4 import BeautifulSoup
import re
import random
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from urllib.parse import urlparse, urljoin, quote_plus
from functools import wraps
import asyncio
import aiohttp
import chardet
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import HexColor, black
import csv
from datetime import date
import math

load_dotenv()

app = Flask(__name__)
CORS(app)

class Config:
    API_KEY = os.getenv("GEMINI_API_KEY")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
    SNIPPET_LENGTH = 5000
    DEEP_RESEARCH_SNIPPET_LENGTH = 10000
    MAX_TOKENS_PER_CHUNK = 100000
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (iPad; CPU OS 14_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1'
    ]
    SEARCH_ENGINES = ["google", "duckduckgo", "bing", "yahoo", "brave", "linkedin"]


config = Config()

logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
genai.configure(api_key=config.API_KEY)
conversation_history = []

# Rate limiting for deep research
deep_research_rate_limits = {
    "gemini-2.0-flash": {"requests_per_minute": 15, "last_request": 0, "lock": asyncio.Lock()},
    "gemini-2.0-flash-thinking-exp-01-21": {"requests_per_minute": 10, "last_request": 0, "lock": asyncio.Lock()}
}
DEFAULT_DEEP_RESEARCH_MODEL = "gemini-2.0-flash"


async def rate_limit_model(model_name):
    """Applies rate limiting to the specified Gemini model using a token bucket algorithm."""
    if model_name not in deep_research_rate_limits:
        return  # No rate limiting for this model

    rate_limit_data = deep_research_rate_limits[model_name]
    async with rate_limit_data["lock"]:  # Use asyncio.Lock
        now = time.time()
        time_since_last_request = now - rate_limit_data["last_request"]
        requests_per_minute = rate_limit_data["requests_per_minute"]
        wait_time = max(0, 60 / requests_per_minute - time_since_last_request)

        if wait_time > 0:
            logging.info(f"Rate limiting {model_name}, waiting for {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)

        rate_limit_data["last_request"] = time.time()


def get_random_user_agent():
    return random.choice(config.USER_AGENTS)

def process_base64_image(base64_string):
    try:
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        image_data = base64.b64decode(base64_string)
        image_stream = io.BytesIO(image_data)
        image = Image.open(image_stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        return {'mime_type': 'image/jpeg', 'data': img_byte_arr.getvalue()}
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None



async def async_get_shortened_url(session, url):
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            url = "http://" + url
        tinyurl_api = f"https://tinyurl.com/api-create.php?url={url}"
        async with session.get(tinyurl_api, timeout=5) as response:
            if response.status == 200:
                return await response.text()
            else:
                logging.warning(f"TinyURL API error {response.status} for URL: {url}")
                return url
    except Exception as e:
        logging.error(f"Error shortening URL '{url}': {e}")
        return url

def fix_url(url):
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            url = "https://" + url
            parsed = urlparse(url)
        if not parsed.netloc:
            return None
        return url
    except Exception:
        return None

async def scrape_search_engine(search_query, engine_name):
    if engine_name == "google":
        return await scrape_google(search_query)
    elif engine_name == "duckduckgo":
        return await scrape_duckduckgo(search_query)
    elif engine_name == "bing":
        return await scrape_bing(search_query)
    elif engine_name == "yahoo":
        return await scrape_yahoo(search_query)
    elif engine_name == "brave":
        return await scrape_brave(search_query)
    elif engine_name == "linkedin":
        return await scrape_linkedin(search_query)
    else:
        logging.warning(f"Unknown search engine: {engine_name}")
        return []

async def scrape_google(search_query):
	search_results = []
	google_url = f"https://www.google.com/search?q={search_query}"
	try:
		async with aiohttp.ClientSession() as session:
			async with session.get(google_url, headers={'User-Agent': get_random_user_agent()}, timeout=10) as google_resp:
				logging.info(f"Google Search Status Code: {google_resp.status} for query: {search_query}")
				if google_resp.status == 200:
					google_soup = BeautifulSoup(await google_resp.text(), 'html.parser')
					google_results = google_soup.find_all('div', class_='yuRUbf')
					if not google_results:
						logging.warning(f"No results found on Google for query: {search_query} - yuRUbf class not found")
						return []
					for res in google_results[:20]:
						a_tag = res.find('a')
						if a_tag:
							href = a_tag.get('href')
							if href:
								fixed_url = fix_url(href)
								if fixed_url:
									search_results.append(fixed_url)
	except Exception as e:
		logging.error(f"Error scraping Google for query: {search_query} - {e}")
	return search_results
async def scrape_duckduckgo(search_query):
    search_results = []
    duck_url = f"https://html.duckduckgo.com/html/?q={search_query}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(duck_url, headers={'User-Agent': get_random_user_agent()}, timeout=10) as duck_resp:
                logging.info(f"DuckDuckGo Search Status Code: {duck_resp.status} for query: {search_query}")
                if duck_resp.status == 200:
                    duck_soup = BeautifulSoup(await duck_resp.text(), 'html.parser')
                    duck_results = duck_soup.find_all('a', class_='result__a')
                    if not duck_results:
                        logging.warning(f"No results found on DuckDuckGo for query: {search_query} - result__a class not found")
                        return []
                    for a_tag in duck_results[:20]:
                        href = a_tag.get('href')
                        if href:
                            fixed_url = fix_url(href)
                            if fixed_url:
                                search_results.append(fixed_url)
    except Exception as e:
        logging.error(f"Error scraping DuckDuckGo for query: {search_query} - {e}")
    return search_results

async def scrape_bing(search_query):
    search_results = []
    bing_url = f"https://www.bing.com/search?q={search_query}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(bing_url, headers={'User-Agent': get_random_user_agent()}, timeout=10) as bing_resp:
                logging.info(f"Bing Search Status Code: {bing_resp.status} for query: {search_query}")
                if bing_resp.status == 200:
                    bing_soup = BeautifulSoup(await bing_resp.text(), 'html.parser')
                    bing_results = bing_soup.find_all('li', class_='b_algo')
                    if not bing_results:
                        logging.warning(f"No results found on Bing for query: {search_query} - b_algo class not found")
                        return []
                    for li in bing_results[:20]:
                        a_tag = li.find('a')
                        if a_tag:
                            href = a_tag.get('href')
                            if href:
                                fixed_url = fix_url(href)
                                if fixed_url:
                                    search_results.append(fixed_url)
    except Exception as e:
        logging.error(f"Error scraping Bing for query: {search_query} - {e}")
    return search_results

async def scrape_yahoo(search_query):
    search_results = []
    yahoo_url = f"https://search.yahoo.com/search?p={search_query}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(yahoo_url, headers={'User-Agent': get_random_user_agent()}, timeout=10) as yahoo_resp:
                logging.info(f"Yahoo Search Status Code: {yahoo_resp.status} for query: {search_query}")
                if yahoo_resp.status == 200:
                    yahoo_soup = BeautifulSoup(await yahoo_resp.text(), 'html.parser')
                    yahoo_results = yahoo_soup.find_all('div', class_='algo-sr')
                    if not yahoo_results:
                        logging.warning(f"No results found on Yahoo for query: {search_query} - algo-sr class not found")
                        return []
                    for res in yahoo_results[:20]:
                        a_tag = res.find('a')
                        if a_tag:
                            href = a_tag.get('href')
                            if href:
                                fixed_url = fix_url(href)
                                if fixed_url:
                                    search_results.append(fixed_url)
    except Exception as e:
        logging.error(f"Error scraping Yahoo for query: {search_query} - {e}")
    return search_results

async def scrape_brave(search_query):
    search_results = []
    brave_url = f"https://search.brave.com/search?q={search_query}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(brave_url, headers={'User-Agent': get_random_user_agent()}, timeout=10) as brave_resp:
                logging.info(f"Brave Search Status Code: {brave_resp.status} for query: {search_query}")
                if brave_resp.status == 200:
                    brave_soup = BeautifulSoup(await brave_resp.text(), 'html.parser')
                    brave_results = brave_soup.find_all('div', class_='web-results')
                    if not brave_results:
                        logging.warning(f"No results found on Brave for query: {search_query} - web-results class not found")
                        return []
                    for res in brave_results:
                        a_tag = res.find('a', class_='result-header-url')
                        if a_tag:
                            href = a_tag.get('href')
                            if href:
                                fixed_url = fix_url(href)
                                if fixed_url:
                                    search_results.append(fixed_url)
    except Exception as e:
        logging.error(f"Error scraping Brave for query: {search_query} - {e}")
    return search_results

async def scrape_linkedin(search_query):
    search_results = []
    linkedin_url = f"https://www.linkedin.com/search/results/all/?keywords={search_query.replace(' ', '%20')}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(linkedin_url, headers={'User-Agent': get_random_user_agent()}, timeout=10) as linkedin_resp:
                logging.info(f"LinkedIn Search Status Code: {linkedin_resp.status} for query: {search_query}")

                if linkedin_resp.status == 200:
                    linkedin_soup = BeautifulSoup(await linkedin_resp.text(), 'html.parser')

                    for result in linkedin_soup.find_all('div', class_='entity-result__item'):
                        try:
                            link_tag = result.find('a', class_='app-aware-link')
                            if not link_tag or not link_tag.get('href'):
                                continue
                            profile_url = fix_url(link_tag.get('href'))
                            if not profile_url or '/in/' not in profile_url:
                                continue

                            if " at " in search_query.lower():
                                context = search_query.lower().split(" at ")[1]
                                name = result.find('span', class_='entity-result__title-text')
                                title_company = result.find('div', class_='entity-result__primary-subtitle')

                                combined_text = ""
                                if name:
                                    combined_text += name.get_text(strip=True).lower() + " "
                                if title_company:
                                    combined_text += title_company.get_text(strip=True).lower()

                                if context not in combined_text:
                                    continue

                            search_results.append(profile_url)

                        except Exception as e:
                            logging.warning(f"Error processing a LinkedIn result: {e}")
                            continue

    except Exception as e:
        logging.error(f"Error scraping LinkedIn for query: {search_query} - {e}")

    return search_results

async def fetch_page_content(session, url, snippet_length=None, extract_links=False, extract_emails=False):
    if snippet_length is None:
        snippet_length = config.SNIPPET_LENGTH

    content_snippets = []
    references = []
    extracted_data = {}

    try:
        async with session.get(url, headers={'User-Agent': get_random_user_agent()}, timeout=60) as page_resp:
            logging.debug(f"Fetching page content status code: {page_resp.status} for URL: {url}")

            if page_resp.status == 200:
                raw_content = await page_resp.read()
                detected_encoding = chardet.detect(raw_content)['encoding']
                logging.debug(f"Detected encoding for {url}: {detected_encoding}")

                page_text = None
                try:
                    page_text = raw_content.decode(detected_encoding, errors='replace')
                except UnicodeDecodeError:
                    logging.warning(f"Decoding failed with {detected_encoding}. Trying UTF-8.")
                    try:
                        page_text = raw_content.decode('utf-8', errors='replace')
                    except UnicodeDecodeError:
                        logging.warning("Decoding failed with UTF-8. Trying latin-1")
                        page_text = raw_content.decode('latin-1', errors='replace')

                if page_text:
                    page_soup = BeautifulSoup(page_text, 'html.parser')
                    for script in page_soup(["script", "style"]):
                        script.decompose()
                    text = page_soup.get_text(separator=' ', strip=True)
                    text = re.sub(r'[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]', '', text)

                    snippet = text[:snippet_length]
                    title = page_soup.title.string if page_soup.title else url
                    formatted_snippet = f"### {title}\n\n{snippet}\n"
                    content_snippets.append(formatted_snippet)
                    references.append(url)

                    if extract_links:
                        extracted_data['links'] = [a['href'] for a in page_soup.find_all('a', href=True) if a['href'] and not a['href'].startswith("#")]
                    if extract_emails:
                        extracted_data['emails'] = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", page_text)

            else:
                logging.error(f"Failed to fetch content: URL={url}, Status={page_resp.status}")

    except aiohttp.ClientConnectorError as e:
        logging.error(f"Connection error fetching URL {url}: {e}")
        raise
    except aiohttp.InvalidURL as e:
        logging.error(f"Invalid URL {url}: {e}")
        raise
    except aiohttp.TooManyRedirects as e:
        logging.error(f"Too many redirects for URL {url}: {e}")
        raise
    except asyncio.TimeoutError:
        logging.error(f"Timeout fetching URL {url}")
        raise
    except Exception as e:
        logging.exception(f"Unexpected error fetching URL {url}: {e}")
        raise

    return content_snippets, references, extracted_data

def generate_alternative_queries(original_query):
    prompt = f"Given '{original_query}', suggest 3 refined search queries for enhanced web scraping, optimizing for broad and effective results."
    parts = [{"role": "user", "parts": [{"text": prompt}]}]
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    response = model.generate_content(parts, safety_settings=safety_settings)
    alternative_queries = [q.strip() for q in response.text.split('\n') if q.strip()]
    return alternative_queries


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
async def async_generate_gemini_response(prompt, model_name="gemini-2.0-flash", response_format="markdown"):
    await rate_limit_model(model_name) #Apply rate limit
    parts = [{"role": "user", "parts": [{"text": prompt}]}]
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
    model = genai.GenerativeModel(model_name=model_name)
    try:
        response = await model.generate_content_async(parts, safety_settings=safety_settings)
        text_response = response.text

        if response_format == "json":
            try:
                return json.loads(text_response)
            except json.JSONDecodeError:
                logging.warning("Gemini response was not valid JSON, returning raw text.")
                return {"error": "Invalid JSON response", "raw_text": text_response}
        elif response_format == "csv":
            try:
                csv_data = io.StringIO(text_response)
                reader = csv.reader(csv_data, delimiter=',', quotechar='"')
                return list(reader)
            except Exception as e:
                logging.warning(f"Gemini response was not valid CSV: {e}")
                return {"error": "Invalid CSV response", "raw_text": text_response}

        else:
            text_response = re.sub(r'\n+', '\n\n', text_response)
            text_response = re.sub(r'  +', ' ', text_response)
            text_response = re.sub(r'^- ', '* ', text_response, flags=re.MULTILINE)
            text_response = text_response.replace("```markdown", "")
            return text_response
    except Exception as e:
        logging.error(f"Error generating Gemini response: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
async def chat():
    global conversation_history
    try:
        data = request.json
        user_message = data.get('message', '')
        image_data = data.get('image')
        custom_instruction = data.get('custom_instruction')
        model_name = data.get('model_name', 'gemini-2.0-flash')  # Default model

        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }

        # System instruction handling (prepend if new conversation)
        if custom_instruction and len(conversation_history) == 0:
             # Initialize conversation with system instruction
            system_instruction = {"role": "system", "parts": [custom_instruction]}
            conversation_history.append(system_instruction)
            conversation_history.append({"role": "model", "parts": ["Understood."] }) # Add a confirmation


        parts = []
        # Add existing conversation history
        for item in conversation_history:
          parts.append({"role": item["role"], "parts": [{"text": part} for part in item["parts"]]})

        # Add the current user message
        parts.append({"role": "user", "parts": [{"text": user_message}]})

        if image_data:
            image_part = process_base64_image(image_data)
            if image_part:
                parts.append({"role": "user", "parts": [image_part]})
            else:
                return jsonify({"error": "Failed to process image"}), 400

        model = genai.GenerativeModel(model_name=model_name) # Use selected model
        response = model.generate_content(parts, safety_settings=safety_settings)
        response_text = response.text

        # --- Post-processing ---
        response_text = re.sub(r'\n+', '\n\n', response_text)
        response_text = re.sub(r'  +', ' ', response_text)
        response_text = re.sub(r'^- ', '* ', response_text, flags=re.MULTILINE)

        # Append user message and AI response to the conversation history
        conversation_history.append({"role": "user", "parts": [user_message]})
        conversation_history.append({"role": "model", "parts": [response_text]})

        return jsonify({"response": response_text, "history": conversation_history})
    except Exception as e:
        logging.error(f"Error processing chat request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear', methods=['POST'])
async def clear_history():
    global conversation_history
    conversation_history = []
    return jsonify({"message": "Conversation history cleared successfully"})

async def process_in_chunks(search_results, search_query, prompt_prefix="", fetch_options=None):
    chunk_summaries = []
    references = []
    processed_tokens = 0
    current_chunk_content = []
    extracted_data_all = []

    if fetch_options is None:
        fetch_options = {}

    async with aiohttp.ClientSession() as session:
        for url in search_results:
            try:
                page_snippets, page_refs, extracted_data = await fetch_page_content(session, url, config.DEEP_RESEARCH_SNIPPET_LENGTH, **fetch_options)
                references.extend(page_refs)
                extracted_data_all.append({'url': url, 'data': extracted_data})

                for snippet in page_snippets:
                    estimated_tokens = len(snippet) // 4
                    if processed_tokens + estimated_tokens > config.MAX_TOKENS_PER_CHUNK:
                        combined_content = "\n\n".join(current_chunk_content)
                        if combined_content.strip():
                            summary_prompt = f"{prompt_prefix}\n\n{combined_content}"
                            summary = await async_generate_gemini_response(summary_prompt)
                            chunk_summaries.append(summary)
                        current_chunk_content = []
                        processed_tokens = 0

                    current_chunk_content.append(snippet)
                    processed_tokens += estimated_tokens

            except Exception as e:
                logging.error(f"Error processing URL {url}: {e}")
                continue

        if current_chunk_content:
            combined_content = "\n\n".join(current_chunk_content)
            if combined_content.strip():
                summary_prompt = f"{prompt_prefix}\n\n{combined_content}"
                summary = await async_generate_gemini_response(summary_prompt)
                chunk_summaries.append(summary)

    return chunk_summaries, references, extracted_data_all

@app.route('/api/online', methods=['POST'])
async def online_search_endpoint():
    try:
        data = request.json
        search_query = data.get('query', '')
        if not search_query:
            return jsonify({"error": "No query provided"}), 400

        references = []
        search_results = []
        content_snippets = []
        search_engines_requested = data.get('search_engines', config.SEARCH_ENGINES)

        for engine in search_engines_requested:
            search_results.extend(await scrape_search_engine(search_query, engine))

        if not search_results:
            logging.warning(f"Initial search failed for: {search_query}. Trying alternative queries.")
            alternative_queries = generate_alternative_queries(search_query)
            if alternative_queries:
                logging.info(f"Alternative queries: {alternative_queries}")
                for alt_query in alternative_queries:
                    for engine in search_engines_requested:
                        search_results.extend(await scrape_search_engine(alt_query, engine))
                    if search_results:
                        logging.info(f"Results found with alternative query: {alt_query}")
                        break
            else:
                logging.warning("Gemini failed to generate alternative queries.")

        if not search_results:
            return jsonify({"error": "No search results found"}), 404

        unique_search_results = {}
        for url in search_results:
            unique_search_results[url] = 'general'

        logging.debug(f"Unique URLs to fetch: {unique_search_results.keys()}")

        async with aiohttp.ClientSession() as session:
            tasks = [fetch_page_content(session, url) for url in unique_search_results]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Error during fetching: {result}")
                continue

            page_snippets, page_refs, _ = result
            content_snippets.extend(page_snippets)
            references.extend(page_refs)

        combined_content = "\n\n".join(content_snippets)

        prompt = (
            f"Analyze web content to concisely summarize information for query: '{search_query}'. "
            f"Extract key facts, statistics, and details. Prioritize clarity and avoid speculation. "
            f"Content:\n\n{combined_content}\n\n"
            "Provide a fact-based summary solely from given content."
        )

        explanation = await async_generate_gemini_response(prompt)

        global conversation_history
        conversation_history.append({"role": "user", "parts": [f"Online search query: {search_query}"]})
        conversation_history.append({"role": "model", "parts": [explanation]})
        async with aiohttp.ClientSession() as session:
            shortened_references = [await async_get_shortened_url(session, ref) for ref in references]

        return jsonify({"explanation": explanation, "references": shortened_references, "history": conversation_history})
    except Exception as e:
        logging.exception(f"Error in online search: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/deep_research', methods=['POST'])
async def deep_research_endpoint():
    try:
        data = request.json
        search_query = data.get('query', '')
        if not search_query:
            return jsonify({"error": "No query provided"}), 400

        model_name = data.get('model_name', DEFAULT_DEEP_RESEARCH_MODEL)
        #Removed rate limit check here.  It's handled in async_generate_gemini_response

        start_time = time.time()
        search_engines_requested = data.get('search_engines', config.SEARCH_ENGINES)
        output_format = data.get('output_format', 'markdown')
        extract_links = data.get('extract_links', False)
        extract_emails = data.get('extract_emails', False)
        download_pdf = data.get('download_pdf', True)
        max_iterations = int(data.get('max_iterations', 3))

        all_summaries = []
        all_references = []
        all_extracted_data = []
        current_query = search_query

        async with aiohttp.ClientSession() as session:
            for iteration in range(max_iterations):
                logging.info(f"Iteration {iteration + 1}: {current_query}")
                search_results = []

                for engine in search_engines_requested:
                    search_results.extend(await scrape_search_engine(current_query, engine))

                unique_results = list(set(search_results))
                logging.debug(f"Iteration {iteration + 1} - URLs: {unique_results}")

                prompt_prefix = (
                    f"Analyze snippets for: '{current_query}'. Extract key facts, figures, and insights. "
                    "Be concise, ignore irrelevant content, and prioritize authoritative sources. "
                    "Focus on the main topic and avoid discussing the research process itself.\n\nContent Snippets:"
                )
                fetch_options = {'extract_links': extract_links, 'extract_emails': extract_emails}
                chunk_summaries, refs, extracted = await process_in_chunks(unique_results, current_query, prompt_prefix, fetch_options)
                all_summaries.extend(chunk_summaries)
                all_references.extend(refs)
                all_extracted_data.extend(extracted)

                if iteration < max_iterations - 1:
                    if all_summaries:
                        refinement_prompt = (
                            "Analyze the following research summaries to identify key themes and entities. "
                            "Suggest 3-5 new, more specific search queries that are *directly related* to the original topic. "
                            "Identify any gaps in the current research and suggest queries to address those gaps. "
                            "Do not suggest overly broad or generic queries. Focus on refining the search.\"Research Summaries:\n" + "\n".join(all_summaries) + "\n\n"
                            "Provide refined search queries."
                        )

                        refined_response = await async_generate_gemini_response(refinement_prompt, model_name=model_name)
                        new_queries = [q.strip() for q in refined_response.split('\n') if q.strip()]
                        current_query = " ".join(new_queries[:3])  # Use top queries
                    else:
                        logging.info("No summaries for refinement. Skipping to next iteration.")
                        break  # or continue, depending on whether you want *any* result

            # --- Final Report Generation ---
            if all_summaries:
                if "table" in data.get('output_format', '').lower():
                    table_prompt = generate_table_prompt(search_query)
                    final_prompt = (
                        f"{table_prompt}\n\n"
                        "Research Content:\n" + "\n\n".join(all_summaries) + "\n\n"
                        "Generate the comparison table."  # Simplified, as instructions are in table_prompt
                    )
                else:
                    final_prompt = (
                        "DEEP RESEARCH REPORT: Synthesize a comprehensive report from web research on the topic: '{search_query}'. "
                        "Integrate information from all iterations, identify major themes, compare different viewpoints, and provide a balanced and objective analysis. "
                        "Discard any redundant or irrelevant information.  Prioritize clarity and conciseness. "
                        "Aim for a well-structured report suitable for conversion to a 5-7 page PDF. "
                        "Do *not* include any discussion of the research methods or tools used – focus solely on the findings.\n\n"
                        "Research Summaries (from all iterations):\n" + "\n\n".join(all_summaries) + "\n\n"
                        "Generate the report in Markdown format."
                    )


                final_explanation = await async_generate_gemini_response(final_prompt, response_format=output_format, model_name=model_name)

                # --- Table Parsing and Fallback ---
                if "table" in output_format.lower():
                    try:
                        parsed_table = parse_markdown_table(final_explanation)
                        if parsed_table:
                            final_explanation = parsed_table  # Use the parsed table
                        else:  # Fallback: If parsing fails completely
                            logging.warning("Table parsing failed.  Returning raw response.")
                            final_explanation = {"error": "Failed to parse table", "raw_text": final_explanation}
                    except Exception as e:
                        logging.error(f"Error during table parsing: {e}")
                        final_explanation = {"error": "Failed to parse table", "raw_text": final_explanation}
            else:
                final_explanation = "No relevant content found for the given query."

            global conversation_history
            conversation_history.append({"role": "user", "parts": [f"Deep research query: {search_query}"]})
            conversation_history.append({"role": "model", "parts": [final_explanation]})

            end_time = time.time()
            elapsed_time = end_time - start_time

            if download_pdf:
                pdf_buffer = await generate_pdf(search_query, final_explanation if isinstance(final_explanation, str) else "\n".join(str(row) for row in final_explanation), all_references) #handles table data for pdf
                sanitized_filename = quote_plus(search_query)
                return send_file(
                    pdf_buffer,
                    as_attachment=True,
                    download_name=f"deep_research_{sanitized_filename}.pdf",
                    mimetype='application/pdf'
                )

            # --- JSON, CSV, and Markdown Responses (Similar to before) ---
            response_data = {
                "explanation": final_explanation,
                "references": all_references,
                "history": conversation_history,
                "elapsed_time": f"{elapsed_time:.2f} seconds",
                "extracted_data": all_extracted_data,
            }

            if output_format == 'json':
                if isinstance(final_explanation, dict):  # Handle potential JSON error
                    response_data = final_explanation
                elif isinstance(final_explanation, list): #handle table data
                    response_data = {"table_data": final_explanation}
                else:
                    response_data = {"explanation": final_explanation} #Ensure it's a dict.
                response_data.update({
                    "references": all_references,
                    "history": conversation_history,
                    "elapsed_time": f"{elapsed_time:.2f} seconds",
                    "extracted_data": all_extracted_data
                })
                return jsonify(response_data)

            elif output_format == 'csv':
                if isinstance(final_explanation, list):
                    response_data = {"explanation": final_explanation}
                elif isinstance(final_explanation, dict) and "raw_text" in final_explanation:
                    response_data = {"explanation": final_explanation["raw_text"]}
                else:
                    response_data = {"explanation": final_explanation}
                response_data.update({
                    "references": all_references,
                    "history": conversation_history,
                    "elapsed_time": f"{elapsed_time:.2f} seconds",
                    "extracted_data": all_extracted_data
                })
                return jsonify(response_data)

            return jsonify(response_data)

    except Exception as e:
        logging.exception(f"Error in deep research: {e}")  # Log full traceback
        return jsonify({"error": str(e)}), 500


def generate_table_prompt(query):
    """
    Generates a prompt that enforces strict table formatting rules for consistent output.
    """
    prompt = (
        f"Create a detailed comparison table analyzing: '{query}'\n\n"
        "STRICT TABLE FORMATTING REQUIREMENTS:\n"
        "1. Output must be ONLY a properly formatted Markdown table\n"
        "2. Table must follow this EXACT structure:\n"
        "   - Header row with clear column names\n"
        "   - Separator row with exactly three dashes (---) per column\n"
        "   - Data rows with consistent formatting\n"
        "3. ALL rows (including header and data) must:\n"
        "   - Start with a pipe (|)\n"
        "   - End with a pipe (|)\n"
        "   - Have exactly one pipe between each column\n"
        "   - Have exactly one space after each pipe\n"
        "   - Have exactly one space before each pipe\n"
        "4. The separator row must:\n"
        "   - Use exactly three dashes (---) for each column\n"
        "   - Include alignment colons if needed (:---:)\n"
        "5. Content rules:\n"
        "   - Keep cell content concise (max 2-3 lines)\n"
        "   - Use consistent capitalization\n"
        "   - Avoid empty cells (use 'N/A' if needed)\n"
        "   - No line breaks within cells\n\n"
        "REQUIRED FORMAT EXAMPLE:\n"
        "| Column 1 | Column 2 | Column 3 |\n"
        "|----------|----------|----------|\n"
        "| Data A   | Data B   | Data C   |\n"
        "| Data D   | Data E   | Data F   |\n\n"
        "GENERATE THE TABLE NOW:\n"
        "- Use 3-5 relevant columns\n"
        "- Include 4-8 data rows\n"
        "- Ensure ALL content is properly aligned\n"
        "- Verify each line follows the pipe and spacing rules exactly\n"
        "Output ONLY the table, with NO additional text before or after."
    )
    return prompt

def parse_markdown_table(markdown_table_string):
    """
    Parses a Markdown table string with improved robustness.
    Handles missing pipes, extra spaces, and inconsistent separators.
    """
    lines = [line.strip() for line in markdown_table_string.split('\n') if line.strip()]
    if not lines:
        return []

    table_data = []
    header_detected = False  # Flag to track if we've found the header

    for line in lines:
        # Clean up the line: Remove leading/trailing pipes AND extra spaces *around* pipes
        line = line.strip().strip('|').replace(' | ', '|').replace('| ', '|').replace(' |', '|')

        # Split by pipe (now that we've handled spacing inconsistencies)
        cells = [cell.strip() for cell in line.split('|')]

        # Header detection (more flexible):  Look for a line with mostly dashes
        if all(c in '-:| ' for c in line) and len(cells) > 1 and not header_detected:
             # Consider it a separator line, but don't require it at a specific index.
            header_detected = True
            continue

        if cells:  # Add non-empty rows
            table_data.append(cells)

    # Ensure all rows have the same number of columns (as the *first* row)
    if table_data:
      max_cols = len(table_data[0])
      normalized_data = []
      for row in table_data:
          normalized_data.append(row + [''] * (max_cols - len(row)))
      return normalized_data
    else:
      return []

async def generate_pdf(report_title, content, references):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                          rightMargin=0.7*inch, leftMargin=0.7*inch,
                          topMargin=0.7*inch, bottomMargin=0.7*inch)
    styles = getSampleStyleSheet()

    # Get current date
    today = date.today()
    formatted_date = today.strftime("%B %d, %Y")

    # Custom Styles Definition
    custom_styles = {
        'Title': ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            leading=32,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=HexColor("#1a237e"),
            fontName='Helvetica-Bold'
        ),
        'Heading1': ParagraphStyle(
            'CustomHeading1',
            parent=styles['Heading1'],
            fontSize=18,
            leading=24,
            spaceBefore=24,
            spaceAfter=16,
            textColor=HexColor("#283593"),
            fontName='Helvetica-Bold',
            keepWithNext=True
        ),
        'Heading2': ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontSize=16,
            leading=22,
            spaceBefore=16,
            spaceAfter=12,
            textColor=HexColor("#3949ab"),
            fontName='Helvetica-Bold',
            keepWithNext=True
        ),
        'Paragraph': ParagraphStyle(
            'CustomParagraph',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            textColor=HexColor("#212121"),
            firstLineIndent=0.25*inch
        ),
        'TableCell': ParagraphStyle(
            'CustomTableCell',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            spaceBefore=6,
            spaceAfter=6,
            textColor=HexColor("#212121")
        ),
        'Bullet': ParagraphStyle(
            'CustomBullet',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            leftIndent=0.5*inch,
            rightIndent=0,
            spaceBefore=6,
            spaceAfter=6,
            bulletIndent=0.3*inch,
            textColor=HexColor("#212121"),
            bulletFontName='Helvetica',
            bulletFontSize=11
        ),
        'Reference': ParagraphStyle(
            'CustomReference',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            spaceAfter=6,
            textColor=HexColor("#1565c0"),
            alignment=TA_LEFT,
            leftIndent=0.5*inch
        ),
        'Footer': ParagraphStyle(
            'CustomFooter',
            parent=styles['Italic'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=HexColor("#757575"),
            spaceBefore=36
        )
    }

    def clean_text(text):
        # Remove markdown symbols while preserving content
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)  # Links
        # Convert HTML entities
        text = text.replace('&', '&').replace('<', '<').replace('>', '>')
        return text.strip()

    def process_table(table_text):
        rows = [row.strip() for row in table_text.split('\n') if row.strip()]
        if len(rows) < 2:  # Need at least header and separator
            return None

        # Process header
        header = [clean_text(cell) for cell in rows[0].strip('|').split('|')]
        
        # Skip separator row
        data_rows = []
        for row in rows[2:]:  # Skip the separator row
            cells = [clean_text(cell) for cell in row.strip('|').split('|')]
            data_rows.append(cells)

        # Create table data with processed text
        table_data = [[Paragraph(cell, custom_styles['TableCell']) for cell in header]]
        for row in data_rows:
            table_data.append([Paragraph(cell, custom_styles['TableCell']) for cell in row])

        # Create table with styling
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#f5f5f5")),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#1a237e")),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, HexColor("#e0e0e0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f8f9fa")]),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ])

        col_widths = [doc.width/len(header) for _ in header]
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(table_style)
        return table

    story = []

    # Title and Date
    story.append(Paragraph(report_title, custom_styles['Title']))
    story.append(Paragraph(formatted_date, custom_styles['Footer']))
    story.append(Spacer(1, 0.3*inch))

    # Process content
    current_table = []
    in_table = False
    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            if in_table and current_table:
                table = process_table('\n'.join(current_table))
                if table:
                    story.append(table)
                    story.append(Spacer(1, 0.2*inch))
                current_table = []
                in_table = False
            story.append(Spacer(1, 0.1*inch))
            i += 1
            continue

        # Table detection and processing
        if '|' in line and (line.count('|') > 1 or (i + 1 < len(lines) and '|' in lines[i + 1])):
            in_table = True
            current_table.append(line)
        elif in_table:
            if current_table:  # End of table
                table = process_table('\n'.join(current_table))
                if table:
                    story.append(table)
                    story.append(Spacer(1, 0.2*inch))
            current_table = []
            in_table = False
            continue
        # Headers
        elif line.startswith('# '):
            story.append(Paragraph(clean_text(line[2:]), custom_styles['Heading1']))
        elif line.startswith('## '):
            story.append(Paragraph(clean_text(line[3:]), custom_styles['Heading2']))
        # Bullet points
        elif line.startswith('* ') or line.startswith('- '):
            story.append(Paragraph(f"• {clean_text(line[2:])}", custom_styles['Bullet']))
        # Regular paragraphs
        else:
            if not in_table:
                story.append(Paragraph(clean_text(line), custom_styles['Paragraph']))

        i += 1

    # Process any remaining table
    if current_table:
        table = process_table('\n'.join(current_table))
        if table:
            story.append(table)
            story.append(Spacer(1, 0.2*inch))

    # References section
    if references:
        story.append(PageBreak())
        story.append(Paragraph("References", custom_styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        for i, ref in enumerate(references, 1):
            story.append(Paragraph(f"[{i}] {ref}", custom_styles['Reference']))

    # Footer
    story.append(Spacer(1, 0.5*inch))
    footer_text = f"Generated by Kv - AI Companion & Deep Research Tool • {formatted_date}"
    story.append(Paragraph(footer_text, custom_styles['Footer']))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


@app.route('/api/scrape_product', methods=['POST'])
async def scrape_product_endpoint():
    try:
        data = request.json
        product_query = data.get('query', '')
        if not product_query:
            return jsonify({"error": "No product query provided"}), 400

        products = await scrape_amazon_product(product_query)
        if not products:
            return jsonify({"message": "No products found for this query on Amazon."}), 200

        prompt_content = "\n".join([f"- {p['title']} - Price: {p['price']} - [Link]({p['link']})" for p in products])
        prompt = (
            f"Here is a list of products found on Amazon for the query '{product_query}':\n\n"
            f"{prompt_content}\n\n"
            "Summarize these product listings, highlighting the key features, price ranges, and any notable details. "
            "Provide a well-formatted markdown summary."
        )
        explanation = await async_generate_gemini_response(prompt)

        return jsonify({"explanation": explanation, "products": products})
    except Exception as e:
        logging.error(f"Error in product scraping endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/scrape_reviews', methods=['POST'])
async def scrape_reviews_endpoint():
    try:
        data = request.json
        business_name = data.get('business_name', '')
        location = data.get('location', '')
        if not business_name:
            return jsonify({"error": "No business name provided"}), 400

        reviews = await scrape_yelp_reviews(business_name, location)
        if not reviews:
            return jsonify({"message": f"No reviews found for '{business_name}' on Yelp."}), 200

        prompt_content = "\n".join([f"- User: {r['user']}, Rating: {r['rating']}, Comment: {r['comment']}" for r in reviews])
        prompt = (
            f"Here are some reviews for '{business_name}' from Yelp:\n\n"
            f"{prompt_content}\n\n"
            "Analyze these reviews and provide a summary of the general sentiment, "
            "mentioning key positive and negative points if available. Format the summary in markdown."
        )
        explanation = await async_generate_gemini_response(prompt)

        return jsonify({"explanation": explanation, "reviews": reviews})

    except Exception as e:
        logging.error(f"Error in reviews scraping endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/scrape_jobs', methods=['POST'])
async def scrape_jobs_endpoint():
    try:
        data = request.json
        job_title = data.get('job_title', '')
        job_location = data.get('job_location', '')
        if not job_title:
            return jsonify({"error": "No job title provided"}), 400

        jobs = await scrape_indeed_jobs(job_title, job_location)
        if not jobs:
            return jsonify({"message": f"No jobs found for '{job_title}' in '{job_location}' on Indeed."}), 200

        prompt_content = "\n".join([f"- {j['title']} at {j['company']} ({j['location']}) - [Link]({j['link']})" for j in jobs if j['link']])
        prompt = (
            f"Here are some job listings for '{job_title}' in '{job_location}' from Indeed:\n\n"
            f"{prompt_content}\n\n"
            "Summarize these job listings, highlighting the types of roles, companies, and locations. "
            "Provide a well-formatted markdown summary."
        )
        explanation = await async_generate_gemini_response(prompt)

        return jsonify({"explanation": explanation, "jobs": jobs})

    except Exception as e:
        logging.error(f"Error in jobs scraping endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/scrape_news_site', methods=['POST'])
async def scrape_news_site_endpoint():
    try:
        data = request.json
        news_url = data.get('news_url', '')
        if not news_url:
            return jsonify({"error": "No news URL provided"}), 400

        articles = await scrape_news_from_site(news_url)
        if not articles:
            return jsonify({"message": f"Could not scrape articles from '{news_url}'."}), 200

        prompt_content = "\n".join([f"- [{article['title']}]({article['link']})" for article in articles])
        prompt = (
            f"Here are the latest articles scraped from '{news_url}':\n\n"
            f"{prompt_content}\n\n"
            "Provide a summary of the topics covered in these articles. "
            "Format the summary in markdown."
        )
        explanation = await async_generate_gemini_response(prompt)
        return jsonify({"explanation": explanation, "articles": articles})

    except Exception as e:
        logging.error(f"Error in news site scraping endpoint: {e}")
        return jsonify({"error": str(e)}), 500

async def scrape_amazon_product(product_name):
    products = []
    amazon_url = f"https://www.amazon.com/s?k={product_name.replace(' ', '+')}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(amazon_url, headers={'User-Agent': get_random_user_agent()}, timeout=10) as amazon_resp:
                if amazon_resp.status == 200:
                    amazon_soup = BeautifulSoup(await amazon_resp.text(), 'html.parser')
                    product_listings = amazon_soup.find_all('div', class_='s-result-item', limit=20)
                    for product in product_listings:
                        title_tag = product.find('span', class_='a-text-normal')
                        link_tag = product.find('a', class_='a-link-normal')
                        price_tag = product.find('span', class_='a-price')
                        if title_tag and link_tag:
                            title = title_tag.text.strip()
                            link = "https://www.amazon.com" + link_tag['href']
                            price = price_tag.find('span', class_='a-offscreen').text.strip() if price_tag and price_tag.find('span', class_='a-offscreen') else "Price not available"
                            products.append({
                                "title": title,
                                "link": link,
                                "price": price
                            })
    except Exception as e:
        logging.error(f"Error scraping Amazon: {e}")
    return products

async def scrape_yelp_reviews(business_name, location=""):
    reviews_data = []
    search_query = f"{business_name} {location}".replace(' ', '+')
    yelp_url = f"https://www.yelp.com/search?find_desc={search_query}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(yelp_url, headers={'User-Agent': get_random_user_agent()}, timeout=10) as yelp_resp:
                if yelp_resp.status == 200:
                    yelp_soup = BeautifulSoup(await yelp_resp.text(), 'html.parser')
                    review_listings = yelp_soup.find_all('div', class_='review__373c0__3MsBX border-color--default__373c0__2oFDT', limit=20)
                    for review_block in review_listings:
                        user_tag = review_block.find('a', class_='css-19v1jt')
                        rating_tag = review_block.find('div', class_='i-stars') # Corrected class name
                        comment_tag = review_block.find('span', class_='raw__373c0__3rcx7')
                        if user_tag and rating_tag and comment_tag:
                            user = user_tag.text.strip()
                            rating = rating_tag.get('aria-label')
                            comment = comment_tag.text.strip()
                            reviews_data.append({
                                "user": user,
                                "rating": rating,
                                "comment": comment
                            })
    except Exception as e:
        logging.error(f"Error scraping Yelp: {e}")
    return reviews_data

async def scrape_indeed_jobs(job_title, job_location=""):
	jobs = []
	search_query = f"{job_title} in {job_location}".replace(' ', '+')
	indeed_url = f"https://www.indeed.com/jobs?q={search_query}&l={job_location}"
	try:
		async with aiohttp.ClientSession() as session:
			async with session.get(indeed_url, headers={'User-Agent': get_random_user_agent()}, timeout=10) as indeed_resp:
				if indeed_resp.status == 200:
					indeed_soup = BeautifulSoup(await indeed_resp.text(), 'html.parser')
					job_listings = indeed_soup.find_all('div', class_='jobsearch-SerpJobCard', limit=20)
					for job_card in job_listings:
						title_tag = job_card.find('a', class_='jobtitle')
						company_tag = job_card.find('span', class_='company')
						location_tag = job_card.find('div', class_='location')
						if title_tag and company_tag and location_tag:
							title = title_tag.text.strip()
							company = company_tag.text.strip()
							location = location_tag.text.strip()
							link = 'https://www.indeed.com' + title_tag['href'] if title_tag.has_attr('href') else None

							jobs.append({
                                "title": title,
                                "company": company,
                                "location": location,
                                "link": link
                            })
	except Exception as e:
		logging.error(f"Error scraping Indeed: {e}")
	return jobs

async def scrape_news_from_site(news_url):
    articles = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(news_url, headers={'User-Agent': get_random_user_agent()}, timeout=10) as news_resp:
                if news_resp.status == 200:
                    news_soup = BeautifulSoup(await news_resp.text(), 'html.parser')
                    headlines = news_soup.find_all('h2', limit=30)
                    for headline in headlines:
                        link_tag = headline.find('a')
                        if link_tag and link_tag.has_attr('href'):
                            article_link = urljoin(news_url, link_tag['href'])
                            articles.append({
                                "title": headline.text.strip(),
                                "link": article_link
                            })
    except Exception as e:
        logging.error(f"Error scraping news site: {e}")
    return articles

@app.before_request
def before_request():
    g.request_active = True

@app.teardown_request
def teardown_request(exception=None):
    g.request_active = False

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        if loop and loop.is_running():
            loop.close()
            print("Event Loop Closed on Exit")
