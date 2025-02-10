from flask import Flask, request, jsonify, render_template,g
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
import time
from urllib.parse import urlparse, urljoin
from functools import lru_cache, wraps
import asyncio
import aiohttp
import chardet
import json  # For JSON output
import csv   # For CSV output

load_dotenv()

app = Flask(__name__)
CORS(app)

# --- Configuration ---
class Config:
    API_KEY = os.getenv("GEMINI_API_KEY")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
    SNIPPET_LENGTH = 50000
    DEEP_RESEARCH_SNIPPET_LENGTH = 100000
    MAX_TOKENS_PER_CHUNK = 5_000_000
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (iPad; CPU OS 14_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1'
    ]
    SEARCH_ENGINES = ["google", "duckduckgo", "bing", "yahoo", "brave", "linkedin"] # Added configurable search engines

config = Config()

# Configure Logging
logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Google Gemini API
genai.configure(api_key=config.API_KEY)

# Initialize conversation history
conversation_history = []

def get_random_user_agent():
    return random.choice(config.USER_AGENTS)

def process_base64_image(base64_string):
    """Convert base64 image data, decode, and prepare for Gemini."""
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

@lru_cache(maxsize=128)
async def async_get_shortened_url(session, url):
    """Asynchronously shorten a URL using TinyURL, with caching."""
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            url = "http://" + url
        tinyurl_api = f"https://tinyurl.com/api-create.php?url={url}"
        async with session.get(tinyurl_api, timeout=5) as response:
            if response.status == 200:
                return await response.text()  # Get the shortened URL
            else:
                logging.warning(f"TinyURL API returned status code {response.status} for URL: {url}")
                return url  # Return the original URL on failure
    except Exception as e:
        logging.error(f"Error shortening URL '{url}': {e}")
        return url  # Return original URL

def fix_url(url):
    """Ensure the URL has a proper scheme."""
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

# --- Search Engine Scraping Functions ---
async def scrape_search_engine(search_query, engine_name):
    """Dispatch search query to the appropriate scraping function."""
    if engine_name == "google":
        return scrape_google(search_query)
    elif engine_name == "duckduckgo":
        return scrape_duckduckgo(search_query)
    elif engine_name == "bing":
        return scrape_bing(search_query)
    elif engine_name == "yahoo":
        return scrape_yahoo(search_query)
    elif engine_name == "brave":
        return scrape_brave(search_query)
    elif engine_name == "linkedin":
        return scrape_linkedin(search_query)
    else:
        logging.warning(f"Unknown search engine: {engine_name}")
        return []

def scrape_google(search_query):
	search_results = []
	google_url = f"https://www.google.com/search?q={search_query}"
	try:
		google_resp = requests.get(google_url, headers={'User-Agent': get_random_user_agent()}, timeout=10)
		logging.info(f"Google Search Status Code: {google_resp.status_code} for query: {search_query}")
		if google_resp.status_code == 200:
			google_soup = BeautifulSoup(google_resp.text, 'html.parser')
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
def scrape_duckduckgo(search_query):
	search_results = []
	duck_url = f"https://html.duckduckgo.com/html/?q={search_query}"
	try:
		duck_resp = requests.get(duck_url, headers={'User-Agent': get_random_user_agent()}, timeout=10)
		logging.info(f"DuckDuckGo Search Status Code: {duck_resp.status_code} for query: {search_query}")
		if duck_resp.status_code == 200:
			duck_soup = BeautifulSoup(duck_resp.text, 'html.parser')
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

def scrape_bing(search_query):
	search_results = []
	bing_url = f"https://www.bing.com/search?q={search_query}"
	try:
		bing_resp = requests.get(bing_url, headers={'User-Agent': get_random_user_agent()}, timeout=10)
		logging.info(f"Bing Search Status Code: {bing_resp.status_code} for query: {search_query}")
		if bing_resp.status_code == 200:
			bing_soup = BeautifulSoup(bing_resp.text, 'html.parser')
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

def scrape_yahoo(search_query):
	search_results = []
	yahoo_url = f"https://search.yahoo.com/search?p={search_query}"
	try:
		yahoo_resp = requests.get(yahoo_url, headers={'User-Agent': get_random_user_agent()}, timeout=10)
		logging.info(f"Yahoo Search Status Code: {yahoo_resp.status_code} for query: {search_query}")
		if yahoo_resp.status_code == 200:
			yahoo_soup = BeautifulSoup(yahoo_resp.text, 'html.parser')
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

def scrape_brave(search_query):
	search_results = []
	brave_url = f"https://search.brave.com/search?q={search_query}"
	try:
		brave_resp = requests.get(brave_url, headers={'User-Agent': get_random_user_agent()}, timeout=10)
		logging.info(f"Brave Search Status Code: {brave_resp.status_code} for query: {search_query}")
		if brave_resp.status_code == 200:
			brave_soup = BeautifulSoup(brave_resp.text, 'html.parser')
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

def scrape_linkedin(search_query):
   search_results = []
   linkedin_url = f"https://www.linkedin.com/search/results/all/?keywords={search_query.replace(' ', '%20')}"

   try:
        linkedin_resp = requests.get(linkedin_url, headers={'User-Agent': get_random_user_agent()}, timeout=10)
        logging.info(f"LinkedIn Search Status Code: {linkedin_resp.status_code} for query: {search_query}")

        if linkedin_resp.status_code == 200:
            linkedin_soup = BeautifulSoup(linkedin_resp.text, 'html.parser')

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
    """Fetch page content, optionally extract links and emails."""
    if snippet_length is None:
        snippet_length = config.SNIPPET_LENGTH  # Default to config snippet length

    content_snippets = []
    references = []
    extracted_data = {} # Dictionary to hold extracted data

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

                    # Data Extraction (simple examples)
                    if extract_links:
                        extracted_data['links'] = [a['href'] for a in page_soup.find_all('a', href=True)]
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
   prompt = f"Given the search query '{original_query}', suggest 3 alternative search queries that might yield better results for web scraping. Focus on rephrasing to be more effective for broad web search."
   parts = [{"role": "user", "parts": [{"text": prompt}]}]
   safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
   model = genai.GenerativeModel(model_name="gemini-1.5-flash")
   response = model.generate_content(parts, safety_settings=safety_settings)
   alternative_queries = [q.strip() for q in response.text.split('\n') if q.strip()]
   return alternative_queries

def retry(func):
    """Decorator for retrying a function with exponential backoff and correct loop handling."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        max_retries = 5
        delay = 1
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logging.debug(f"Retrying due to {e}, sleeping {delay}s ...")
                loop = asyncio.get_running_loop()
                await asyncio.sleep(delay)
                delay *= 2
    return wrapper

@retry
async def async_generate_gemini_response(prompt, model_name="gemini-1.5-flash", response_format="markdown"):
    """Generate Gemini response, optionally format as JSON."""
    parts = [{"role": "user", "parts": [{"text": prompt}]}]
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
    model = genai.GenerativeModel(model_name=model_name)
    response = await model.generate_content_async(parts, safety_settings=safety_settings)
    text_response = response.text

    if response_format == "json":
        try:
            return json.loads(text_response)  # Attempt to parse as JSON
        except json.JSONDecodeError:
            logging.warning("Gemini response was not valid JSON, returning raw text.")
            return text_response
    elif response_format == "csv":
        # Basic CSV formatting - might need more sophisticated handling
        lines = text_response.strip().split('\n')
        return [line.split(',') for line in lines] # Simple comma split, improve as needed
    else: # default markdown formatting
        text_response = re.sub(r'\n+', '\n\n', text_response)
        text_response = re.sub(r'  +', ' ', text_response)
        text_response = re.sub(r'^- ', '* ', text_response, flags=re.MULTILINE)
        return text_response

# --- Flask Route Handlers ---
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
        model_name = data.get('model_name', 'gemini-1.5-flash')

        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }

        if custom_instruction and len(conversation_history) == 0:
            conversation_history.append({"role": "user", "parts": [custom_instruction]})
            conversation_history.append({"role": "model", "parts": ["Understood."] })

        parts = []
        for item in conversation_history:
            parts.append({"role": item["role"], "parts": [{"text": part} for part in item["parts"]]})
        parts.append({"role": "user", "parts": [{"text": user_message}]})

        if image_data:
            image_part = process_base64_image(image_data)
            if image_part:
                parts.append({"role": "user", "parts": [image_part]})
            else:
                return jsonify({"error": "Failed to process image"}), 400

        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(parts, safety_settings=safety_settings)
        response_text = response.text
        response_text = re.sub(r'\n+', '\n\n', response_text)
        response_text = re.sub(r'  +', ' ', response_text)
        response_text = re.sub(r'^- ', '* ', response_text, flags=re.MULTILINE)

        conversation_history.append({"role": "user", "parts": [user_message]})
        conversation_history.append({"role": "model", "parts": [response_text]})

        return jsonify({"response": response_text, "history": conversation_history})
    except Exception as e:
        logging.error(f"Error processing chat request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear', methods=['POST'])
async def clear_history():
    """Clears the conversation history."""
    global conversation_history
    conversation_history = []
    return jsonify({"message": "Conversation history cleared successfully"})

async def process_in_chunks(search_results, search_query, prompt_prefix="", fetch_options=None):
    """Process content in chunks, with fetch options."""
    chunk_summaries = []
    references = []
    processed_tokens = 0
    current_chunk_content = []
    extracted_data_all = [] # List to store extracted data from each page

    if fetch_options is None:
        fetch_options = {} # Default fetch options if none provided

    async with aiohttp.ClientSession() as session:
        for url in search_results:
            try:
                page_snippets, page_refs, extracted_data = await fetch_page_content(session, url, config.DEEP_RESEARCH_SNIPPET_LENGTH, **fetch_options) # Pass fetch options
                references.extend(page_refs)
                extracted_data_all.append({'url': url, 'data': extracted_data}) # Store extracted data with URL

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
        search_engines_requested = data.get('search_engines', config.SEARCH_ENGINES) # Get engines from request or config

        # --- Scraping ---
        for engine in search_engines_requested: # Iterate through selected search engines
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

        # --- Fetching and Prompting ---
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_page_content(session, url) for url in unique_search_results]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Error during fetching: {result}")
                continue

            page_snippets, page_refs, _ = result # Ignore extracted data here for online search
            content_snippets.extend(page_snippets)
            references.extend(page_refs)

        combined_content = "\n\n".join(content_snippets)

        prompt = (
            f"Provide the current weather information for: {search_query}. "
            f"Give a concise answer with key details like temperature, conditions, and wind. "
            f"Here is content from various online sources:\n\n{combined_content}\n\n"
            "Provide current weather information based on the provided content. Do NOT give a general analysis.  Give direct, specific answers."
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

        start_time = time.time()

        search_engines_requested = data.get('search_engines', config.SEARCH_ENGINES) # Get engines from request or config
        output_format = data.get('output_format', 'markdown') # Get output format from request, default markdown
        extract_links = data.get('extract_links', False) # Example data extraction options from request
        extract_emails = data.get('extract_emails', False)

        # --- Scraping ---
        search_results = []
        for engine in search_engines_requested: # Use selected search engines
            engine_results = await scrape_search_engine(search_query, engine)
            search_results.extend(engine_results[:30] if engine != 'linkedin' else engine_results[:20]) # Limit results per engine

        alternative_queries = generate_alternative_queries(search_query)
        if alternative_queries:
            logging.info(f"Alternative queries generated: {alternative_queries}")
            for alt_query in alternative_queries:
                for engine in search_engines_requested:
                    alt_engine_results = await scrape_search_engine(alt_query, engine)
                    search_results.extend(alt_engine_results[:20] if engine != 'linkedin' else alt_engine_results[:10])

        unique_search_results = list(set(search_results))
        logging.debug(f"Unique URLs to fetch: {unique_search_results}")

        # --- Chunking and Prompting ---
        fetch_options = {'extract_links': extract_links, 'extract_emails': extract_emails} # Bundle fetch options
        chunk_summaries, references, extracted_data_all = await process_in_chunks(
            unique_search_results,
            search_query,
            prompt_prefix=(
                "Summarize the following content, focusing on extracting key information *relevant to the query*. "
                "If the content is irrelevant or low-quality, briefly note that and move on. Prioritize reputable sources. "
                f"Content related to '{search_query}':"
            ),
            fetch_options=fetch_options # Pass fetch options to chunk processing
        )

        # Final summarization
        if chunk_summaries:
            final_prompt = (
                "This is a DEEP RESEARCH request. Synthesize the following summaries, "
                f"derived from extensive web scraping on the topic: '{search_query}'. "
                "Provide a detailed and comprehensive analysis, integrating information, "
                "identifying key themes, and conflicting viewpoints. Discard irrelevant information. "
                "Format the output in markdown for readability.\n\n"
                + "\n\n".join(chunk_summaries)
            )
            final_explanation = await async_generate_gemini_response(final_prompt, response_format=output_format) # Pass output format
        else:
            final_explanation = "No content could be summarized for the given query."

        global conversation_history
        conversation_history.append({"role": "user", "parts": [f"Deep research query: {search_query}"]})
        conversation_history.append({"role": "model", "parts": [final_explanation]})

        end_time = time.time()
        elapsed_time = end_time - start_time
        async with aiohttp.ClientSession() as session:
            shortened_references = [await async_get_shortened_url(session, ref) for ref in references]

        response_data = {
            "explanation": final_explanation,
            "references": shortened_references,
            "history": conversation_history,
            "elapsed_time": f"{elapsed_time:.2f} seconds"
        }
        if extracted_data_all: # Include extracted data if any
            response_data["extracted_data"] = extracted_data_all

        return jsonify(response_data)

    except Exception as e:
        logging.exception(f"Error in deep research: {e}")
        return jsonify({"error": str(e)}), 500

# --- Other Scraping Endpoints (Made Async, and using async_generate_gemini_response) ---

@app.route('/api/scrape_product', methods=['POST'])
async def scrape_product_endpoint():
    try:
        data = request.json
        product_query = data.get('query', '')
        if not product_query:
            return jsonify({"error": "No product query provided"}), 400

        products = scrape_amazon_product(product_query)
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

        reviews = scrape_yelp_reviews(business_name, location)
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

        jobs = scrape_indeed_jobs(job_title, job_location)
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

        articles = scrape_news_from_site(news_url)
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

def scrape_amazon_product(product_name):
    products = []
    amazon_url = f"https://www.amazon.com/s?k={product_name.replace(' ', '+')}"
    try:
        amazon_resp = requests.get(amazon_url, headers={'User-Agent': get_random_user_agent()}, timeout=10)
        if amazon_resp.status_code == 200:
            amazon_soup = BeautifulSoup(amazon_resp.text, 'html.parser')
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

def scrape_yelp_reviews(business_name, location=""):
    reviews_data = []
    search_query = f"{business_name} {location}".replace(' ', '+')
    yelp_url = f"https://www.yelp.com/search?find_desc={search_query}"
    try:
        yelp_resp = requests.get(yelp_url, headers={'User-Agent': get_random_user_agent()}, timeout=10)
        if yelp_resp.status_code == 200:
            yelp_soup = BeautifulSoup(yelp_resp.text, 'html.parser')
            review_listings = yelp_soup.find_all('div', class_='review__373c0__3MsBX border-color--default__373c0__2oFDT', limit=20)
            for review_block in review_listings:
                user_tag = review_block.find('a', class_='css-19v1jt')
                rating_tag = review_block.find('div', class_='i-stars')
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

def scrape_indeed_jobs(job_title, job_location=""):
	jobs = []
	search_query = f"{job_title} in {job_location}".replace(' ', '+')
	indeed_url = f"https://www.indeed.com/jobs?q={search_query}&l={job_location}"
	try:
		indeed_resp = requests.get(indeed_url, headers={'User-Agent': get_random_user_agent()}, timeout=10)
		if indeed_resp.status_code == 200:
			indeed_soup = BeautifulSoup(indeed_resp.text, 'html.parser')
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

def scrape_news_from_site(news_url):
    articles = []
    try:
        news_resp = requests.get(news_url, headers={'User-Agent': get_random_user_agent()}, timeout=10)
        if news_resp.status_code == 200:
            news_soup = BeautifulSoup(news_resp.text, 'html.parser')
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

# --- Event Loop Management ---
@app.before_request
def before_request():
    g.request_active = True

@app.teardown_request
def teardown_request(exception=None):
    g.request_active = False

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# --- Run the App ---
if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        if loop and loop.is_running():
            loop.close()
            print("Event Loop Closed on Exit")