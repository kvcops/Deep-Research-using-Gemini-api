# app.py
import chardet
from flask import Flask, request, jsonify, render_template, send_file
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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time
from urllib.parse import urlparse, urljoin, quote_plus, unquote
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
import concurrent.futures
import brotli
from pdfminer.high_level import extract_text as pdf_extract_text  # For PDF parsing
import docx2txt # For docx processing


load_dotenv()

app = Flask(__name__)
CORS(app)

class Config:
    API_KEY = os.getenv("GEMINI_API_KEY")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    SNIPPET_LENGTH = 5000
    DEEP_RESEARCH_SNIPPET_LENGTH = 10000
    MAX_TOKENS_PER_CHUNK = 25000
    REQUEST_TIMEOUT = 60
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (iPad; CPU OS 14_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1'
    ]
    SEARCH_ENGINES = ["google", "duckduckgo", "bing", "yahoo", "brave", "linkedin"]
    JOB_SEARCH_ENGINES = ["linkedin", "indeed", "glassdoor", "monster"]
    MAX_WORKERS = 10
    CACHE_ENABLED = True  # Enable/disable caching
    CACHE = {} # Simple in-memory cache
    CACHE_TIMEOUT = 300  # Cache timeout in seconds

config = Config()

logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
genai.configure(api_key=config.API_KEY)
conversation_history = []
deep_research_rate_limits = {
    "gemini-2.0-flash": {"requests_per_minute": 15, "last_request": 0},
    "gemini-2.0-flash-thinking-exp-01-21": {"requests_per_minute": 10, "last_request": 0}
}
DEFAULT_DEEP_RESEARCH_MODEL = "gemini-2.0-flash"


def rate_limit_model(model_name):
    if model_name not in deep_research_rate_limits:
        return
    rate_limit_data = deep_research_rate_limits[model_name]
    now = time.time()
    time_since_last_request = now - rate_limit_data["last_request"]
    requests_per_minute = rate_limit_data["requests_per_minute"]
    wait_time = max(0, 60 / requests_per_minute - time_since_last_request)
    if wait_time > 0:
        logging.info(f"Rate limiting {model_name}, waiting for {wait_time:.2f} seconds")
        time.sleep(wait_time)
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

def get_shortened_url(url):
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            url = "http://" + url
        tinyurl_api = f"https://tinyurl.com/api-create.php?url={quote_plus(url)}"
        response = requests.get(tinyurl_api, timeout=5)
        if response.status_code == 200:
            return response.text
        else:
            logging.warning(f"TinyURL API error {response.status_code} for URL: {url}")
            return url
    except requests.exceptions.RequestException as e:
        logging.error(f"Error shortening URL '{url}': {e}")
        return url
    except Exception as e:
        logging.error(f"Unexpected error shortening URL '{url}': {e}")
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

def scrape_search_engine(search_query, engine_name):
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
    google_url = f"https://www.google.com/search?q={quote_plus(search_query)}&num=20"
    try:
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(google_url, headers=headers, timeout=config.REQUEST_TIMEOUT)
        logging.info(f"Google Search Status Code: {response.status_code} for query: {search_query}")
        if response.status_code == 200:
            google_soup = BeautifulSoup(response.text, 'html.parser')
            for result in google_soup.find_all('div', class_='tF2Cxc'):
                link = result.find('a', href=True)
                if link:
                    href = link['href']
                    fixed_url = fix_url(href)
                    if fixed_url:
                        search_results.append(fixed_url)
        elif response.status_code == 429:
            logging.warning("Google rate limit hit (429).")
        else:
            logging.warning(f"Google search failed with status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping Google: {e}")
    return list(set(search_results))

def scrape_duckduckgo(search_query):
    search_results = []
    duck_url = f"https://html.duckduckgo.com/html/?q={quote_plus(search_query)}"
    try:
        response = requests.get(duck_url, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
        logging.info(f"DuckDuckGo Status Code: {response.status_code}")
        if response.status_code == 200:
            duck_soup = BeautifulSoup(response.text, 'html.parser')
            for a_tag in duck_soup.find_all('a', class_='result__a', href=True):
                href = a_tag['href']
                fixed_url = fix_url(urljoin("https://html.duckduckgo.com/", href))
                if fixed_url: search_results.append(fixed_url)
    except Exception as e:
        logging.error(f"Error scraping DuckDuckGo: {e}")
    return list(set(search_results))

def scrape_bing(search_query):
    search_results = []
    bing_url = f"https://www.bing.com/search?q={quote_plus(search_query)}"
    try:
        response = requests.get(bing_url, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
        logging.info(f"Bing Status Code: {response.status_code}")
        if response.status_code == 200:
            bing_soup = BeautifulSoup(response.text, 'html.parser')
            for li in bing_soup.find_all('li', class_='b_algo'):
                for a_tag in li.find_all('a', href=True):
                    href = a_tag['href']
                    fixed_url = fix_url(href)
                    if fixed_url: search_results.append(fixed_url)
    except Exception as e:
        logging.error(f"Error scraping Bing: {e}")
    return list(set(search_results))

def scrape_yahoo(search_query):
    search_results = []
    yahoo_url = f"https://search.yahoo.com/search?p={quote_plus(search_query)}"
    try:
        response = requests.get(yahoo_url, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
        logging.info(f"Yahoo Status Code: {response.status_code}")
        if response.status_code == 200:
            yahoo_soup = BeautifulSoup(response.text, 'html.parser')
            for div in yahoo_soup.find_all('div', class_=lambda x: x and x.startswith('dd')):
                for a_tag in div.find_all('a', href=True):
                    href = a_tag['href']
                    match = re.search(r'/RU=(.*?)/RK=', href)
                    if match:
                        try:
                            decoded_url = unquote(match.group(1))
                            fixed_url = fix_url(decoded_url)
                            if fixed_url: search_results.append(fixed_url)
                        except:
                            logging.warning(f"Error decoding Yahoo URL: {href}")
                    elif href:
                        fixed_url = fix_url(href)
                        if fixed_url: search_results.append(fixed_url)
    except Exception as e:
        logging.error(f"Error scraping Yahoo: {e}")
    return list(set(search_results))

def scrape_brave(search_query):
    search_results = []
    brave_url = f"https://search.brave.com/search?q={quote_plus(search_query)}"
    try:
        response = requests.get(brave_url, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
        logging.info(f"Brave Status Code: {response.status_code}")
        if response.status_code == 200:
            if response.headers.get('Content-Encoding') == 'br':
                try:
                    content = brotli.decompress(response.content)
                    brave_soup = BeautifulSoup(content, 'html.parser')
                except brotli.error as e:
                    logging.error(f"Error decoding Brotli content: {e}")
                    return []
            else:
                brave_soup = BeautifulSoup(response.text, 'html.parser')

            for a_tag in brave_soup.find_all('a', class_='result-title', href=True):
                href = a_tag['href']
                fixed_url = fix_url(href)
                if fixed_url: search_results.append(fixed_url)
    except Exception as e:
        logging.error(f"Error scraping Brave: {e}")
    return list(set(search_results))

def scrape_linkedin(search_query):
    search_results = []
    linkedin_url = f"https://www.linkedin.com/search/results/all/?keywords={quote_plus(search_query)}"
    try:
        response = requests.get(linkedin_url, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
        logging.info(f"LinkedIn Status Code: {response.status_code}")
        if response.status_code == 200:
            linkedin_soup = BeautifulSoup(response.text, 'html.parser')
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
                    logging.warning(f"Error processing LinkedIn result: {e}")
                    continue
    except Exception as e:
        logging.error(f"Error scraping LinkedIn: {e}")
    return search_results

def _decode_content(response):
    detected_encoding = chardet.detect(response.content)['encoding']
    if detected_encoding is None:
        logging.warning(f"Chardet failed. Using UTF-8.")
        detected_encoding = 'utf-8'
    logging.debug(f"Detected encoding: {detected_encoding}")
    try:
        return response.content.decode(detected_encoding, errors='replace')
    except UnicodeDecodeError:
        logging.warning(f"Decoding failed with {detected_encoding}. Trying UTF-8.")
        try: return response.content.decode('utf-8', errors='replace')
        except:
            logging.warning("Decoding failed. Using latin-1 (may cause data loss).")
            return response.content.decode('latin-1', errors='replace')

def fetch_page_content(url, snippet_length=None, extract_links=False, extract_emails=False):
    if snippet_length is None:
        snippet_length = config.SNIPPET_LENGTH
    content_snippets = []
    references = []
    extracted_data = {}

    # Caching Logic
    if config.CACHE_ENABLED:
        if url in config.CACHE:
            if time.time() - config.CACHE[url]['timestamp'] < config.CACHE_TIMEOUT:
                logging.info(f"Using cached content for: {url}")
                return config.CACHE[url]['content_snippets'], config.CACHE[url]['references'], config.CACHE[url]['extracted_data']
            else:
                logging.info(f"Cache expired for: {url}")
                del config.CACHE[url]

    try:
        response = requests.get(url, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
        logging.debug(f"Fetching page content status: {response.status_code} for: {url}")
        if response.status_code == 200:
            page_text = _decode_content(response)
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

                # Cache the results
                if config.CACHE_ENABLED:
                    config.CACHE[url] = {
                        'content_snippets': content_snippets,
                        'references': references,
                        'extracted_data': extracted_data,
                        'timestamp': time.time()
                    }


        elif response.status_code == 403:
            logging.warning(f"Access forbidden (403) for: {url}")
        else:
            logging.error(f"Failed to fetch: URL={url}, Status={response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error fetching {url}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: URL={url}, Error={e}")
    return content_snippets, references, extracted_data

def generate_alternative_queries(original_query):
    prompt = f"Suggest 3 refined search queries for '{original_query}', optimizing for broad and effective web results."
    parts = [{"role": "user", "parts": [{"text": prompt}]}]
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    try:
      response = model.generate_content(parts, safety_settings=safety_settings)
      return [q.strip() for q in response.text.split('\n') if q.strip()]
    except Exception as e:
       logging.error(f"Error generating alternative queries: {e}")
       return []

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception))
def generate_gemini_response(prompt, model_name="gemini-2.0-flash", response_format="markdown"):
    rate_limit_model(model_name)
    parts = [{"role": "user", "parts": [{"text": prompt}]}]
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
    model = genai.GenerativeModel(model_name=model_name)
    try:
        response = model.generate_content(parts, safety_settings=safety_settings)
        text_response = response.text
        if response_format == "json":
            try: return json.loads(text_response)
            except:
                logging.warning("Invalid JSON, returning raw text.")
                return {"error": "Invalid JSON", "raw_text": text_response}
        elif response_format == "csv":
            try:
                csv_data = io.StringIO(text_response)
                return list(csv.reader(csv_data, delimiter=',', quotechar='"'))
            except:
                logging.warning("Invalid CSV, returning raw text.")
                return {"error": "Invalid CSV", "raw_text": text_response}
        else:
            text_response = re.sub(r'\n+', '\n\n', text_response)
            text_response = re.sub(r'  +', ' ', text_response)
            return text_response.replace("```markdown", "").replace("```", "").strip()
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global conversation_history
    try:
        data = request.json
        user_message = data.get('message', '')
        image_data = data.get('image')
        custom_instruction = data.get('custom_instruction')
        model_name = data.get('model_name', 'gemini-2.0-flash')

        # Initialize or Update Chat
        if custom_instruction and len(conversation_history) == 0:
            model = genai.GenerativeModel(model_name=model_name)
            chat = model.start_chat(history=[
                {"role": "user", "parts": [{"text": custom_instruction}]},
                {"role": "model", "parts": ["Understood."]}
            ])
            conversation_history = chat.history
        else:
            model = genai.GenerativeModel(model_name=model_name)
            chat = model.start_chat(history=conversation_history)

        # Send Message and Get Response
        if image_data:
            image_part = process_base64_image(image_data)
            if image_part:
                image = Image.open(io.BytesIO(image_part['data']))
                response = chat.send_message([user_message, image], stream=False)
            else:
                return jsonify({"error": "Failed to process image"}), 400
        else:
            response = chat.send_message(user_message, stream=False)

        response_text = response.text
        response_text = re.sub(r'\n+', '\n\n', response_text)
        response_text = re.sub(r'  +', ' ', response_text)
        response_text = re.sub(r'^- ', '* ', response_text, flags=re.MULTILINE)
        response_text = response_text.replace("```markdown", "").replace("```", "").strip()

        # Update Conversation History
        conversation_history = chat.history

        # Serialize History for JSON Response
        def content_to_dict(content):
            return {
                "role": content.role,
                "parts": [part.text if hasattr(part, 'text') else str(part) for part in content.parts]
            }

        serialized_history = [content_to_dict(content) for content in conversation_history]

        return jsonify({"response": response_text, "history": serialized_history})

    except Exception as e:
        logging.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_history():
    global conversation_history
    conversation_history = []
    return jsonify({"message": "Cleared history."})

def process_in_chunks(search_results, search_query, prompt_prefix="", fetch_options=None):
    chunk_summaries = []
    references = []
    processed_tokens = 0
    current_chunk_content = []
    extracted_data_all = []
    if fetch_options is None:
        fetch_options = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_page_content, url, config.DEEP_RESEARCH_SNIPPET_LENGTH, **fetch_options): url for url in search_results}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                page_snippets, page_refs, extracted_data = future.result()
                references.extend(page_refs)
                extracted_data_all.append({'url': url, 'data': extracted_data})
                for snippet in page_snippets:
                    estimated_tokens = len(snippet) // 4
                    if processed_tokens + estimated_tokens > config.MAX_TOKENS_PER_CHUNK:
                        combined_content = "\n\n".join(current_chunk_content)
                        if combined_content.strip():
                            summary_prompt = f"{prompt_prefix}\n\n{combined_content}"
                            summary = generate_gemini_response(summary_prompt)
                            chunk_summaries.append(summary)
                        current_chunk_content = []
                        processed_tokens = 0
                    current_chunk_content.append(snippet)
                    processed_tokens += estimated_tokens
            except Exception as e:
                logging.error(f"Error processing {url}: {e}")
                continue
        if current_chunk_content:
            combined_content = "\n\n".join(current_chunk_content)
            if combined_content.strip():
                summary_prompt = f"{prompt_prefix}\n\n{combined_content}"
                summary = generate_gemini_response(summary_prompt)
                chunk_summaries.append(summary)
    return chunk_summaries, references, extracted_data_all

@app.route('/api/online', methods=['POST'])
def online_search_endpoint():
    try:
        data = request.json
        search_query = data.get('query', '')
        if not search_query:
            return jsonify({"error": "No query"}), 400
        references = []
        search_results = []
        content_snippets = []
        search_engines_requested = data.get('search_engines', config.SEARCH_ENGINES)
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            search_futures = [executor.submit(scrape_search_engine, search_query, engine) for engine in search_engines_requested]
            for future in concurrent.futures.as_completed(search_futures):
                try:
                    search_results.extend(future.result())
                except Exception as e:
                    logging.error(f"Search engine scrape error: {e}")
            if not search_results:
                logging.warning(f"Initial search failed: {search_query}. Trying alternatives.")
                alternative_queries = generate_alternative_queries(search_query)
                if alternative_queries:
                    logging.info(f"Alternative queries: {alternative_queries}")
                    for alt_query in alternative_queries:
                        alt_search_futures = [executor.submit(scrape_search_engine, alt_query, engine) for engine in search_engines_requested]
                        for future in concurrent.futures.as_completed(alt_search_futures):
                            try:
                                result = future.result()
                                if result:
                                    search_results.extend(result)
                                    logging.info(f"Results found with alternative: {alt_query}")
                                    break
                            except Exception as e:
                                logging.error(f"Alternative query scrape error: {e}")
                        if search_results: break
                else:
                    logging.warning("Gemini failed to generate alternatives.")
            if not search_results:
                return jsonify({"error": "No results"}), 404
            unique_search_results = {url: 'general' for url in search_results}
            logging.debug(f"Unique URLs to fetch: {unique_search_results.keys()}")
            fetch_futures = {executor.submit(fetch_page_content, url): url for url in unique_search_results}
            for future in concurrent.futures.as_completed(fetch_futures):
                url = fetch_futures[future]
                try:
                    page_snippets, page_refs, _ = future.result()
                    content_snippets.extend(page_snippets)
                    references.extend(page_refs)
                except Exception as e:
                    logging.error(f"Error fetching {url}: {e}")
        combined_content = "\n\n".join(content_snippets)
        prompt = (f"Analyze web content for: '{search_query}'. Extract key facts, figures, and details. Be concise. Content:\n\n{combined_content}\n\nProvide a fact-based summary.")
        explanation = generate_gemini_response(prompt)
        global conversation_history
        conversation_history.append({"role": "user", "parts": [f"Online: {search_query}"]})
        conversation_history.append({"role": "model", "parts": [explanation]})

         # Shorten URLs concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            shortened_references = list(executor.map(get_shortened_url, references))

        return jsonify({"explanation": explanation, "references": shortened_references, "history": conversation_history})

    except Exception as e:
        logging.exception(f"Online search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/deep_research', methods=['POST'])
def deep_research_endpoint():
    try:
        data = request.json
        search_query = data.get('query', '')
        if not search_query:
            return jsonify({"error": "No query provided"}), 400

        model_name = data.get('model_name', DEFAULT_DEEP_RESEARCH_MODEL)

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

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            for iteration in range(max_iterations):
                logging.info(f"Iteration {iteration + 1}: {current_query}")
                search_results = []

                search_futures = [executor.submit(scrape_search_engine, current_query, engine) for engine in
                                  search_engines_requested]
                for future in concurrent.futures.as_completed(search_futures):
                    search_results.extend(future.result())

                unique_results = list(set(search_results))  # Remove duplicates
                logging.debug(f"Iteration {iteration + 1} - URLs: {unique_results}")

                prompt_prefix = (
                    f"Analyze snippets for: '{current_query}'. Extract key facts, figures, and insights. "
                    "Be concise, ignore irrelevant content, and prioritize authoritative sources. "
                    "Focus on the main topic and avoid discussing the research process itself.\n\nContent Snippets:"
                )
                fetch_options = {'extract_links': extract_links, 'extract_emails': extract_emails}

                chunk_summaries, refs, extracted = process_in_chunks(unique_results, current_query, prompt_prefix,
                                                                    fetch_options)
                all_summaries.extend(chunk_summaries)
                all_references.extend(refs)
                all_extracted_data.extend(extracted)

                if iteration < max_iterations - 1:
                    if all_summaries:
                        refinement_prompt = (
                            "Analyze the following research summaries to identify key themes and entities. "
                            "Suggest 3-5 new, more specific search queries that are *directly related* to the original topic. "
                            "Identify any gaps in the current research and suggest queries to address those gaps. "
                            "Do not suggest overly broad or generic queries. Focus on refining the search.\n\n"
                            "Research Summaries:\n" + "\n".join(all_summaries) + "\n\n"
                            "Provide refined search queries."
                        )

                        refined_response = generate_gemini_response(refinement_prompt, model_name=model_name)
                        new_queries = [q.strip() for q in refined_response.split('\n') if q.strip()]
                        current_query = " ".join(new_queries[:3])  # Use top 3 queries
                    else:
                        logging.info("No summaries for refinement. Skipping to next iteration.")
                        break

            # Final Report Generation
            if all_summaries:
                if "table" in data.get('output_format', '').lower():
                    table_prompt = generate_table_prompt(search_query)
                    final_prompt = (
                        f"{table_prompt}\n\n"
                        "Research Content:\n" + "\n\n".join(all_summaries) + "\n\n"
                                                "Generate the comparison table."
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

                final_explanation = generate_gemini_response(final_prompt, response_format=output_format, model_name=model_name)

                # Table Parsing and Fallback
                if "table" in output_format.lower():
                    try:
                        parsed_table = parse_markdown_table(final_explanation)
                        if parsed_table:
                            final_explanation = parsed_table
                        else:
                            logging.warning("Table parsing failed. Returning raw response.")
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
                # Generate PDF *synchronously* (ReportLab is synchronous)
                pdf_buffer = generate_pdf(search_query, final_explanation if isinstance(final_explanation, str)
                                        else "\n".join(str(row) for row in final_explanation), all_references)
                sanitized_filename = quote_plus(search_query)
                return send_file(
                    pdf_buffer,
                    as_attachment=True,
                    download_name=f"deep_research_{sanitized_filename}.pdf",
                    mimetype='application/pdf'
                )

            # JSON, CSV, and Markdown Responses
            response_data = {
                "explanation": final_explanation,
                "references": all_references,
                "history": conversation_history,
                "elapsed_time": f"{elapsed_time:.2f} seconds",
                "extracted_data": all_extracted_data,
            }

            if output_format == 'json':
                if isinstance(final_explanation, dict):
                    response_data = final_explanation
                elif isinstance(final_explanation, list):
                    response_data = {"table_data": final_explanation}
                else:
                    response_data = {"explanation": final_explanation}
                response_data.update({
                    "references": all_references,
                    "history": conversation_history,
                    "elapsed_time": f"{elapsed_time:.2f} seconds",
                    "extracted_data": all_extracted_data
                })
                return jsonify(response_data)

            elif output_format == 'csv':
                if isinstance(final_explanation, list):
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerows(final_explanation)
                    response_data["explanation"] = output.getvalue()
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
        logging.exception(f"Error in deep research: {e}")
        return jsonify({"error": str(e)}), 500

def generate_table_prompt(query):
    """Generates a prompt for creating a comparison table."""
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
    """Parses a Markdown table string with improved robustness."""
    lines = [line.strip() for line in markdown_table_string.split('\n') if line.strip()]
    if not lines:
        return []

    table_data = []
    header_detected = False

    for line in lines:
        line = line.strip().strip('|').replace(' | ', '|').replace('| ', '|').replace(' |', '|')
        cells = [cell.strip() for cell in line.split('|')]

        if all(c in '-:| ' for c in line) and len(cells) > 1 and not header_detected:
            header_detected = True
            continue

        if cells:
            table_data.append(cells)

    if table_data:
      max_cols = len(table_data[0])
      normalized_data = []
      for row in table_data:
          normalized_data.append(row + [''] * (max_cols - len(row)))
      return normalized_data
    else:
      return []

def generate_pdf(report_title, content, references):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                          rightMargin=0.7*inch, leftMargin=0.7*inch,
                          topMargin=0.7*inch, bottomMargin=0.7*inch)
    styles = getSampleStyleSheet()
    today = date.today()
    formatted_date = today.strftime("%B %d, %Y")

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
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)
        text = text.replace('&', '&').replace('<', '<').replace('>', '>')
        return text.strip()

    def process_table(table_text):
        rows = [row.strip() for row in table_text.split('\n') if row.strip()]
        if len(rows) < 2:
            return None

        header = [clean_text(cell) for cell in rows[0].strip('|').split('|')]
        data_rows = []
        for row in rows[2:]:
            cells = [clean_text(cell) for cell in row.strip('|').split('|')]
            data_rows.append(cells)

        table_data = [[Paragraph(cell, custom_styles['TableCell']) for cell in header]]
        for row in data_rows:
            table_data.append([Paragraph(cell, custom_styles['TableCell']) for cell in row])

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
    story.append(Paragraph(report_title, custom_styles['Title']))
    story.append(Paragraph(formatted_date, custom_styles['Footer']))
    story.append(Spacer(1, 0.3*inch))

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

        if '|' in line and (line.count('|') > 1 or (i + 1 < len(lines) and '|' in lines[i + 1])):
            in_table = True
            current_table.append(line)
        elif in_table:
            if current_table:
                table = process_table('\n'.join(current_table))
                if table:
                    story.append(table)
                    story.append(Spacer(1, 0.2*inch))
            current_table = []
            in_table = False
            continue
        elif line.startswith('# '):
            story.append(Paragraph(clean_text(line[2:]), custom_styles['Heading1']))
        elif line.startswith('## '):
            story.append(Paragraph(clean_text(line[3:]), custom_styles['Heading2']))
        elif line.startswith('* ') or line.startswith('- '):
            story.append(Paragraph(f"• {clean_text(line[2:])}", custom_styles['Bullet']))
        else:
            if not in_table:
                story.append(Paragraph(clean_text(line), custom_styles['Paragraph']))

        i += 1

    if current_table:
        table = process_table('\n'.join(current_table))
        if table:
            story.append(table)
            story.append(Spacer(1, 0.2*inch))

    if references:
        story.append(PageBreak())
        story.append(Paragraph("References", custom_styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        for i, ref in enumerate(references, 1):
            story.append(Paragraph(f"[{i}] {ref}", custom_styles['Reference']))

    story.append(Spacer(1, 0.5*inch))
    footer_text = f"Generated by Kv - AI Companion & Deep Research Tool • {formatted_date}"
    story.append(Paragraph(footer_text, custom_styles['Footer']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# --- Product Scraping ---
def scrape_product_details(url):
    """Scrapes product details from a given URL."""
    try:
        response = requests.get(url, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()  # Raises HTTPError for bad requests (4xx or 5xx)

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract Product Information (Highly adaptable)
        product_data = {}

        # Title (Adapt these selectors to common patterns)
        title_tags = ['h1', 'h2', 'span', 'div']
        title_classes = ['product-title', 'title', 'productName', 'product-name']
        for tag in title_tags:
            for class_name in title_classes:
                title_element = soup.find(tag, class_=class_name)
                if title_element:
                    product_data['title'] = title_element.get_text(strip=True)
                    break
            if 'title' in product_data:
                break

        # Price
        price_tags = ['span', 'div', 'p']
        price_classes = ['price', 'product-price', 'sales-price', 'regular-price']
        for tag in price_tags:
            for class_name in price_classes:
                price_element = soup.find(tag, class_=class_name)
                if price_element:
                    product_data['price'] = price_element.get_text(strip=True)
                    break
            if 'price' in product_data:
                break

        # Description (Look for structured data, then paragraphs)
        description_element = soup.find('div', {'itemprop': 'description'})
        if description_element:
             product_data['description'] = description_element.get_text(strip=True)
        else:
            # Look for common description containers
            description_classes = ['description', 'product-description', 'product-details', 'details']
            for class_name in description_classes:
                desc_element = soup.find(['div', 'p'], class_=class_name)
                if desc_element:
                    product_data['description'] = desc_element.get_text(separator='\n', strip=True)
                    break

        # Image (Prioritize schema.org, then common tags/classes)
        image_element = soup.find('img', {'itemprop': 'image'})
        if image_element:
            product_data['image_url'] = urljoin(url, image_element['src'])
        else:
            image_tags = ['img', 'div']
            image_classes = ['product-image', 'image', 'main-image', 'productImage']
            for tag in image_tags:
                for class_name in image_classes:
                    image_element = soup.find(tag, class_=class_name)
                    if image_element and image_element.get('src'):
                        product_data['image_url'] = urljoin(url, image_element['src'])
                        break
                if 'image_url' in product_data:
                    break

        # Additional Data (Ratings, Reviews, Availability - Add as needed)
        # Example: Rating (Adapt to common patterns)
        rating_element = soup.find(['span', 'div'], class_=['rating', 'star-rating', 'product-rating'])
        if rating_element:
            product_data['rating'] = rating_element.get_text(strip=True)

        return product_data

    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping product details from {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error scraping {url}: {e}")
        return None

@app.route('/api/scrape_product', methods=['POST'])
def scrape_product_endpoint():
    try:
        data = request.json
        product_query = data.get('query', '')
        if not product_query:
            return jsonify({"error": "No product query provided"}), 400

        # Use multiple search engines and refine results
        search_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = [executor.submit(scrape_search_engine, product_query, engine)
                       for engine in config.SEARCH_ENGINES]
            for future in concurrent.futures.as_completed(futures):
                try:
                    search_results.extend(future.result())
                except Exception as e:
                    logging.error(f"Error in search engine scrape: {e}")

        unique_urls = list(set(search_results))

        # Scrape details from each URL, handling None results
        all_product_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = {executor.submit(scrape_product_details, url): url for url in unique_urls}
            for future in concurrent.futures.as_completed(futures):
                try:
                    product_data = future.result()
                    if product_data:
                        all_product_data.append(product_data)
                except Exception as e:
                    url = futures[future]
                    logging.error(f"Error processing {url}: {e}")

        # Use Gemini to summarize or analyze the scraped data
        if all_product_data:
            # Create a prompt for Gemini
            prompt = "Summarize the following product information:\n\n"
            for product in all_product_data:
                prompt += f"- Title: {product.get('title', 'N/A')}\n"
                prompt += f"  Price: {product.get('price', 'N/A')}\n"
                prompt += f"  Description: {product.get('description', 'N/A')}\n"
                # Add other fields as necessary
                prompt += "\n"

            prompt += "\nProvide a concise summary, including key features and price range."

            summary = generate_gemini_response(prompt)

            return jsonify({"summary": summary, "products": all_product_data})
        else:
            return jsonify({"error": "No product information found"}), 404

    except Exception as e:
        logging.error(f"Error in product scraping endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# --- Job Scraping ---
def extract_text_from_resume(resume_data):
    """Extracts text from a resume (PDF, DOCX, or plain text)."""
    try:
        if resume_data.startswith(b"%PDF"):  # Check for PDF
            resume_text = pdf_extract_text(io.BytesIO(resume_data))
        elif resume_data.startswith(b"PK\x03\x04"): # Check for docx
            # DOCX extraction (using docx2txt - install if needed)
            resume_text = docx2txt.process(io.BytesIO(resume_data))
        else: # Assume plain text.  Decode.
            try:
                resume_text = resume_data.decode('utf-8')
            except UnicodeDecodeError:
                resume_text = resume_data.decode('latin-1', errors='replace') # Fallback
        return resume_text
    except Exception as e:
        logging.error(f"Error extracting resume text: {e}")
        return ""

def scrape_linkedin_jobs(job_title, job_location, resume_text=None):
    search_results = []
    base_url = "https://www.linkedin.com/jobs/search"
    params = {
        'keywords': job_title,
        'location': job_location,
        'f_TPR': 'r86400' # Past 24 hours
    }

    try:
        response = requests.get(base_url, params=params, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find Job Listings (Adapt selectors as needed)
        for job_card in soup.find_all('div', class_='base-card'): # Common container
            try:
                # Basic Information
                job_url_element = job_card.find('a', class_='base-card__full-link')
                job_url = job_url_element['href'] if job_url_element else None
                if not job_url:
                    continue

                title_element = job_card.find('h3', class_='base-search-card__title')
                title = title_element.get_text(strip=True) if title_element else "N/A"

                company_element = job_card.find('h4', class_='base-search-card__subtitle')
                company = company_element.get_text(strip=True) if company_element else "N/A"

                location_element = job_card.find('span', class_='job-search-card__location')
                location = location_element.get_text(strip=True) if location_element else "N/A"

                # Extract and match job description (if resume provided)
                if resume_text and job_url:
                    try:
                        job_response = requests.get(job_url, headers={'User-Agent': get_random_user_agent()},timeout=config.REQUEST_TIMEOUT)
                        job_response.raise_for_status()
                        job_soup = BeautifulSoup(job_response.text, 'html.parser')
                        # More robust description extraction
                        description_element = job_soup.find('div', class_='description__text')
                        if description_element:
                            job_description = description_element.get_text(separator='\n', strip=True)
                        else:
                            job_description = ""

                        # Relevance Check with Gemini
                        relevance_prompt = (
                            f"Assess the relevance of this job description to the provided resume.  "
                            f"Job Description:\n{job_description}\n\nResume:\n{resume_text}\n\n"
                            "Provide a brief assessment (e.g., 'Highly Relevant', 'Somewhat Relevant', 'Not Relevant') "
                            "and a short justification (1-2 sentences)."
                        )
                        relevance_assessment = generate_gemini_response(relevance_prompt)
                        job_data = {
                            'url': job_url,
                            'title': title,
                            'company': company,
                            'location': location,
                            'relevance': relevance_assessment
                        }
                        search_results.append(job_data)

                    except requests.exceptions.RequestException as e:
                        logging.warning(f"Failed to fetch job description from {job_url}: {e}")
                        continue
                else:
                    job_data = {
                        'url': job_url,
                        'title': title,
                        'company': company,
                        'location': location,
                        'relevance': 'N/A'
                    }
                    search_results.append(job_data)

            except Exception as e:
                logging.warning(f"Error processing a LinkedIn job card: {e}")
                continue

    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping LinkedIn Jobs: {e}")
    return search_results

def scrape_indeed_jobs(job_title, job_location, resume_text=None):
    """Scrapes jobs from Indeed.com."""
    search_results = []
    base_url = "https://www.indeed.com/jobs"
    params = {
        'q': job_title,
        'l': job_location,
        'sort': 'date'  # Sort by date for fresher results
    }

    try:
        response = requests.get(base_url, params=params, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # --- Find Job Cards (Indeed's structure changes frequently, adjust as needed) ---
        for job_card in soup.find_all('div', class_=lambda x: x and x.startswith('job_')):
            try:
                # --- Basic Job Info ---
                title_element = job_card.find(['h2','a'], class_=lambda x: x and ('title' in x or 'jobtitle' in x))
                title = title_element.get_text(strip=True) if title_element else "N/A"

                company_element = job_card.find(['span', 'a'], class_='companyName')
                company = company_element.get_text(strip=True) if company_element else "N/A"

                location_element = job_card.find('div', class_='companyLocation')
                location = location_element.get_text(strip=True) if location_element else "N/A"

                # ---  URL (Indeed uses different methods; handle both) ---
                job_url = None
                # Method 1:  Direct link in 'a' tag
                link_element = job_card.find('a', href=True)
                if link_element and 'pagead' not in link_element['href']:  # Exclude sponsored links if possible
                   job_url = urljoin(base_url, link_element['href'])

                # Method 2:  'data-jk' attribute (often used)
                if not job_url:
                    data_jk = job_card.get('data-jk')
                    if data_jk:
                        job_url = f"https://www.indeed.com/viewjob?jk={data_jk}"

                if not job_url:
                    continue  # Skip this job if no URL

                 # --- Relevance Check (if resume is provided) ---
                if resume_text and job_url:
                    try:
                        job_response = requests.get(job_url, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
                        job_response.raise_for_status()
                        job_soup = BeautifulSoup(job_response.text, 'html.parser')

                        # --- Job Description Extraction (Indeed is also tricky here) ---
                        description_element = job_soup.find('div', id='jobDescriptionText') # ID is often reliable
                        if description_element:
                            job_description = description_element.get_text(separator='\n', strip=True)
                        else:
                            job_description = ""

                        # --- Gemini Relevance Assessment ---
                        relevance_prompt = (
                            f"Assess relevance (Highly, Somewhat, Not): Job Description:\n{job_description}\n\nResume:\n{resume_text}\n\n"
                            "Justification (1-2 sentences)."
                        )
                        relevance_assessment = generate_gemini_response(relevance_prompt)

                        job_data = {
                            'url': job_url,
                            'title': title,
                            'company': company,
                                                        'location': location,
                            'relevance': relevance_assessment  # Store assessment
                        }
                        search_results.append(job_data)

                    except requests.exceptions.RequestException as e:
                        logging.warning(f"Failed to fetch Indeed job description from {job_url}: {e}")
                        continue  # Move to next job

                else:  # No resume, store basic info
                    job_data = {
                        'url': job_url,
                        'title': title,
                        'company': company,
                        'location': location,
                        'relevance': 'N/A'  # No resume
                    }
                    search_results.append(job_data)

            except Exception as e:
                logging.warning(f"Error processing an Indeed job card: {e}")
                continue # One bad card doesn't stop the process.

    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping Indeed Jobs: {e}")
    return search_results


@app.route('/api/scrape_jobs', methods=['POST'])
def scrape_jobs_endpoint():
    try:
        data = request.json
        job_title = data.get('job_title', '')
        job_location = data.get('job_location', '')
        resume_base64 = data.get('resume')

        if not job_title:
            return jsonify({"error": "No job title provided"}), 400

        resume_text = None
        if resume_base64:
            try:
                if 'base64,' in resume_base64:
                    resume_base64 = resume_base64.split('base64,')[1]
                resume_data = base64.b64decode(resume_base64)
                resume_text = extract_text_from_resume(resume_data)
            except Exception as e:
                logging.error(f"Error processing resume: {e}")
                return jsonify({"error": "Failed to process resume"}), 400

        # Use multiple job search engines
        all_job_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            linkedin_future = executor.submit(scrape_linkedin_jobs, job_title, job_location, resume_text)
            indeed_future = executor.submit(scrape_indeed_jobs, job_title, job_location, resume_text) # Add Indeed

            # Collect results, handling potential errors
            try:
                all_job_results.extend(linkedin_future.result())
            except Exception as e:
                 logging.error(f"Error scraping LinkedIn: {e}")
            try:
                all_job_results.extend(indeed_future.result()) # Add Indeed results
            except Exception as e:
                logging.error(f"Error scraping Indeed: {e}")


        if all_job_results:
            # Use Gemini to summarize, filter, or further process
            prompt = "Summarize the following job search results:\n\n"
            for job in all_job_results:
                prompt += f"- Title: {job.get('title', 'N/A')}, Company: {job.get('company', 'N/A')}, Location: {job.get('location', 'N/A')}, Relevance: {job.get('relevance', 'N/A')}\n"
            summary = generate_gemini_response(prompt)

            return jsonify({"summary": summary, 'jobs': all_job_results})

        else:
            return jsonify({"error": "No job listings found"}), 404

    except Exception as e:
        logging.error(f"Error in jobs scraping endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# Image Analysis Tool
@app.route('/api/analyze_image', methods=['POST'])
def analyze_image_endpoint():
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        image_part = process_base64_image(image_data)
        if not image_part:
             return jsonify({"error": "Failed to process image"}), 400

        model = genai.GenerativeModel('gemini-pro-vision')
        image = Image.open(io.BytesIO(image_part['data']))
        response = model.generate_content(["Describe this image in detail", image]) # Simple description prompt
        response.resolve()

        return jsonify({"description": response.text})

    except Exception as e:
        logging.exception("Error in image analysis")
        return jsonify({"error": "Image analysis failed."}), 500


# Sentiment Analysis Tool
@app.route('/api/analyze_sentiment', methods=['POST'])
def analyze_sentiment_endpoint():
    try:
        data = request.json
        text = data.get('text')
        if not text:
            return jsonify({"error": "No text provided"}), 400

        prompt = f"Analyze the sentiment of the following text and classify it as 'Positive', 'Negative', or 'Neutral'. Provide a brief justification:\n\n{text}"
        sentiment_result = generate_gemini_response(prompt)

        return jsonify({"sentiment": sentiment_result})
    except Exception as e:
        logging.exception("Error in sentiment analysis")
        return jsonify({"error": "Sentiment analysis failed."}), 500

# Website Summarization Tool
@app.route('/api/summarize_website', methods=['POST'])
def summarize_website_endpoint():
    try:
        data = request.json
        url = data.get('url')
        if not url:
            return jsonify({"error": "No URL provided"}), 400

        # Reuse existing functions!
        content_snippets, _, _ = fetch_page_content(url, snippet_length=config.SNIPPET_LENGTH)
        if not content_snippets:
            return jsonify({"error": "Could not fetch website content"}), 400

        combined_content = "\n\n".join(content_snippets)
        prompt = f"Summarize the following webpage content concisely:\n\n{combined_content}"
        summary = generate_gemini_response(prompt)

        return jsonify({"summary": summary})

    except Exception as e:
        logging.exception("Error in website summarization")
        return jsonify({"error": "Website summarization failed."}), 500



if __name__ == '__main__':
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"An error occurred: {e}")
