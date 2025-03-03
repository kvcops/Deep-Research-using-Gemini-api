# app.py (Complete, Modified)

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Tuple, Optional, Union, Any
import google.generativeai as genai
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import io
import requests
from bs4 import BeautifulSoup, SoupStrainer
import re
import random
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import time
from urllib.parse import urlparse, urljoin, quote_plus, unquote
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import HexColor, black, lightgrey
import csv
from datetime import date, datetime
import math
import concurrent.futures
import brotli
from pdfminer.high_level import extract_text as pdf_extract_text
import docx2txt
import chardet
import asyncio


load_dotenv()

app = FastAPI()

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


templates = Jinja2Templates(directory="templates")

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
        'Mozilla/5.0 (iPad; CPU OS 14_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
    ]
    SEARCH_ENGINES = ["google", "duckduckgo", "bing", "yahoo", "brave", "linkedin"]
    JOB_SEARCH_ENGINES = ["linkedin", "indeed", "glassdoor"]
    MAX_WORKERS = 10
    CACHE_ENABLED = True
    CACHE = {}
    CACHE_TIMEOUT = 300
    INDEED_BASE_DELAY = 2
    INDEED_MAX_DELAY = 10
    INDEED_RETRIES = 5
    JOB_RELEVANCE_MODEL = os.getenv("JOB_RELEVANCE_MODEL", "gemini-2.0-flash")
    # REMOVED PDF PAGE LIMIT

    # ---  Prompts:  More Modular and Specific ---
    DEEP_RESEARCH_TABLE_PROMPT = (
        "Create a detailed comparison table analyzing: '{query}'.\n\n"
        "**Strict Table Formatting:**\n"
        "*   **Markdown table ONLY.**\n"
        "*   **Structure:** Header row, separator row (---), data rows.\n"
        "*   **Rows:** Start and end with a pipe (|), spaces around pipes.\n"
        "*   **Separator:** Three dashes (---) per column, alignment colons (:---:).\n"
        "*   **Cells:** Concise (max 2-3 lines), consistent capitalization, 'N/A' for empty.\n"
        "*   **NO line breaks within cells.** Use <br> for internal line breaks if absolutely necessary.\n"
        "**Content Guidelines:**\n"
        "*   3-5 relevant columns.\n"
        "*   4-8 data rows.\n"
        "*   Proper alignment (usually center or left).\n"
        "*   Verify all pipe and spacing rules.\n"
        "*   **Output ONLY the table, NO extra text.**"
    )
    DEEP_RESEARCH_REFINEMENT_PROMPT = (
        "Analyze the following research summaries to identify key themes and entities. "
        "Suggest 3-5 new, more specific search queries that are *directly related* to the original topic: '{original_query}'. "
        "Identify any gaps in the current research and suggest queries to address those gaps. "
        "Do not suggest overly broad or generic queries. Focus on refining the search and addressing specific aspects. "
        "Prioritize queries that are likely to yield *different* results than the previous searches."
    )
    DEEP_RESEARCH_SUMMARY_PROMPT = (
         "Analyze snippets for: '{query}'. Extract key facts, figures, and insights. "
        "Be concise, ignore irrelevant content, and prioritize authoritative sources. "
        "Focus on the main topic and avoid discussing the research process itself.\n\nContent Snippets:"
    )

    DEEP_RESEARCH_REPORT_PROMPT = (
        "DEEP RESEARCH REPORT: Synthesize a comprehensive report from web research on: '{search_query}'.\n\n"
        "{report_structure}\n\n"
        "Research Summaries (all iterations):\n{summaries}\n\n"
        "Generate the report in Markdown."
    )

config = Config()

# --- Logging Configuration ---
logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Gemini Configuration ---
genai.configure(api_key=config.API_KEY)
if not config.API_KEY:
    logging.error("GEMINI_API_KEY not set. Exiting.")
    exit(1)

conversation_history = []
deep_research_rate_limits = {
    "gemini-2.0-flash": {"requests_per_minute": 15, "last_request": 0},
    "gemini-2.0-flash-thinking-exp-01-21": {"requests_per_minute": 10, "last_request": 0}
}
DEFAULT_DEEP_RESEARCH_MODEL = "gemini-2.0-flash"

def rate_limit_model(model_name):
    if model_name in deep_research_rate_limits:
        rate_limit_data = deep_research_rate_limits[model_name]
        now = time.time()
        time_since_last_request = now - rate_limit_data["last_request"]
        requests_per_minute = rate_limit_data["requests_per_minute"]
        wait_time = max(0, 60 / requests_per_minute - time_since_last_request)
        if wait_time > 0:
            logging.info(f"Rate limiting {model_name}, waiting for {wait_time:.2f} seconds")
            time.sleep(wait_time)
        rate_limit_data["last_request"] = time.time()

_user_agents = config.USER_AGENTS

def get_random_user_agent():
    return random.choice(_user_agents)

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
            url = "http://" + url  # Add scheme if missing
        tinyurl_api = f"https://tinyurl.com/api-create.php?url={quote_plus(url)}"
        response = requests.get(tinyurl_api, timeout=5)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error shortening URL '{url}': {e}")
        return url  # Return original URL on error
    except Exception as e:
        logging.error(f"Unexpected error shortening URL '{url}': {e}")
        return url

def fix_url(url):
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            url = "https://" + url
            parsed = urlparse(url)  # Re-parse with scheme
        if not parsed.netloc:
            return None  # Invalid URL
        return url.split("?")[0] # Removes parameters
    except Exception:
        return None

def scrape_search_engine(search_query: str, engine_name: str) -> List[str]:
    """Scrapes search results from specified search engine."""
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

def scrape_google(search_query: str) -> List[str]:
    """Scrapes Google search results."""
    search_results = []
    google_url = f"https://www.google.com/search?q={quote_plus(search_query)}&num=20"
    try:
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(google_url, headers=headers, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        logging.info(f"Google Status Code: {response.status_code} for query: {search_query}")
        if response.status_code == 200:
            only_results = SoupStrainer('div', class_='tF2Cxc')
            google_soup = BeautifulSoup(response.text, 'html.parser', parse_only=only_results)
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
    return list(set(search_results))  # Remove duplicates

def scrape_duckduckgo(search_query: str) -> List[str]:
    """Scrapes DuckDuckGo search results."""
    search_results = []
    duck_url = f"https://html.duckduckgo.com/html/?q={quote_plus(search_query)}"
    try:
        response = requests.get(duck_url, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        logging.info(f"DuckDuckGo Status Code: {response.status_code}")
        if response.status_code == 200:
            only_results = SoupStrainer('a', class_='result__a')
            duck_soup = BeautifulSoup(response.text, 'html.parser', parse_only=only_results)
            for a_tag in duck_soup.find_all('a', class_='result__a', href=True):
                href = a_tag['href']
                fixed_url = fix_url(urljoin("https://html.duckduckgo.com/", href)) # Make absolute
                if fixed_url: search_results.append(fixed_url)
    except Exception as e:
        logging.error(f"Error scraping DuckDuckGo: {e}")
    return list(set(search_results))

def scrape_bing(search_query: str) -> List[str]:
    """Scrapes Bing search results."""
    search_results = []
    bing_url = f"https://www.bing.com/search?q={quote_plus(search_query)}"
    try:
        response = requests.get(bing_url, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        logging.info(f"Bing Status Code: {response.status_code}")
        if response.status_code == 200:
            only_results = SoupStrainer('li', class_='b_algo')
            bing_soup = BeautifulSoup(response.text, 'html.parser', parse_only=only_results)
            for li in bing_soup.find_all('li', class_='b_algo'):
                for a_tag in li.find_all('a', href=True):
                    href = a_tag['href']
                    fixed_url = fix_url(href) # Fix the URL
                    if fixed_url: search_results.append(fixed_url)
    except Exception as e:
        logging.error(f"Error scraping Bing: {e}")
    return list(set(search_results))

def scrape_yahoo(search_query: str) -> List[str]:
    """Scrapes Yahoo search results."""
    search_results = []
    yahoo_url = f"https://search.yahoo.com/search?p={quote_plus(search_query)}"
    try:
        response = requests.get(yahoo_url, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        logging.info(f"Yahoo Status Code: {response.status_code}")
        if response.status_code == 200:
            only_dd_divs = SoupStrainer('div', class_=lambda x: x and x.startswith('dd'))
            yahoo_soup = BeautifulSoup(response.text, 'html.parser', parse_only=only_dd_divs)
            for div in yahoo_soup.find_all('div', class_=lambda x: x and x.startswith('dd')):
                for a_tag in div.find_all('a', href=True):
                    href = a_tag['href']
                    match = re.search(r'/RU=(.*?)/RK=', href) # Yahoo uses a redirect
                    if match:
                        try:
                            decoded_url = unquote(match.group(1))
                            fixed_url = fix_url(decoded_url)  # Fix the URL
                            if fixed_url: search_results.append(fixed_url)
                        except:
                            logging.warning(f"Error decoding Yahoo URL: {href}")
                    elif href: # Sometimes the direct URL is present
                         fixed_url = fix_url(href)
                         if fixed_url: search_results.append(fixed_url)
    except Exception as e:
        logging.error(f"Error scraping Yahoo: {e}")
    return list(set(search_results))

def scrape_brave(search_query: str) -> List[str]:
    """Scrapes Brave search results."""
    search_results = []
    brave_url = f"https://search.brave.com/search?q={quote_plus(search_query)}"
    try:
        response = requests.get(brave_url, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        logging.info(f"Brave Status Code: {response.status_code}")

        if response.status_code == 200:
            if response.headers.get('Content-Encoding') == 'br':
                try:
                    content = brotli.decompress(response.content) # Decompress Brotli
                    only_links = SoupStrainer('a', class_='result-title')
                    brave_soup = BeautifulSoup(content, 'html.parser', parse_only=only_links)

                except brotli.error as e:
                    logging.error(f"Error decoding Brotli content: {e}")
                    return []
            else:
                only_links = SoupStrainer('a', class_='result-title')
                brave_soup = BeautifulSoup(response.text, 'html.parser', parse_only=only_links)

            for a_tag in brave_soup.find_all('a', class_='result-title', href=True):
                href = a_tag['href']
                fixed_url = fix_url(href)  # Fix URL
                if fixed_url:
                    search_results.append(fixed_url)

        elif response.status_code == 429:
            logging.warning("Brave rate limit hit (429).")
        else:
            logging.warning(f"Brave search failed with status code: {response.status_code}")

    except Exception as e:
        logging.error(f"Error scraping Brave: {e}")
    return list(set(search_results))

def scrape_linkedin(search_query: str) -> List[str]:
    """Scrapes LinkedIn search results (people primarily)."""
    search_results = []
    linkedin_url = f"https://www.linkedin.com/search/results/all/?keywords={quote_plus(search_query)}"
    try:
        response = requests.get(linkedin_url, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        logging.info(f"LinkedIn Status Code: {response.status_code}")
        if response.status_code == 200:
            only_results = SoupStrainer('div', class_='entity-result__item')
            linkedin_soup = BeautifulSoup(response.text, 'html.parser', parse_only=only_results)
            for result in linkedin_soup.find_all('div', class_='entity-result__item'):
                try:
                    link_tag = result.find('a', class_='app-aware-link')
                    if not link_tag or not link_tag.get('href'):
                        continue
                    profile_url = fix_url(link_tag.get('href'))  # Fix the URL
                    if not profile_url or '/in/' not in profile_url: # Check for profile
                        continue
                    #  check for company context
                    if " at " in search_query.lower():
                        context = search_query.lower().split(" at ")[1]  #  company name
                        name = result.find('span', class_='entity-result__title-text')
                        title_company = result.find('div', class_='entity-result__primary-subtitle')
                        combined_text = ""
                        if name:
                            combined_text += name.get_text(strip=True).lower() + " "
                        if title_company:
                            combined_text += title_company.get_text(strip=True).lower()
                        if context not in combined_text: # Check if the company matches
                            continue

                        search_results.append(profile_url)
                except Exception as e:
                    logging.warning(f"Error processing LinkedIn result: {e}")
                    continue
    except Exception as e:
        logging.error(f"Error scraping LinkedIn: {e}")
    return search_results  # No need to remove duplicates for LinkedIn


def _decode_content(response: requests.Response) -> str:
    """Decodes response content, handling different encodings."""
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

def fetch_page_content(url: str, snippet_length: Optional[int] = None,
                      extract_links: bool = False, extract_emails: bool = False) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Fetches content, handles caching, extracts data."""
    if snippet_length is None:
                snippet_length = config.SNIPPET_LENGTH
    content_snippets = []
    references = []
    extracted_data = {}

    if config.CACHE_ENABLED:
        if url in config.CACHE:
            if time.time() - config.CACHE[url]['timestamp'] < config.CACHE_TIMEOUT:
                logging.info(f"Using cached content for: {url}")
                return config.CACHE[url]['content_snippets'], config.CACHE[url]['references'], config.CACHE[url]['extracted_data']
            else:
                logging.info(f"Cache expired for: {url}")
                del config.CACHE[url]  # Remove expired entry

    try:
        response = requests.get(url, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status() # Raise HTTPError for bad responses
        logging.debug(f"Fetching page content status: {response.status_code} for: {url}")
        if response.status_code == 200:
            page_text = _decode_content(response)
            if page_text:
                page_soup = BeautifulSoup(page_text, 'html.parser')
                for script in page_soup(["script", "style"]):
                    script.decompose()  # Remove script and style tags

                text = page_soup.get_text(separator=' ', strip=True)
                text = re.sub(r'[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]', '', text) # Remove invalid unicode

                snippet = text[:snippet_length]
                title = page_soup.title.string if page_soup.title else url #get title
                formatted_snippet = f"### {title}\n\n{snippet}\n"  # Keep ### for titles
                content_snippets.append(formatted_snippet)
                references.append(url)

                if extract_links:
                    extracted_data['links'] = [a['href'] for a in page_soup.find_all('a', href=True) if a['href'] and not a['href'].startswith("#")]  # Simple link extraction
                if extract_emails:
                    extracted_data['emails'] = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", page_text)  # Basic email regex

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

def generate_alternative_queries(original_query: str) -> List[str]:
    """Generates alternative search queries using Gemini."""
    prompt = f"Suggest 3 refined search queries for '{original_query}', optimizing for broad and effective web results."
    parts = [{"role": "user", "parts": [{"text": prompt}]}]
    safety_settings = {  #  Gemini Pro safety settings
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
    model = genai.GenerativeModel(model_name="gemini-2.0-flash") #Using gemini-2.0-flash for generating alternative quires
    try:
      response = model.generate_content(parts, safety_settings=safety_settings)
      return [q.strip() for q in response.text.split('\n') if q.strip()] # returns refined prompts
    except Exception as e:
       logging.error(f"Error generating alternative queries: {e}")
       return []

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception))
def generate_gemini_response(prompt: str, model_name: str = "gemini-2.0-flash", response_format: str = "markdown") -> Union[str, Dict, List]:
    """Generates a response from Gemini, handling retries/formats."""

    if model_name not in deep_research_rate_limits: # No rate limit for job relevance model
        logging.info(f"Using model: {model_name}")
    else:
        rate_limit_model(model_name) # Rate limit the deep research model

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
            try:
                # More robust JSON parsing:  Handle leading/trailing text, comments.
                response_text_cleaned = re.sub(r"```json\n?|```|[\s]*//.*|[\s]*/\*[\s\S]*?\*/[\s]*", "", text_response).strip()
                return json.loads(response_text_cleaned)
            except json.JSONDecodeError as e:
                logging.warning(f"Invalid JSON, returning raw text. Error: {e}, Response: {text_response}")
                return {"error": "Invalid JSON", "raw_text": text_response}

        elif response_format == "csv":
            try:
                csv_data = io.StringIO(text_response)
                return list(csv.reader(csv_data, delimiter=',', quotechar='"'))
            except Exception as e:
                logging.warning(f"Invalid CSV, returning raw text. Error: {e}")
                return {"error": "Invalid CSV", "raw_text": text_response}
        else:  # Default to Markdown
            text_response = re.sub(r'\n+', '\n\n', text_response)  # Consistent newlines
            text_response = re.sub(r'  +', ' ', text_response)  # Single spaces
            return text_response.replace("```markdown", "").replace("```", "").strip()  # Remove Markdown fences
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        raise

# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
     return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    global conversation_history
    try:
        data = await request.json()
        user_message = data.get('message', '')
        image_data = data.get('image')
        custom_instruction = data.get('custom_instruction')
        model_name = data.get('model_name', 'gemini-2.0-flash')

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

        if image_data:
            image_part = process_base64_image(image_data)
            if image_part:
                # Correctly pass the image to the model
                response = chat.send_message([user_message, {"mime_type": image_part["mime_type"], "data": image_part["data"]}], stream=False)

            else:
                raise HTTPException(status_code=400, detail="Failed to process image")
        else:
            response = chat.send_message(user_message, stream=False)

        response_text = response.text
        response_text = re.sub(r'\n+', '\n\n', response_text)
        response_text = re.sub(r'  +', ' ', response_text)
        response_text = re.sub(r'^- ', '* ', response_text, flags=re.MULTILINE)  # Correct bullet points
        response_text = response_text.replace("```markdown", "").replace("```", "").strip()  # Remove markdown

        conversation_history = chat.history

        def content_to_dict(content):
            return {
                "role": content.role,
                "parts": [part.text if hasattr(part, 'text') else str(part) for part in content.parts]
            }
        serialized_history = [content_to_dict(content) for content in conversation_history]

        return JSONResponse({"response": response_text, "history": serialized_history})

    except Exception as e:
        logging.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear")
async def clear_history_endpoint():
    global conversation_history
    conversation_history = []
    return JSONResponse({"message": "Cleared history."})

def process_in_chunks(search_results: List[str], search_query: str, prompt_prefix: str = "",
                     fetch_options: Optional[Dict] = None) -> Tuple[List[str], List[str], List[Dict]]:
    """Processes search results in chunks, fetching/summarizing."""
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
                    estimated_tokens = len(snippet) // 4  # Estimate tokens
                    if processed_tokens + estimated_tokens > config.MAX_TOKENS_PER_CHUNK:
                        # Combine, summarize, and reset
                        combined_content = "\n\n".join(current_chunk_content)
                        if combined_content.strip():
                            summary_prompt = config.DEEP_RESEARCH_SUMMARY_PROMPT.format(query=search_query) + f"\n\n{combined_content}"
                            summary = generate_gemini_response(summary_prompt, model_name=DEFAULT_DEEP_RESEARCH_MODEL)
                            chunk_summaries.append(summary)
                        current_chunk_content = []
                        processed_tokens = 0

                    current_chunk_content.append(snippet)
                    processed_tokens += estimated_tokens

            except Exception as e:
                logging.error(f"Error processing {url}: {e}")
                continue

        # Process any remaining content
        if current_chunk_content:
            combined_content = "\n\n".join(current_chunk_content)
            if combined_content.strip():
                summary_prompt = config.DEEP_RESEARCH_SUMMARY_PROMPT.format(query=search_query) + f"\n\n{combined_content}"
                summary = generate_gemini_response(summary_prompt, model_name=DEFAULT_DEEP_RESEARCH_MODEL)
                chunk_summaries.append(summary)

    return chunk_summaries, references, extracted_data_all

@app.post("/api/online")
async def online_search_endpoint(request: Request):
    """Performs an online search and summarizes results."""
    try:
        data = await request.json()
        search_query = data.get('query', '')
        if not search_query:
             raise HTTPException(status_code=400, detail="No query provided")

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
                        alt_search_futures = [executor.submit(scrape_search_engine, alt_query, engine) for engine in
                                              search_engines_requested]
                        for future in concurrent.futures.as_completed(alt_search_futures):
                            try:
                                result = future.result()
                                if result:
                                    search_results.extend(result)
                                    logging.info(f"Results found with alternative: {alt_query}")
                                    break  # Stop on first result
                            except Exception as e:
                                logging.error(f"Alternative query scrape error: {e}")
                        if search_results:
                            break # Stop after finding results
                else:
                    logging.warning("Gemini failed to generate alternatives.")

            if not search_results:
                raise HTTPException(status_code=404, detail="No results found")

            unique_search_results = list(set(search_results))
            logging.debug(f"Unique URLs to fetch: {unique_search_results}")
            # Fetch content concurrently
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
        prompt = (f"Analyze web content for: '{search_query}'. Extract key facts, figures, and details. Be concise. "
                  f"Content:\n\n{combined_content}\n\nProvide a fact-based summary.")
        explanation = generate_gemini_response(prompt) #Default model
        global conversation_history  # Access global variable

        def serialize_content(content):
            if isinstance(content, list):
                return [serialize_content(item) for item in content]
            elif hasattr(content, 'role') and hasattr(content, 'parts'):
                return {
                    "role": content.role,
                    "parts": [part.text if hasattr(part, 'text') else str(part) for part in content.parts]
                }
            else:
                return content

        conversation_history.append({"role": "user", "parts": [f"Online: {search_query}"]})
        conversation_history.append({"role": "model", "parts": [explanation]})
        serialized_history = [serialize_content(item) for item in conversation_history]

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            shortened_references = list(executor.map(get_shortened_url, references)) #Shorten URLs concurrently


        return JSONResponse({"explanation": explanation, "references": shortened_references, "history": serialized_history})

    except HTTPException as e:
        raise e  # Re-raise HTTP exceptions
    except Exception as e:
        logging.exception(f"Online search error: {e}")  # Log with traceback
        raise HTTPException(status_code=500, detail=str(e))


def serialize_content(content): #helper function
    if isinstance(content, list):
        return [serialize_content(item) for item in content]
    elif hasattr(content, 'role') and hasattr(content, 'parts'):
        return {
            "role": content.role,
            "parts": [part.text if hasattr(part, 'text') else str(part) for part in content.parts]
        }
    else:
        return content


@app.post("/api/deep_research")
async def deep_research_endpoint(request: Request):
    try:
        data = await request.json()
        search_query = data.get('query', '')
        if not search_query:
            raise HTTPException(status_code=400, detail="No query provided")

        model_name = data.get('model_name', DEFAULT_DEEP_RESEARCH_MODEL)
        start_time = time.time()
        search_engines_requested = data.get('search_engines', config.SEARCH_ENGINES)
        output_format = data.get('output_format', 'markdown')  # Default to markdown
        extract_links = data.get('extract_links', False)
        extract_emails = data.get('extract_emails', False)
        download_pdf = data.get('download_pdf', True) # Default to True
        max_iterations = int(data.get('max_iterations', 3))  # Default to 3

        all_summaries = []
        all_references = []
        all_extracted_data = []
        current_query = search_query # initial query


        # ---  Enhanced Report Structure Logic ---
        report_structure = (
            "**Structure your report with clear headings and subheadings.**\n"
            "Use bullet points and numbered lists where appropriate.\n"
            "Include a concise introduction and conclusion.\n\n"
        )

        # Conditionally add table instructions
        if "table" in output_format.lower():
            report_structure += (
                "**Include a comparison table summarizing key findings.**  "
                "Use the detailed table formatting guidelines provided earlier.\n"
            )
            table_prompt = config.DEEP_RESEARCH_TABLE_PROMPT.format(query=search_query) #prepare the table prompt
        else:
            report_structure += "**Do NOT include a table.** Focus on a narrative report.\n"



        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            for iteration in range(max_iterations):
                logging.info(f"Iteration {iteration + 1}: {current_query}")
                search_results = []
                #current_query = search_query if iteration == 0 else current_query # To keep track of current
                search_futures = [executor.submit(scrape_search_engine, current_query, engine) for engine in
                                  search_engines_requested]
                for future in concurrent.futures.as_completed(search_futures):
                    search_results.extend(future.result())

                unique_results = list(set(search_results))  # Remove duplicates
                logging.debug(f"Iteration {iteration + 1} - URLs: {unique_results}")

                prompt_prefix = config.DEEP_RESEARCH_SUMMARY_PROMPT.format(query=current_query)
                fetch_options = {'extract_links': extract_links, 'extract_emails': extract_emails}

                chunk_summaries, refs, extracted = process_in_chunks(unique_results, current_query, prompt_prefix,
                                                                    fetch_options)
                all_summaries.extend(chunk_summaries)
                all_references.extend(refs)
                all_extracted_data.extend(extracted)

                if iteration < max_iterations - 1:
                    # Refine the search query
                    if all_summaries: # Check if we have any summaries to work with.
                        refinement_prompt = config.DEEP_RESEARCH_REFINEMENT_PROMPT.format(original_query=search_query) + "\n\nResearch Summaries:\n" + "\n".join(all_summaries)
                        refined_response = generate_gemini_response(refinement_prompt, model_name=model_name)
                        new_queries = [q.strip() for q in refined_response.split('\n') if q.strip()]
                        current_query = " ".join(new_queries[:3])  # Use top queries
                    else:
                        logging.info("No summaries for refinement. Skipping to next iteration.")
                        break # If no summaries, stop refining.


            # ---  Final Report Generation (with structure) ---
            if all_summaries:
                final_prompt = config.DEEP_RESEARCH_REPORT_PROMPT.format(
                    search_query=search_query,
                    report_structure=report_structure,
                    summaries="\n\n".join(all_summaries)
                )

                #  If table is requested, prepend the table prompt.
                if "table" in output_format.lower():
                    final_prompt = table_prompt + "\n\n" + final_prompt


                final_explanation = generate_gemini_response(final_prompt, response_format=output_format,
                                                            model_name=model_name)

                # --- Table Parsing (if applicable) ---
                if "table" in output_format.lower():
                    try:
                        parsed_table = parse_markdown_table(final_explanation)
                        if parsed_table:
                            final_explanation = parsed_table  # Use parsed table
                        else:
                            logging.warning("Table parsing failed. Returning raw response.")
                            final_explanation = {"error": "Failed to parse table", "raw_text": final_explanation}
                    except Exception as e:
                        logging.error(f"Error during table parsing: {e}")
                        final_explanation = {"error": "Failed to parse table", "raw_text": final_explanation}
            else:
                final_explanation = "No relevant content found for the given query."



            global conversation_history  # Access global variable
            conversation_history.append({"role": "user", "parts": [f"Deep research query: {search_query}"]})
            conversation_history.append({"role": "model", "parts": [final_explanation]})
            serialized_history = [serialize_content(item) for item in conversation_history]

            end_time = time.time()
            elapsed_time = end_time - start_time


            response_data = {
                "explanation": final_explanation,
                "references": all_references,
                "history": serialized_history,
                "elapsed_time": f"{elapsed_time:.2f} seconds",
                "extracted_data": all_extracted_data,
                "current_query": current_query,  # Include the final query used
                "iteration": iteration + 1  #  Include the final iteration number

            }
            if download_pdf:
                pdf_buffer = generate_pdf(
                    "",  # Pass an EMPTY STRING as the title.
                    final_explanation if isinstance(final_explanation, str)
                    else "\n".join(str(row) for row in final_explanation),
                    all_references
                )
                headers = {
                    'Content-Disposition': f'attachment; filename="{quote_plus(search_query)}_report.pdf"'
                }
                return StreamingResponse(iter([pdf_buffer.getvalue()]), media_type="application/pdf", headers=headers)


            if output_format == "json":
                if isinstance(final_explanation, dict):
                     # If it's already a dict (like error case), return it directly
                    response_data = final_explanation
                elif isinstance(final_explanation, list):
                    #  table data
                    response_data = {"table_data": final_explanation}
                else:
                    #  text explanation
                    response_data = {"explanation": final_explanation}
                # Add other data to the JSON response
                response_data.update({
                    "references": all_references,
                    "history": serialized_history,
                    "elapsed_time": f"{elapsed_time:.2f} seconds",
                    "extracted_data": all_extracted_data
                })
                return JSONResponse(response_data)


            elif output_format == "csv":
                if isinstance(final_explanation, list):
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerows(final_explanation)  # Write the list of lists
                    response_data["explanation"] = output.getvalue()
                elif isinstance(final_explanation, dict) and "raw_text" in final_explanation:
                    # Handle potential error dict
                    response_data = {"explanation": final_explanation["raw_text"]}
                else:
                    response_data = {"explanation": final_explanation} # for normal text

                response_data.update({
                    "references": all_references,
                    "history": serialized_history,
                    "elapsed_time": f"{elapsed_time:.2f} seconds",
                    "extracted_data": all_extracted_data
                })

                return JSONResponse(response_data)

            # If not JSON or CSV, return as is (Markdown)
            return JSONResponse(response_data)

    except HTTPException as e:
        raise e  # Re-raise HTTP exceptions
    except Exception as e:
        logging.exception(f"Error in deep research: {e}")  # Log full traceback
        raise HTTPException(status_code=500, detail=str(e))

def parse_markdown_table(markdown_table_string):
    """Parses a Markdown table string with improved robustness."""
    lines = [line.strip() for line in markdown_table_string.split('\n') if line.strip()]
    if not lines:
        return []

    table_data = []
    header_detected = False

    for line in lines:
        line = line.strip().strip('|').replace(' | ', '|').replace('| ', '|').replace(' |', '|')  # Normalize spacing
        cells = [cell.strip() for cell in line.split('|')]

        if all(c in '-:| ' for c in line) and len(cells) > 1 and not header_detected:
            # Skip header separator, but only *before* processing the first non-separator row
            header_detected = True
            continue

        if cells:
            table_data.append(cells)

    # Handle missing cells and inconsistent column counts.
    if table_data:
      max_cols = len(table_data[0])
      normalized_data = []
      for row in table_data:
          normalized_data.append(row + [''] * (max_cols - len(row)))  # Pad with empty strings
      return normalized_data
    else:
      return []

def generate_pdf(report_title, content, references):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                          rightMargin=0.7*inch, leftMargin=0.7*inch,
                          topMargin=0.7*inch, bottomMargin=0.7*inch)
    styles = getSampleStyleSheet()
    today = date.today()
    formatted_date = today.strftime("%B %d, %Y")

    # --- Custom Styles ---
    custom_styles = {
        'Title': ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            leading=32,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=HexColor("#1a237e"),
            fontName='Helvetica-Bold'
        ),
        'Heading1': ParagraphStyle(
            'CustomHeading1',
            parent=styles['Heading1'],
            fontSize=18,
            leading=24,
            spaceBefore=20,
            spaceAfter=12,
            textColor=HexColor("#283593"),
            fontName='Helvetica-Bold',
            keepWithNext=True  # Keep with the following paragraph
        ),
        'Heading2': ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontSize=16,
            leading=22,
            spaceBefore=16,
            spaceAfter=10,
            textColor=HexColor("#3949ab"),
            fontName='Helvetica-Bold',
            keepWithNext=True
        ),
        'Heading3': ParagraphStyle(
            'CustomHeading3',
            parent=styles['Heading3'],
            fontSize=14,
            leading=20,
            spaceBefore=14,
            spaceAfter=8,
            textColor=HexColor("#455a64"),
            fontName='Helvetica-Bold',
            keepWithNext=True
        ),
        'Paragraph': ParagraphStyle(
            'CustomParagraph',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            spaceAfter=10,
            alignment=TA_JUSTIFY,
            textColor=HexColor("#212121"),
            firstLineIndent=0.25*inch
        ),
        'TableCell': ParagraphStyle(
            'CustomTableCell',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            spaceBefore=4,
            spaceAfter=4,
            textColor=HexColor("#212121")
        ),
        'Bullet': ParagraphStyle(  # Corrected bullet style
            'CustomBullet',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            leftIndent=0.5*inch,
            rightIndent=0,
            spaceBefore=4,
            spaceAfter=4,
            bulletIndent=0.3*inch,
            textColor=HexColor("#212121"),
            bulletFontName='Helvetica',  # Ensure consistent font
            bulletFontSize=11
        ),
        'Reference': ParagraphStyle(
            'CustomReference',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            spaceAfter=4,
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
            spaceBefore=24  # Space above the footer
        )
    }

    def clean_text(text):
        # Convert common Markdown formatting to ReportLab equivalents
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)  # Italics
        text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)  # Inline code
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text) # Links
        # Escape HTML entities to prevent issues in Paragraph
        text = text.replace('&', '&').replace('<', '<').replace('>', '>')
        return text.strip()


    def process_table(table_text):
        rows = [row.strip() for row in table_text.split('\n') if row.strip()]
        if len(rows) < 2:
            return None  # Not a valid table

        header = [clean_text(cell) for cell in rows[0].strip('|').split('|')]
        data_rows = []
        for row in rows[2:]:  # Skip header and separator lines
            cells = [clean_text(cell) for cell in row.strip('|').split('|')]
            data_rows.append(cells)

        # Convert to ReportLab Paragraph objects
        table_data = [[Paragraph(cell, custom_styles['TableCell']) for cell in header]]
        for row in data_rows:
            table_data.append([Paragraph(cell, custom_styles['TableCell']) for cell in row])

        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#f5f5f5")),  # Header background
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#1a237e")),    # Header text color
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),               # Center-align all cells
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),     # Header font
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),                # Header padding
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 1, HexColor("#e0e0e0")),     # Grid
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f8f9fa")]),  # Alternating row colors
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Left-align cell content
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),      # Body font
            ('FONTSIZE', (0, 1), (-1, -1), 9),                  # Body font size
            ('TOPPADDING', (0, 1), (-1, -1), 6),     #  padding
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
             ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), # Vertically center cell content
        ])

        col_widths = [doc.width/len(header) for _ in header]  # Equal column widths
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(table_style)
        return table

    def footer(canvas, doc):
        canvas.saveState()
        footer_text = f"Generated by Kv - AI Companion & Deep Research Tool • {formatted_date}"
        footer = Paragraph(footer_text, custom_styles['Footer'])
        w, h = footer.wrap(doc.width, doc.bottomMargin)
        footer.drawOn(canvas, doc.leftMargin, h)  # Draw footer
        canvas.restoreState()

    story = []
    story.append(Paragraph(report_title, custom_styles['Title'])) # Title
    story.append(Paragraph(formatted_date, custom_styles['Footer']))  # Date
    story.append(Spacer(1, 0.2*inch))  # Initial space

    current_table = []
    in_table = False
    lines = content.split('\n')
    i = 0
    current_paragraph = [] # Accumulate lines for a paragraph

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            # Empty line:  End current paragraph (if any), add it to story.
            if current_paragraph:
                story.append(Paragraph(clean_text(" ".join(current_paragraph)), custom_styles['Paragraph']))
                current_paragraph = []
            # Handle any pending table
            if in_table and current_table:
                table = process_table('\n'.join(current_table))
                if table:
                    story.append(table)
                    story.append(Spacer(1, 0.1*inch))
                current_table = []
                in_table = False
            story.append(Spacer(1, 0.05*inch)) # Consistent spacing
            i += 1
            continue

        if '|' in line and (line.count('|') > 1 or (i + 1 < len(lines) and '|' in lines[i + 1])):
            # Likely a table row.  Start/continue accumulating table lines.
            in_table = True
            current_table.append(line)
            # End any current paragraph before starting a table.
            if current_paragraph:
                story.append(Paragraph(clean_text(" ".join(current_paragraph)), custom_styles['Paragraph']))
                current_paragraph = []
        elif in_table:
            # End of table. Process accumulated table lines.
            if current_table:
                table = process_table('\n'.join(current_table))
                if table:
                    story.append(table)
                    story.append(Spacer(1, 0.1*inch))
            current_table = []
            in_table = False
            continue #Crucial to continue here, and not add to current_paragraph below.

        elif line.startswith('# '):
             # End current paragraph (if any) before starting a heading.
            if current_paragraph:
                story.append(Paragraph(clean_text(" ".join(current_paragraph)), custom_styles['Paragraph']))
                current_paragraph = []
            story.append(Paragraph(clean_text(line[2:]), custom_styles['Heading1']))  # H1
        elif line.startswith('## '):
            if current_paragraph:
                story.append(Paragraph(clean_text(" ".join(current_paragraph)), custom_styles['Paragraph']))
                current_paragraph = []
            story.append(Paragraph(clean_text(line[3:]), custom_styles['Heading2']))  # H2
        elif line.startswith('### '):
            if current_paragraph:
                story.append(Paragraph(clean_text(" ".join(current_paragraph)), custom_styles['Paragraph']))
                current_paragraph = []
            story.append(Paragraph(clean_text(line[4:]), custom_styles['Heading3'])) # H3
        elif line.startswith('* ') or line.startswith('- '):
            # End current paragraph before a list item.
            if current_paragraph:
                story.append(Paragraph(clean_text(" ".join(current_paragraph)), custom_styles['Paragraph']))
                current_paragraph = []
            story.append(Paragraph(f"• {clean_text(line[2:])}", custom_styles['Bullet'])) # Bullet points

        else:
            # Regular text line: add to the current paragraph.
            current_paragraph.append(line)

        i += 1
    # Add any remaining paragraph content (important!).
    if current_paragraph:
        story.append(Paragraph(clean_text(" ".join(current_paragraph)), custom_styles['Paragraph']))

    if current_table:
        table = process_table('\n'.join(current_table))
        if table:
            story.append(table)
            story.append(Spacer(1, 0.1*inch))

    if references:
        story.append(PageBreak()) # References on a new page
        story.append(Paragraph("References", custom_styles['Heading1']))
        story.append(Spacer(1, 0.1*inch))
        for i, ref in enumerate(references, 1):
            story.append(Paragraph(f"[{i}] {ref}", custom_styles['Reference']))

    doc.build(story, onLaterPages=footer, onFirstPage=footer)  # Apply to all pages.
    buffer.seek(0)
    return buffer

# --- Product Scraping ---
def scrape_product_details(url):
    """Scrapes product details from a given URL."""
    try:
        response = requests.get(url, headers={'User-Agent': get_random_user_agent()}, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        product_data = {}
                # Title
        for tag in ['h1', 'h2', 'span', 'div']:
            for class_name in ['product-title', 'title', 'productName', 'product-name']:
                if title_element := soup.find(tag, class_=class_name):
                    product_data['title'] = title_element.get_text(strip=True)
                    break
            if 'title' in product_data:
                break

        # Price
        for tag in ['span', 'div', 'p']:
            for class_name in ['price', 'product-price', 'sales-price', 'regular-price']:
                if price_element := soup.find(tag, class_=class_name):
                    product_data['price'] = price_element.get_text(strip=True)
                    break
            if 'price' in product_data:
                break

        # Description
        if (description_element := soup.find('div', {'itemprop': 'description'})):
            product_data['description'] = description_element.get_text(strip=True)
        else:
            for class_name in ['description', 'product-description', 'product-details', 'details']:
                if desc_element := soup.find(['div', 'p'], class_=class_name):
                    product_data['description'] = desc_element.get_text(separator='\n', strip=True)
                    break

        # Image URL
        if (image_element := soup.find('img', {'itemprop': 'image'})):
            product_data['image_url'] = urljoin(url, image_element['src'])
        else:
            for tag in ['img', 'div']:
                for class_name in ['product-image', 'image', 'main-image', 'productImage']:
                    if (image_element := soup.find(tag, class_=class_name)) and image_element.get('src'):
                        product_data['image_url'] = urljoin(url, image_element['src'])
                        break
                if 'image_url' in product_data:
                    break

        # Rating
        if (rating_element := soup.find(['span', 'div'], class_=['rating', 'star-rating', 'product-rating'])):
            product_data['rating'] = rating_element.get_text(strip=True)

        return product_data

    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping product details from {url}: {e}")
        return None  # Return None on error
    except Exception as e:
        logging.error(f"Unexpected error scraping {url}: {e}")
        return None  # Return None on unexpected error

@app.post("/api/scrape_product")
async def scrape_product_endpoint(request: Request):
    try:
        data = await request.json()
        product_query = data.get('query', '')
        if not product_query:
             raise HTTPException(status_code=400, detail="No product query provided")

        search_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = [executor.submit(scrape_search_engine, product_query, engine)
                       for engine in config.SEARCH_ENGINES]
            for future in concurrent.futures.as_completed(futures):
                try:
                    search_results.extend(future.result())
                except Exception as e:
                    logging.error(f"Error in search engine scrape: {e}")

        unique_urls = list(set(search_results))  # Remove duplicate URLs

        all_product_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = {executor.submit(scrape_product_details, url): url for url in unique_urls}
            for future in concurrent.futures.as_completed(futures):
                try:
                    product_data = future.result()
                    if product_data:  # Only add if data was successfully scraped
                        all_product_data.append(product_data)
                except Exception as e:
                    url = futures[future]
                    logging.error(f"Error processing {url}: {e}")

        if all_product_data:
            # Create a prompt to summarize the product information
            prompt = "Summarize the following product information:\n\n"
            for product in all_product_data:
                prompt += f"- Title: {product.get('title', 'N/A')}\n"
                prompt += f"  Price: {product.get('price', 'N/A')}\n"
                prompt += f"  Description: {product.get('description', 'N/A')}\n"
                prompt += "\n"  # Add a separator between products

            prompt += "\nProvide a concise summary, including key features and price range."

            summary = generate_gemini_response(prompt) # default model

            return JSONResponse({"summary": summary, "products": all_product_data})
        else:
            raise HTTPException(status_code=404, detail="No product information found")

    except HTTPException as e:
        raise e  # Re-raise HTTP exceptions
    except Exception as e:
        logging.error(f"Error in product scraping endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Job Scraping ---
def extract_text_from_resume(resume_data: bytes) -> str:
    """Extracts text from a resume (PDF, DOCX, or plain text)."""
    try:
        if resume_data.startswith(b"%PDF"):
            # PDF file
            resume_text = pdf_extract_text(io.BytesIO(resume_data))
        elif resume_data.startswith(b"PK\x03\x04"):  # Common DOCX header
            # DOCX file
            resume_text = docx2txt.process(io.BytesIO(resume_data))
        else:
            # Assume plain text
            try:
                resume_text = resume_data.decode('utf-8')
            except UnicodeDecodeError:
                resume_text = resume_data.decode('latin-1', errors='replace') # Fallback encoding
        return resume_text
    except Exception as e:
        logging.error(f"Error extracting resume text: {e}")
        return ""

# --- Helper functions for job scraping ---

def linkedin_params(job_title: str, job_location: str, start: int = 0, experience_level: Optional[str] = None) -> Dict:
    """Generates parameters for a LinkedIn job search URL."""
    params = {
        'keywords': job_title,
        'location': job_location,
        'f_TPR': 'r86400',  # Past 24 hours (consider making configurable)
        'sortBy': 'R',  # Sort by relevance
        'start': start  #  pagination
    }
    if experience_level:
        # LinkedIn-specific experience level filters (add others as needed)
        if experience_level.lower() == "fresher":
            params['f_E'] = '1'  # Internship
        elif experience_level.lower() == "entry-level":
            params['f_E'] = '2' # Entry-Level
        elif experience_level.lower() == "mid-level":
          params['f_E'] = '3' # Associate
        elif experience_level.lower() == "senior":
            params['f_E'] = '4' # Senior
        elif experience_level.lower() == "executive":
            params['f_E'] = '5' # Director , '6' - executive
            # params['f_E'] = ['4', '5']  # Combine Senior/Executive for LinkedIn

    return params

def indeed_params(job_title: str, job_location: str, start: int = 0, experience_level: Optional[str] = None) -> Dict:
    """Generates parameters for an Indeed job search URL."""
    params = {
        'q': job_title,
        'l': job_location,
        'sort': 'relevance',  # Sort by relevance
        'fromage': '1',      # Past 24 hours
        'limit': 50,         # Fetch more results per page
        'start': start       # Pagination
    }
    if experience_level:
        params['q'] = f"{experience_level} {params['q']}" # Add to the main query

    return params
def parse_linkedin_job_card(job_card: BeautifulSoup) -> Dict:
    """Parses a single LinkedIn job card and extracts relevant information."""
    try:
        job_url_element = job_card.find('a', class_='base-card__full-link')
        job_url = job_url_element['href'] if job_url_element else None

        title_element = job_card.find('h3', class_='base-search-card__title')
        title = title_element.get_text(strip=True) if title_element else "N/A"

        company_element = job_card.find('h4', class_='base-search-card__subtitle')
        company = company_element.get_text(strip=True) if company_element else "N/A"

        location_element = job_card.find('span', class_='job-search-card__location')
        location = location_element.get_text(strip=True) if location_element else "N/A"

        return {
            'url': job_url,
            'title': title,
            'company': company,
            'location': location,
            'relevance': 0.0,
            'missing_skills': [],
            'justification': "Relevance not assessed.",
            'experience': 'N/A' # default value
        }
    except Exception as e:
        logging.error(f"Error parsing LinkedIn job card: {e}")
        return {  # Return defaults on error
            'url': None,
            'title': "N/A",
            'company': "N/A",
            'location': "N/A",
            'relevance': 0.0,
            'missing_skills': [],
            'justification': f"Error parsing job card: {type(e).__name__}",
            'experience': 'N/A'
        }

def parse_indeed_job_card(job_card: BeautifulSoup) -> Dict:
    """Parses a single Indeed job card and extracts relevant information."""
    try:
        title_element = job_card.find(['h2', 'a'], class_=lambda x: x and ('title' in x or 'jobtitle' in x))
        title = title_element.get_text(strip=True) if title_element else "N/A"

        company_element = job_card.find(['span', 'a'], class_='companyName')
        company = company_element.get_text(strip=True) if company_element else "N/A"

        location_element = job_card.find('div', class_='companyLocation')
        location = location_element.get_text(strip=True) if location_element else "N/A"

        job_url = None
        link_element = job_card.find('a', href=True)
        if link_element and 'pagead' not in link_element['href']:
            job_url = urljoin("https://www.indeed.com/jobs", link_element['href'])
        if not job_url:
            data_jk = job_card.get('data-jk')
            if data_jk:
                job_url = f"https://www.indeed.com/viewjob?jk={data_jk}"

        return {
            'url': job_url,
            'title': title,
            'company': company,
            'location': location,
            'relevance': 0.0,
            'missing_skills': [],
            'justification': "Relevance not assessed.",
            'experience':'N/A' # Default
        }
    except Exception as e:
        logging.error(f"Error parsing Indeed job card: {e}")
        return {  # Return defaults on error
            'url': None,
            'title': "N/A",
            'company': "N/A",
            'location': "N/A",
            'relevance': 0.0,
            'missing_skills': [],
            'justification': f"Error parsing job card: {type(e).__name__}",
            'experience':'N/A'
        }

@retry(
    wait=wait_exponential(multiplier=config.INDEED_BASE_DELAY, max=config.INDEED_MAX_DELAY),
    stop=stop_after_attempt(config.INDEED_RETRIES),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    before_sleep=lambda retry_state: logging.warning(
        f"Indeed request failed (attempt {retry_state.attempt_number}). Retrying in {retry_state.next_action.sleep} seconds..."
    )
)
def scrape_job_site(job_title: str, job_location: str, resume_text: Optional[str],
                    base_url: str, params_func: callable, parse_func: callable, site_name:str,
                    experience_level: Optional[str] = None) -> List[Dict]:
    """
    Generic function to scrape job listings from a given site.

    Args:
        job_title: The job title to search for.
        job_location: The location to search for jobs.
        resume_text: Optional resume text for relevance assessment.
        base_url: The base URL of the job site.
        params_func: A function that generates the URL parameters for the site.
        parse_func: A function that parses a job card from the site's HTML.
        site_name: The name of the job site (e.g., "LinkedIn", "Indeed").
        experience_level: Optional experience level string

    Returns:
        A list of dictionaries, where each dictionary represents a job listing.
    """

    search_results = []
    start = 0  #  pagination
    MAX_PAGES = 10  # Limit pages to prevent infinite loops.  Adjust as needed.

    while True:
        params = params_func(job_title, job_location, start, experience_level) # Pass experience
        try:
            headers = {'User-Agent': get_random_user_agent()}  # Rotate User-Agent
            response = requests.get(base_url, params=params, headers=headers, timeout=config.REQUEST_TIMEOUT)
            response.raise_for_status() # Raises HTTPError for bad (4xx, 5xx) responses

            if "captcha" in response.text.lower():
                logging.warning(f"{site_name} CAPTCHA detected. Stopping.")
                break  # Exit pagination

            soup = BeautifulSoup(response.text, 'html.parser')

            # Use a general way to find job cards (more robust to site changes)
            job_cards = soup.find_all('div', class_=lambda x: x and x.startswith('job_'))
            if not job_cards:
                job_cards = soup.find_all('div', class_='base-card') # For linkedin try another method


            if not job_cards:
                if start == 0:
                    logging.warning(f"No {site_name} jobs found for: {params}")
                else:
                    logging.info(f"No more {site_name} jobs found (page {start//50 + 1}).")
                break  # No more jobs, stop pagination

            for job_card in job_cards:
                try:
                    job_data = parse_func(job_card) # Parse Individual job card.

                    if not job_data['url']:  # Skip if no URL
                        continue

                    if resume_text:
                        try:
                            # Fetch the full job description
                            job_response = requests.get(job_data['url'], headers={'User-Agent': get_random_user_agent()},
                                                        timeout=config.REQUEST_TIMEOUT)
                            job_response.raise_for_status()
                            job_soup = BeautifulSoup(job_response.text, 'html.parser')
                            description_element = job_soup.find('div', id='jobDescriptionText') # Indeed
                            if not description_element: # for linkedin
                                description_element = job_soup.find('div', class_='description__text')
                            job_description = description_element.get_text(separator='\n', strip=True) if description_element else ""

                            # --- Experience Level Extraction (from description) ---
                            experience_match = re.search(r'(\d+\+?)\s*(?:-|to)?\s*(\d*)\s*years?', job_description, re.IGNORECASE)
                            if experience_match:
                                if experience_match.group(2): # If range
                                   job_data['experience'] = f"{experience_match.group(1)}-{experience_match.group(2)} years"
                                else: # Just single number
                                    job_data['experience'] = f"{experience_match.group(1)} years"
                            else: # Check for keywords
                                exp_keywords = {
                                    'fresher': ['fresher', 'graduate', 'entry level', '0 years', 'no experience'],
                                    'entry-level': ['0-2 years', '1-3 years', 'entry level', 'junior'],
                                    'mid-level' : ['3-5 years','2-5 years','mid level','intermediate'],
                                    'senior' : ['5+ years','5-10 years', 'senior','expert', 'lead'],
                                    'executive': ['10+ years', 'executive', 'director', 'vp', 'c-level']
                                }
                                for level, keywords in exp_keywords.items():
                                    for keyword in keywords:
                                        if keyword.lower() in job_description.lower():
                                          job_data['experience'] = level
                                          break # Stop checking once a level is found
                                    if job_data['experience'] != 'N/A':
                                        break # Stop checking other levels




                            job_description = job_description[:2000]  # Truncate!
                            resume_text_trunc = resume_text[:2000]  # Truncate!

                            relevance_prompt = (
                                f"Assess the relevance of the following job to the resume. "
                                f"Provide a JSON object with ONLY the following keys:\n"
                                f"'relevance': float (between 0.0 and 1.0, where 1.0 is perfectly relevant),\n"
                                f"'missing_skills': list of strings (skills in the job description but not in the resume, or an empty list if none),\n"
                                 f"'justification': string (REQUIRED. Explain the relevance score, including factors like experience level mismatch, skill gaps, or industry differences.).\n\n"
                                f"Job Description:\n{job_description}\n\nResume:\n{resume_text_trunc}"
                            )
                            relevance_assessment = generate_gemini_response(relevance_prompt, response_format="json", model_name=config.JOB_RELEVANCE_MODEL)
                            if isinstance(relevance_assessment, dict):
                                job_data['relevance'] = relevance_assessment.get('relevance', 0.0)  # Provide default
                                job_data['missing_skills'] = relevance_assessment.get('missing_skills', [])
                                job_data['justification'] = relevance_assessment.get('justification', "Relevance assessed.")
                                 # Basic validation (optional, but good practice):
                                if not isinstance(job_data['relevance'], (int, float)):
                                    logging.warning(f"Invalid relevance value: {job_data['relevance']}")
                                    job_data['relevance'] = 0.0
                                if not isinstance(job_data['missing_skills'], list):
                                    logging.warning(f"Invalid missing_skills: {job_data['missing_skills']}")
                                    job_data['missing_skills'] = []
                                if not isinstance(job_data['justification'], str):
                                    logging.warning(f"Invalid justification: {job_data['justification']}")
                                    job_data['justification'] = "Error: Could not assess relevance properly."
                            elif isinstance(relevance_assessment, dict) and "error" in relevance_assessment:
                                # Handle the "Invalid JSON" case specifically
                                logging.warning(f"Invalid JSON from Gemini: {relevance_assessment['raw_text']}")
                                job_data['relevance'] = 0.0  # Set default values
                                job_data['missing_skills'] = []
                                job_data['justification'] = "Error: Could not assess relevance due to invalid JSON response."

                            else: # Unexpected return from Gemini
                                logging.warning(f"Unexpected response from relevance assessment: {relevance_assessment}")
                                job_data['relevance'] = 0.0
                                job_data['missing_skills'] = []
                                job_data['justification'] = "Error: Could not assess relevance (unexpected response)."

                        except requests.exceptions.RequestException as e:
                            logging.warning(f"Failed to fetch job description from {job_data['url']}: {e}")
                            job_data['justification'] = f"Error: Could not fetch job description ({type(e).__name__})."
                        except Exception as e:
                            logging.exception(f"Error during relevance assessment for {job_data['url']}: {e}")
                            job_data['relevance'] = 0.0
                            job_data['missing_skills'] = []
                            job_data['justification'] = "Error: Could not assess relevance (unexpected error)."

                    search_results.append(job_data) # Append even with errors

                except Exception as e:
                    logging.warning(f"Error processing a {site_name} job card: {e}")
                    continue  #  skip to the next job card

            start += 50  #  pagination
            if start//50 +1 > MAX_PAGES:
                logging.info(f"Reached max pages ({MAX_PAGES}) for {site_name}.")
                break

        except requests.exceptions.HTTPError as e:
            logging.error(f"{site_name} HTTP Error: {e}")
            break  #  unrecoverable errors
        except requests.exceptions.RequestException as e:
            logging.error(f"{site_name} Request Exception: {e}")
            break  #  network errors

    return search_results


@app.post("/api/scrape_jobs")
async def scrape_jobs_endpoint(job_title: Optional[str] = Form(""), job_location: str = Form(...), resume: UploadFile = File(None),
                               job_experience: Optional[str] = Form(None)): # New parameter
    try:
        # Removed the 'not job_title' check to allow it to be optional

        resume_text = None
        if resume:
            resume_content = await resume.read()
            resume_text = extract_text_from_resume(resume_content)
            if not resume_text:
                raise HTTPException(status_code=400, detail="Could not extract text from resume.")

        all_job_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            # Submit both scraping tasks *concurrently*
            linkedin_future = executor.submit(scrape_job_site, job_title, job_location, resume_text,
                                               "https://www.linkedin.com/jobs/search", linkedin_params, parse_linkedin_job_card, "LinkedIn", job_experience) # Pass experience
            indeed_future = executor.submit(scrape_job_site, job_title, job_location, resume_text,
                                             "https://www.indeed.com/jobs", indeed_params, parse_indeed_job_card, "Indeed", job_experience) # Pass experience

            # Get results, handling exceptions gracefully.  Don't stop if one fails.
            try:
                all_job_results.extend(linkedin_future.result())
            except Exception as e:
                logging.error(f"Error scraping LinkedIn: {e}")  # Log, but don't stop
            try:
                all_job_results.extend(indeed_future.result())
            except Exception as e:
                logging.error(f"Error scraping Indeed: {e}") # Log, but don't stop

        # ---  Filtering and Sorting ---
        # Filter by experience first
        if job_experience:
            filtered_jobs = [job for job in all_job_results if job.get('experience', '').lower() == job_experience.lower()]
        else:
            filtered_jobs = all_job_results


        # Then, sort by experience level, then by relevance WITHIN each experience level
        experience_order = ['fresher', 'entry-level', 'mid-level', 'senior', 'executive', 'N/A']
        def sort_key(job):
             # Get experience level (default to 'N/A' if missing, put last)
            exp = job.get('experience', 'N/A').lower()
            if exp not in experience_order:
                exp = 'N/A' # Normalize to 'N/A'

            return (experience_order.index(exp), -job.get('relevance', 0.0))

        filtered_jobs.sort(key=sort_key)



        if filtered_jobs:
            return JSONResponse({'jobs': filtered_jobs, 'jobs_found': len(all_job_results)})
        else:
            # More specific message if no jobs *after* filtering
            return JSONResponse({'jobs': [], 'jobs_found': len(all_job_results)}, status_code=200) # Return 200 OK even if no jobs are found after filtering



    except HTTPException as e:
        raise e  # Re-raise HTTP exceptions for FastAPI to handle
    except Exception as e:
        logging.error(f"Error in jobs scraping endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Image Analysis Tool
@app.post("/api/analyze_image")
async def analyze_image_endpoint(request: Request):
    try:
        data = await request.json()
        image_data = data.get('image')
        if not image_data:
            return JSONResponse({"error": "No image provided"}, status_code=400)

        image_part = process_base64_image(image_data)
        if not image_part:
             return JSONResponse({"error": "Failed to process image"}, status_code=400)

        model = genai.GenerativeModel('gemini-2.0-flash')
       # image = Image.open(io.BytesIO(image_part['data'])) # No longer needed
        response = model.generate_content(["Describe this image in detail", image_part]) # Simple description prompt ,pass image part directly
        response.resolve()  # Ensure generation is complete

        return JSONResponse({"description": response.text})

    except Exception as e:
        logging.exception("Error in image analysis")
        return JSONResponse({"error": "Image analysis failed."}, status_code=500)


# Sentiment Analysis Tool
@app.post("/api/analyze_sentiment")
async def analyze_sentiment_endpoint(request: Request):
    try:
        data = await request.json()
        text = data.get('text')
        if not text:
             return JSONResponse({"error": "No text provided"}, status_code=400)

        prompt = f"Analyze the sentiment of the following text and classify it as 'Positive', 'Negative', or 'Neutral'. Provide a brief justification:\n\n{text}"
        sentiment_result = generate_gemini_response(prompt)

        return JSONResponse({"sentiment": sentiment_result})
    except Exception as e:
        logging.exception("Error in sentiment analysis")
        return JSONResponse({"error": "Sentiment analysis failed."}, status_code=500)

# Website Summarization Tool
@app.post("/api/summarize_website")
async def summarize_website_endpoint(request: Request):
    try:
        data = await request.json()
        url = data.get('url')
        if not url:
            return JSONResponse({"error": "No URL provided"}, status_code=400)

        content_snippets, _, _ = fetch_page_content(url, snippet_length=config.SNIPPET_LENGTH)
        if not content_snippets:
             return JSONResponse({"error": "Could not fetch website content"}, status_code=400)
        combined_content = "\n\n".join(content_snippets)
        prompt = f"Summarize the following webpage content concisely:\n\n{combined_content}"
        summary = generate_gemini_response(prompt)
        return JSONResponse({"summary": summary})
    except Exception as e:
        logging.exception("Error in website summarization")
        return JSONResponse({"error": "Website summarization failed."}, status_code=500)
