# Kv - AI-Powered Deep Research Tool 🚀 - Open Source & Web Scraping Based!

[![Project Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/your-github-username/your-repo-name)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python->=3.8-blue.svg)](https://www.python.org/downloads/)

**Founder: K. Vamsi Krishna**

**Unleash the Power of Deep Research - Without the API Costs!** Kv is an intelligent AI research companion that leverages **web scraping** to deliver comprehensive analysis, unlike costly API-dependent tools. Dive deep into any topic and extract valuable insights, all while staying completely open-source and cost-effective.

---

## ✨ Screenshots

Showcase your project with compelling visuals! Replace these placeholders with actual screenshots of Kv in action.

1.  **Chat Interface:**
    ![image](https://github.com/user-attachments/assets/b9366540-2a69-4c2f-8f56-2db66deacd89)

  

2.  **Deep Research in Action - Web Scraping Power:**
    ![image](https://github.com/user-attachments/assets/1af98352-e909-44e0-b714-42805dc262c1)
![image](https://github.com/user-attachments/assets/0ba492bf-6691-4ad2-b27c-1618d84e6724)
![image](https://github.com/user-attachments/assets/d7cac571-62d8-4d7b-8783-b33cb560942f)


3.  **Options Menu - Control & Customization:**

   
    ![image](https://github.com/user-attachments/assets/5a7775c4-78a6-4243-9973-d5fd0cffa46e)



---

## 🌟 Key Differentiator: Web Scraping for Cost-Effective Research

**Kv stands apart from other deep research tools by utilizing direct web scraping techniques instead of relying on expensive Google Search APIs or similar paid services.** This fundamental difference offers significant advantages:

*   **Zero API Costs:**  Eliminate recurring expenses associated with API usage, making deep research accessible to everyone.
*   **Unrestricted Data Access:** Go beyond API limitations and directly access a wider range of web content for more comprehensive research.
*   **Open Source & Transparent:**  Benefit from a fully open-source solution where the data acquisition process is transparent and customizable.

---

## 💡 Features

Kv is packed with features designed to empower your research workflow and provide unparalleled insights:

*   **Intelligent & Cost-Effective Research Engine:**
    *   **Web Scraping Powered:** Leverages direct web scraping across multiple search engines, avoiding costly APIs and expanding data access.
    *   **Multi-Search Engine Support:**  Simultaneously searches Google, DuckDuckGo, Bing, Yahoo, Brave, and LinkedIn for broader coverage.
    *   **Alternative Query Generation:**  Intelligently refines search queries to maximize results and overcome search engine limitations.
    *   **Deep Content Analysis:**  Scrapes and analyzes web page content, extracting relevant information and discarding irrelevant data.
    *   **Comprehensive Summarization:**  Synthesizes detailed summaries from vast amounts of scraped data, identifying key themes and diverse perspectives.

*   **Intuitive Chat Interface:**
    *   Engage in interactive conversations to guide your research and refine queries.
    *   Upload images for context-aware analysis and visual research.
    *   Maintains full conversation history for seamless research sessions.
    *   Customize system instructions to fine-tune Kv's behavior and output.

*   **Flexible Output & Data Extraction:**
    *   **Markdown Output (Default):**  Clean, well-formatted Markdown for easy reading and integration into notes and reports.
    *   **Structured Data Formats:**  JSON and CSV output options for programmatic analysis and data manipulation.
    *   **Data Extraction Capabilities:**  Optionally extract links and emails from scraped web pages for targeted data gathering.
    *   **Reference Citations:**  Includes clear references with shortened URLs to easily verify sources and maintain academic rigor.
    *   **Performance Metrics:**  Provides elapsed time for research tasks, giving insights into processing efficiency.

*   **User-Centric Customization:**
    *   **Search Engine Selection:**  Choose and prioritize specific search engines to tailor research scope.
    *   **Output Format Control:** Select Markdown, JSON, or CSV to suit your data needs.
    *   **Data Extraction Toggle:**  Enable or disable link and email extraction as required.
    *   **Gemini Model Selection:**  Experiment with different Gemini models for varied response styles.
    *   **Custom Instructions:**  Inject specific prompts and guidelines for highly tailored AI behavior.
    *   **Theme Switching:**  Effortlessly toggle between light and dark themes for optimal viewing in any environment.

---


## 🚀 Getting Started

Experience the power of cost-effective deep research by setting up Kv on your local machine:

### Prerequisites

*   **Python 3.8 or higher:**  Download from [python.org](https://www.python.org/downloads/).
*   **pip:** Python's package installer (usually included with Python).

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/kvcops/Deep-Research-using-Gemini-api.git
    cd Deep-Research-using-Gemini-api

    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate  # Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Key:**
    *   Get a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   Create a `.env` file in the project root.
    *   Add your API key:

        ```env
        GEMINI_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY
        ```

### Running Kv

1.  **Start the Flask app:**

    ```bash
    python app.py
    ```

2.  **Open Kv in your browser:** `http://127.0.0.1:5000/`

---

## 🧑‍💻 Usage

Kv's interface is designed for ease of use and efficient research:

1.  **Chat Interaction:**  Type messages, send with "Send" or Enter. Use "Clear Chat" in Options to reset. Upload images using the image icon.

2.  **Web Search:** Enter query, click "Web" for summarized results and references.

3.  **Deep Research:** Input research topic, click "Deep Dive" for comprehensive analysis (may take longer).

4.  **Customize Options (Dropdown Menu):**
    *   **Model:** Select Gemini model.
    *   **Custom Instructions:** Tailor Kv's behavior with specific prompts.
    *   **Search Engines:** Choose engines for web searches and deep research.
    *   **Output Format:** Select Markdown, JSON, or CSV for Deep Research.
    *   **Data Extraction:** Enable link/email extraction (JSON output recommended).

5.  **Theme Toggle:**  Switch between light/dark themes with the header icon.

---

## 🤝 Contributing - Help Shape the Future of Open-Source Research!

Kv is a community-driven project, and your contributions are highly valued! We believe in making powerful research tools accessible to everyone, and your help is crucial to achieving this vision.

**Ways to Contribute:**

*   **Code Contributions:**  Implement new features, improve existing functionality, fix bugs, enhance scraping robustness, optimize performance, and expand data extraction capabilities.
*   **Documentation Improvements:**  Enhance the README, create tutorials, improve code comments, and build comprehensive user guides.
*   **Testing & Bug Reporting:**  Thoroughly test Kv, identify bugs, and provide detailed bug reports to help improve stability and reliability.
*   **Feature Suggestions:**  Share your ideas for new features and enhancements to make Kv even more powerful and user-friendly.
*   **UI/UX Design:**  Contribute to improving the user interface and user experience to make Kv even more intuitive and visually appealing.
*   **Spread the Word:**  Share Kv with your network, write blog posts, create demos, and help grow the community!

**Ready to contribute?**

1.  **Fork the repository.**
2.  **Create a feature branch.**
3.  **Code your amazing contribution!**
4.  **Submit a pull request with a clear description of your changes.**

Let's build the future of open-source, cost-effective deep research together!

---

## 📜 License

Kv is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.  This means you are free to use, modify, and distribute Kv for both commercial and non-commercial purposes.

---

## 📞 Contact & Support

For questions, feedback, bug reports, or feature requests:

*   **GitHub Issues:**  [https://github.com/kvcops/Deep-Research-using-Gemini-api/issues]
*   **Email (Optional):** 21131A05C6@gvpce.ac.in
*   **Project Founder - K. Vamsi Krishna:** https://www.linkedin.com/in/karri-vamsi-krishna-966537251/

Your input is invaluable in shaping Kv's future. We are committed to building a robust and accessible research tool for the global community!

---

**Thank you for joining the Kv journey! Let's democratize deep research!** 📚
