# careerwise_gemini_notify

A plug-and-play career guidance chatbot powered by Google Gemini Free models, with ntfy open-source notifications—ideal for integrating into your personal portfolio site!

## Overview

`careerwise_gemini_notify` helps users get career advice, resources, and personalized responses via a conversational chatbot. Notifications (like reminders or new advice) are pushed using [ntfy](https://ntfy.sh/). The chatbot is deployed as an API (e.g., on Google Cloud), and can be easily integrated with any web-based personal portfolio.

---

## Features

- **Runs on Gemini Free Models**: Cost-effective, easy-to-setup AI for answering career questions.
- **ntfy Notifications**: Instant notification support for events (advice, new messages, reminders, etc.)
- **API-Based**: Can be deployed as a microservice and connected to web frontends/applications.
- **Fast Portfolio Integration**: Simple steps for adding career chat features to your portfolio site.

---

## How it Works

1. **User interacts** with chatbot via your portfolio site.
2. **Gemini model** generates responses for career queries.
3. **ntfy integration** delivers notifications to users (subscribe to new advice, reminders, etc.).

---

## Folder Structure

```
careerwise_gemini_notify/
  ├─ README.md             # This file
  ├─ gemini_chatbot.py     # Gemini model integration chatbot script
  ├─ ntfy_integration.py   # ntfy notification integration module
  ├─ api_server.py         # Flask/FastAPI server for API deployment
  ├─ deployment_guide.md   # How to deploy the API on Google Cloud
  ├─ portfolio_integration.md # Guide: Adding the chatbot to your portfolio
  ├─ requirements.txt      # Python dependencies
```

---

## Quick Start (Local Demo)

1. **Clone this repo/folder into your project:**

```bash
git clone https://github.com/ed-donner/agents.git
cd agents/1_foundations/community_contributions/careerwise_gemini_notify
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Configure Gemini & ntfy:**
   - Add your Gemini API access keys/config.
   - Set your ntfy server URL or use defaults in `ntfy_integration.py`.

4. **Run locally:**

```bash
python api_server.py
```

---

## Steps: Add the Chatbot to Your Personal Portfolio

1. **Deploy the API**
   - See [`deployment_guide.md`](deployment_guide.md) for deployment steps on Google Cloud (or other platforms).

2. **Integrate with Your Site**
   - Frontend (React, Vue, static HTML, etc.):  
     - Add a chat UI component (see sample in `portfolio_integration.md`).
     - Connect your frontend to the chatbot API endpoint (e.g., using Axios/Fetch).
     - Subscribe users to ntfy notifications either on your backend or let them set up ntfy subscriptions using their own app.

3. **Showcase!**
   - Highlight the chatbot and notification feature on your portfolio.
   - (Optional) Customize bot persona and advice responses to fit your brand.

---

## Contribution

We welcome suggestions and improvements!  
Open an Issue/PR in [ed-donner/agents](https://github.com/ed-donner/agents) under `1_foundations/community_contributions/careerwise_gemini_notify`.

---

## License

MIT (see repository root for full details).

---

## Acknowledgments

- [Google Gemini](https://ai.google.dev/)
- [ntfy (Open Source Notifications)](https://ntfy.sh/)
- [ed-donner/agents](https://github.com/ed-donner/agents)

---

**Happy career chatting! 🚀**
