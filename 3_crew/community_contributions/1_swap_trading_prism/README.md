# 💹 PRISM - Position Risk Intelligence & Swap Monitor

AI-powered USD SOFR swap trading monitoring system using CrewAI multi-agent framework.

**🌐 Live Demo:** [prism.lisekarimi.com](https://prism.lisekarimi.com)

## 📊 Overview
Monitors USD SOFR swap positions in real-time and generates trading signals. **5 AI Agents:** Market Data, Position Manager, Risk Manager, Risk Calculator, and Trading Decision.

**Thresholds:** Close when profit ≥ $50K or loss ≤ -$25K, otherwise HOLD.

## 🛠️ Tech Stack
CrewAI • PostgreSQL (Neon) • Python 3.11

## 🚀 Quick Start

1. **Prerequisites:** PostgreSQL database, OpenAI API key, Serper API key
2. **Setup:**
   ```bash
   cd .\3_crew\community_contributions\1_swap_trading_prism\
   uv sync
   # Add the following keys in .env: DATABASE_URL=, OPENAI_API_KEY=, SERPER_API_KEY=
   crewai run
   ```

For full codebase with Gradio dashboard: [github.com/lisekarimi/prism](https://github.com/lisekarimi/prism)
