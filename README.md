# Mobile Carrier Tracking with AI CAPTCHA Resolver

An automated system for tracking mobile carrier information from an official government portal.

Because the target website is protected by CAPTCHA, this project integrates a custom-trained AI model to automatically decode CAPTCHA before submitting requests.

CAPTCHA model (Hugging Face):  
https://huggingface.co/m1nhb1e/captchaResolve

---

## Overview

The system performs the following steps:

1. Access the government carrier lookup page  
2. Download the CAPTCHA image  
3. Decode CAPTCHA using the trained AI model  
4. Submit the decoded result  
5. Extract and return carrier information  

---

## Architecture

```
User Input
     ↓
Fetch Government Portal
     ↓
Download CAPTCHA
     ↓
Model Inference
     ↓
Submit Form
     ↓
Parse Carrier Data
     ↓
Return Result
```

---

## Tech Stack

- Python  
- Requests / Selenium  
- PyTorch (or relevant deep learning framework)  
- Hugging Face Hub integration  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/m1nhb1ee/worker.git
cd worker
```

(Optional) Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main script:

```bash
python main.py (For fast with Self-trained Model)
python ver1.py (For advance with Microsoft Model)
python worker.py (For advance with Self-trained Model)
```

The system will automatically:
- Fetch CAPTCHA  
- Decode it using the AI model  
- Submit the request  
- Retrieve carrier information  

---

## Project Structure

```
.
├── app.py
├── ver1.py
├── worker.py
├── requirements.txt
└── README.md
```

---

## Disclaimer

This project is intended for educational and research purposes only.

Users are responsible for complying with:
- The target website’s terms of service  
- Applicable laws and regulations  

The author assumes no responsibility for misuse.
