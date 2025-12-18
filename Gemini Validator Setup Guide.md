<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Gemini Validator Setup Guide

Quick guide to swap from Azure OpenAI to Google Gemini for answer validation.

***

## Step 1: Install Gemini SDK

```bash
pip install google-generativeai
```


***

## Step 2: Get Gemini API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Click **"Create API Key"**
3. Copy your API key

***

## Step 3: Update `.env` File

Add this line to your `.env` file:

```bash
# Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here
```

**Keep your existing Azure variables for generation:**

```bash
AZURE_OPENAI_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://prapocopenai.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=pra-poc-gpt-4o
```


***

## Step 4: Update `config/settings.py`

Add these constants:

```python
# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # or "gemini-1.5-pro"
```


***

## Step 5: Update `tools/validation_tools.py`

### 5.1: Update `_initialize_llm_client` method

Find the `_initialize_llm_client` method and add the Gemini section:

```python
def _initialize_llm_client(self):
    """Initialize LLM client based on provider"""
    try:
        if self.llm_provider == "azure_openai":
            # ... existing Azure code (keep as is) ...
            
        elif self.llm_provider == "gemini":
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            
            genai.configure(api_key=api_key)
            
            self.llm_client = genai.GenerativeModel(
                model_name=self.llm_model or "gemini-1.5-flash"
            )
            
            logger.info(f"✅ Gemini client initialized (model: {self.llm_model or 'gemini-1.5-flash'})")
            
        else:
            logger.warning(f"Unknown provider: {self.llm_provider}, using rule-based")
            self.llm_client = None
            
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        self.llm_client = None
```


### 5.2: Update `_llm_judge_validate` method

Find the try block in `_llm_judge_validate` and update it:

```python
try:
    if self.llm_provider == "azure_openai":
        # Existing Azure code (keep as is)
        response = self.llm_client.chat.completions.create(
            model=self.llm_deployment,
            messages=[
                {"role": "system", "content": "You are a precise answer validator. Always respond in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        llm_result = json.loads(response.choices[0].message.content)
    
    elif self.llm_provider == "gemini":
        # New Gemini code
        response = self.llm_client.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json"
            }
        )
        llm_result = json.loads(response.text)
    
    # ... rest of the method stays the same ...
```


***

## Step 6: Use Gemini in Your Code

Update your test script or orchestrator:

```python
from tools.validation_tools import ValidationTools

# Change from Azure to Gemini
validator = ValidationTools(
    enable_llm_judge=True,
    llm_provider="gemini",           # ← Change this
    llm_model="gemini-1.5-flash",    # ← Add this
    fallback_to_rules=True
)
```


***

## Step 7: Test

Run your validation test:

```bash
python test_validation_final.py
```

You should see:

```
✅ Gemini client initialized (model: gemini-1.5-flash)
```


***

## Cost Comparison

| Provider | Model | Cost per 1M tokens |
| :-- | :-- | :-- |
| Azure OpenAI | GPT-4o | \$5-10 |
| Azure OpenAI | GPT-4o-mini | \$0.15-0.60 |
| **Gemini** | **1.5 Flash** | **\$0.075** ✅ |
| Gemini | 1.5 Pro | \$3.50 |

**💡 Gemini Flash is ~20x cheaper than Azure GPT-4o!**

***

## Swap Back to Azure

```python
validator = ValidationTools(
    enable_llm_judge=True,
    llm_provider="azure_openai",  # ← Change back
    fallback_to_rules=True
)
```


***

## Done! ✅

Your validator now uses Gemini instead of Azure OpenAI.

