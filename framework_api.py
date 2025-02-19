import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from deep_translator import GoogleTranslator

# ✅ Load environment variables
load_dotenv()

# ✅ Load OpenAI API Key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key! Set OPENAI_API_KEY environment variable.")

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Load OpenAI GPT-4-turbo model
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY)

# ✅ Define the request model for input
class FrameworkRequest(BaseModel):
    topic: str

# ✅ Define the framework prompt template
framework_prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="""
    Generate a structured framework prompt for {topic}.
    The framework should include:
    - Problem definition
    - Key categories/stages
    - Step-by-step actions
    - Customization points
    - Output format suggestions
    """
)

framework_chain = LLMChain(llm=llm, prompt=framework_prompt_template)

# ✅ Create API endpoint to generate framework prompts
@app.post("/generate-framework")
async def generate_framework(request: FrameworkRequest):
    generated_prompt = framework_chain.run(topic=request.topic)
    return {
        "topic": request.topic,
        "framework_prompt": generated_prompt
    }

# ✅ Initialize Google Translator
translator = GoogleTranslator(source="auto", target="en")
translation = translator.translate("Bonjour")
print(translation)  # Output: Hello

# ✅ Run Uvicorn Server (Only when run directly)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)