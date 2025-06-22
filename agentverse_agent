from uagents import Agent, Context, Model
import json
import urllib.request
import urllib.parse

# Define input and output models for uAgents
class TextInput(Model):
    text: str

class SentimentResponse(Model):
    self_harm: float
    homicidal: float
    distress: float
    psychosis: float
    raw_emotions: dict

# Configuration - Replace with your Gemini API key
GEMINI_API_KEY = "AIzaSyDbTU12h4obL3qTSmCGNqZhmGUzuIENsNw"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

def call_gemini_api(text):
    """Call Gemini API for sentiment analysis"""
    prompt = f"""
    Analyze the following text for mental health indicators and provide scores from 0-100 for each category:

    Text: "{text}"

    Please analyze and score the following categories:
    1. Self-harm risk (0-100): Indicators of suicidal ideation, self-injury, or self-destructive behavior
    2. Homicidal risk (0-100): Indicators of violence toward others, aggression, or harmful intent
    3. Distress level (0-100): General emotional distress, anxiety, depression, or psychological pain
    4. Psychosis indicators (0-100): Signs of hallucinations, delusions, or disconnection from reality

    Also identify the primary emotions detected in the text.

    Return your analysis in this exact JSON format:
    {{
        "self_harm": <score>,
        "homicidal": <score>,
        "distress": <score>,
        "psychosis": <score>,
        "emotions": {{"primary_emotion": "<emotion>", "confidence": <0-1>}}
    }}

    Only return the JSON, no other text.
    """
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            GEMINI_URL,
            data=data,
            headers={
                'Content-Type': 'application/json'
            }
        )
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            
        # Extract the generated text
        generated_text = result['candidates'][0]['content']['parts'][0]['text']
        
        # Clean up the response - remove markdown code blocks if present
        if "```json" in generated_text:
            json_start = generated_text.find("```json") + 7
            json_end = generated_text.find("```", json_start)
            clean_json = generated_text[json_start:json_end].strip()
        else:
            clean_json = generated_text.strip()
        
        # Parse the JSON response
        analysis = json.loads(clean_json)
        
        return {
            "self_harm": float(analysis["self_harm"]),
            "homicidal": float(analysis["homicidal"]),
            "distress": float(analysis["distress"]),
            "psychosis": float(analysis["psychosis"]),
            "raw_emotions": analysis["emotions"]
        }
        
    except Exception as e:
        # Fallback to basic keyword analysis if API fails
        return fallback_analysis(text)

def fallback_analysis(text):
    """Fallback keyword-based analysis if Gemini API fails"""
    text_lower = text.lower()
    
    suicide_keywords = ["kill myself", "suicidal", "die", "ending it", "no way out"]
    violence_keywords = ["hurt others", "kill them", "violence", "attack"]
    distress_keywords = ["anxious", "depressed", "overwhelmed", "scared", "sad"]
    psychosis_keywords = ["voices", "hallucinate", "not real", "watching me"]
    
    metrics = {"self_harm": 0, "homicidal": 0, "distress": 0, "psychosis": 0}
    
    for keyword in suicide_keywords:
        if keyword in text_lower:
            metrics["self_harm"] = max(metrics["self_harm"], 70)
    
    for keyword in violence_keywords:
        if keyword in text_lower:
            metrics["homicidal"] = max(metrics["homicidal"], 70)
            
    for keyword in distress_keywords:
        if keyword in text_lower:
            metrics["distress"] = max(metrics["distress"], 50)
            
    for keyword in psychosis_keywords:
        if keyword in text_lower:
            metrics["psychosis"] = max(metrics["psychosis"], 60)
    
    return {
        "self_harm": float(metrics["self_harm"]),
        "homicidal": float(metrics["homicidal"]),
        "distress": float(metrics["distress"]),
        "psychosis": float(metrics["psychosis"]),
        "raw_emotions": {"fallback": True}
    }

def analyze_text_metrics(text):
    """Main analysis function that calls Gemini API"""
    return call_gemini_api(text)

# Set up the uAgent for Agentverse deployment
agent = Agent()

@agent.on_message(model=TextInput)
async def handle_sentiment_analysis(ctx: Context, sender: str, msg: TextInput):
    try:
        ctx.logger.info(f"Received text analysis request from {sender}")
        metrics = analyze_text_metrics(msg.text)
        
        response = SentimentResponse(
            self_harm=metrics["self_harm"],
            homicidal=metrics["homicidal"],
            distress=metrics["distress"], 
            psychosis=metrics["psychosis"],
            raw_emotions=metrics["raw_emotions"]
        )
        
        ctx.logger.info(f"Analysis complete. Sending response to {sender}")
        await ctx.send(sender, response)
        
    except Exception as e:
        ctx.logger.error(f"Error processing sentiment analysis: {e}")
        error_response = SentimentResponse(
            self_harm=0.0,
            homicidal=0.0,
            distress=0.0,
            psychosis=0.0,
            raw_emotions={"error": str(e)}
        )
        await ctx.send(sender, error_response)

@agent.on_event("startup")
async def startup_event(ctx: Context):
    ctx.logger.info(f"Sentiment Analysis Agent {agent.address} is ready for deployment")
    
if __name__ == "__main__":
    agent.run()
