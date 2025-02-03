import gradio as gr
import requests
import json

# Server URL
MODEL_SERVER_URL = "http://model-server:8000/generate"

def generate_text(prompt: str, max_length: int = 100, temperature: float = 0.8):
    try:
        # Prepare the request payload
        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature
        }
        
        # Send POST request to the model server
        response = requests.post(MODEL_SERVER_URL, json=payload)
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            return result["generated_text"]
        else:
            return f"Error: Server returned status code {response.status_code}\n{response.text}"
            
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the model server. Please make sure it's running."
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
        gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="SmolLM2 Text Generation Client",
    description="Enter a prompt and the model will generate text based on it. This client communicates with a separate model server container.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True) 