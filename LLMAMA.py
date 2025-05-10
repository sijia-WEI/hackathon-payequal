"""
LLaMA 3.2 Chatbot

This module implements a simple command-line chatbot that uses the LLaMA 3.2 API
to generate responses to user inputs in a conversational format.
"""
import os
import requests
import json
import re
import csv
import joblib
from dotenv import load_dotenv
from datetime import datetime
from salary_predictor import predict_salary


# Load environment variables from .env file for secure API credentials storage
load_dotenv()

# Retrieve API configuration from environment variables
API_URL = os.getenv("LLAMA_API_URL")
API_KEY = os.getenv("LLAMA_API_KEY")

# Set up the HTTP headers for API authentication and content type
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def extract_information(text):
    info = {}

    # try to split the string and get key info
    experience_level_match = re.search(r"(SE|Junior|Senior|Lead|Junior Developer|Senior Developer)", text, re.IGNORECASE)
    info['experience_level'] = experience_level_match.group(0) if experience_level_match else None

    employment_type_match = re.search(r"(FT|PT|Contract|Intern)", text, re.IGNORECASE)
    info['employment_type'] = employment_type_match.group(0) if employment_type_match else None

    job_title_match = re.search(r"(Data Scientist|Software Engineer|Data Analyst|Product Manager)", text, re.IGNORECASE)
    info['job_title'] = job_title_match.group(0) if job_title_match else None

    remote_ratio_match = re.search(r"(\d{1,3})\s*%?\s*(remote|work from home)", text, re.IGNORECASE)
    info['remote_ratio'] = int(remote_ratio_match.group(1)) if remote_ratio_match else None 

    company_location_match = re.search(r"(US|Canada|UK|Australia)", text, re.IGNORECASE)
    info['company_location'] = company_location_match.group(0) if company_location_match else None

    company_size_match = re.search(r"\b(L|M|S)\b", text, re.IGNORECASE)
    info['company_size'] = company_size_match.group(0) if company_size_match else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"user_info_{timestamp}.json"
    csv_filename = f"salaries.csv"

    # save as JSON
    with open(json_filename, 'w') as jf:
        json.dump(info, jf, indent=4)

    # save as CSV
    with open(csv_filename, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=info.keys())
        writer.writeheader()
        writer.writerow(info)

    # print(f"✅ Saved to {json_filename} and {csv_filename}")

    return info


def chat():
    """
    Implements an interactive chat loop with the LLaMA 3.3 using Azure AI Foundry API.
    
    This function manages the conversation state, sends user messages to the API,
    and displays the AI's responses until the user exits.
    """
    # Initialize conversation with a system message to define AI behavior
    messages = [{"role": "system", "content": "Just give some career advice"}]
    print("🤖 Welcome to the LLaMA 3.3 Chatbot! \n")
    print("Please include details like:\n")
    print("  - Experience level (e.g., Junior, Senior, Lead)\n")
    print("  - Job title (e.g., Data Scientist, Software Engineer)\n")
    print("  - Employee residence (e.g., US, Canada, UK, Australia)\n")
    print("  - Remote work ratio (e.g., 50% remote, 100% remote)\n")
    print("  - Company location (e.g., US, Canada, UK, Australia)\n")
    print("  - Company size (L, M, S)\n")
    print("\nFeel free to type 'exit' to quit the chat.\n")
    
    while True:
        # Get user input and handle exit commands
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        
        # Add user message to conversation history
        messages.append({"role": "user", "content": user_input})
        
        # Prepare the API request payload
        payload = {
            "messages": messages,  # Full conversation history
            "temperature": 0.9,    # Controls randomness (0.0=deterministic, 1.0=creative)
            "max_tokens": 500,     # Limits response length
            "model": "Llama-3.3-70B-Instruct"  # Specifies which model to use
        }
        
        # Send request to LLaMA API
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # Process the API response
        if response.status_code == 200:
            # Extract and display the AI's response
            assistant_message = response.json()["choices"][0]["message"]["content"]
            print(f"AI: {assistant_message}\n")
            
            # Add AI response to conversation history for context in next turn
            messages.append({"role": "assistant", "content": assistant_message})
            info = extract_information(user_input)

            if all(info.values()):
                try:
                    predicted_salary = predict_salary(info)
                    print(f"\nPredicted salary: ${predicted_salary:,.2f}")
                except Exception as e:
                    print(f"Prediction failed: {e}")

        else:
            # Handle API errors
            print(f"Error: {response.status_code} - {response.text}")
            break

if __name__ == "__main__":
    chat()