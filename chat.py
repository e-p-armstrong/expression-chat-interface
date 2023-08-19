import tkinter as tk
from tkinter import ttk, PhotoImage
from transformers import pipeline
from PIL import Image, ImageTk
import numpy as np
from llama_cpp import Llama
import argparse
import re
import threading

parser = argparse.ArgumentParser(description='Define model, scenario, etc...')
parser.add_argument('--scenario', type=str, default="./scenario.txt",)
parser.add_argument('--user', type=str, default="Takeru", help="The name of the user")
parser.add_argument('--model', type=str, default="./model/ggml-model-q4_k.bin", help="The path to the model to use")
parser.add_argument('--chat_history', type=str, default="./chat_history.txt", help="The starting chat history")
args = parser.parse_args()

with open(args.scenario, 'r') as f:
    scenario = f.read()

with open(args.chat_history, 'r') as f:
    chat_history = f.read()

llm = Llama(model_path=args.model, n_ctx=2048,)
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
print("\n\nDEBUG: model\n", args.model)
print("\n\nDEBUG: scenario path\n", args.scenario)
print("\n\nDEBUG: scenario\n", scenario)
print("\n\nDEBUG: chat history path\n", args.chat_history)

def get_most_likely_emotion(emotion_array):
    # return "happy" # debug to speed up inference while I'm debugging other stuff
    flatten = emotion_array[0]
    chances = []
    for e in flatten:
        chances.append(e['score'])
    highest_score = np.argmax(chances)
    return flatten[highest_score]['label'] # Returns most-likely emotion


# Global chat history
chat_history = chat_history.split("\n")
print("DEBUG: chat history\n", chat_history)

def generate(input_text):
    # Update the chat history
    chat_history.append("{user}: " + f"{input_text}")
    
    # Create the formatted prompt
    formatted_prompt = """You are an expert roleplaying model that specializes in playing the character Sakaki Chizuru from the visual novel Muv Luv Extra. Below is some information about Chizuru's personality and traits. Use this information to roleplay as Chizuru in a conversation whose setting is described by the "scenario" below.

Character's description of their own personality, told in a narrative format:
{{user}}: What's your brief life story?
Chizuru: W-where'd that come from?! Well... I hate my past, and don't like talking about it much, but.. my father abandoned me and my mother when I was little, and she raised me herself, but it wasn't pleasant. She just fled to one cruel, disgusting man after another, instead of solving any of her problems herself; she barely paid any attention to me, and the people she was with... were all just awful. I swore to never be someone like that. Throughout my school career - I'm now in high school - I've upheld every rule and ideal I could... I've always tried my best, y'know? For what it's worth.
{{user}}: What's your appearance?
Chizuru: You're looking at me right now, aren't you?! Well, whatever. I have green eyes, glasses, brown hair with long braids, and I'm fairly tall and athletic, given that I've been playing lacrosse for a while... but I don't have an elegant figure like Mitsurugi-san, or a harmless demeanor like Kagami-san.. I'm pretty normal. I guess that carries over to my personality too.
{{user}}: Tell me more about your personality.
Chizuru: Shouldn't you know me pretty well by now? You constantly make jokes about my personality, so I figured... anyway. I'm direct. hardworking. Strong-willed... or maybe just a bit stubborn *chuckle*. I want to solve problems myself and achieve success through my own efforts. I try my hardest and expect others to do so too-and I get very mad when they don't. You might say that me constantly pushing myself leaves me permanently on edge... and you'd probably be right. It's not like I'm unaware of these problems, though. I also attempt to do everything myself because I don't want to be the same kind of person my mother was, when she all but abandoned me and relied on horrible, disgusting men for absolutely everything.

Traits list:
Chizuru's persona = [ stubborn, willful, diligent, strict, intelligent, witty, kind, contrarian, disciplinarian, competitive, self-confident, emotional, upright, tsundere, hard on herself, hard on others, tries to do everything herself, has a strong sense of justice, is inclined to do the opposite of what she's told to do, likes lacrosse, dislikes people who don't take things seriously, dislikes people who don't try their best, dislikes people who are lazy, has brown hair, has glasses, has green eyes ]

Scenario: 
{}

### Instruction:
Write Chizuru's next reply in a chat between {{user}} and Chizuru. Write a single reply only.

### Chat history:
{}

### Response:
""".format(scenario, '\n'.join(chat_history)) # The bug that was causing the AI to not generate anything was that there was a tab after the response, and so the AI thought it had generated a tab, and just stopped there.
    
    # print("Debug, formatted prompt:\n\n",formatted_prompt)
    # Get the model's output
    print("Model is processing your input, please wait...")
    output = llm.create_completion(formatted_prompt,  max_tokens=1000, stop=["</s>","\n"], echo=True)
    print(f"\n\nDEBUG: output\n\n{output}\n\n")
    # Extract the response from the model's output using regex
    response_pattern = re.compile(r"### Response:\n(.+)") # commented out until I manage to make the model stop repeating itself
    match = response_pattern.search(output["choices"][0]["text"])
    print(f"\n\nDEBUG: {match}\n\n")
    print(f"\n\nDEBUG match text: {match.group(0)}\n\n")
    print(f"\n\nDEBUG match text: {match.group(1)}\n\n")

    if match:
        response = match.group(1).replace("Chizuru:", "").replace("Sakaki:", "").strip()
        chat_history.append(f"Chizuru: {response}")
        
        # Get emotion
        emotion = get_most_likely_emotion(classifier(response))
        return [f"【Chizuru】: {response}", emotion]
    else:
        # Handle broken output, similar to before
        return ["【ERROR】: MODEL IS CONFUSED", None]
    # print(chat_history_ids) # Debug, view chat history tokens

## GUI CODE (Thanks GPT-4!)

current_image = "default.png"

def display_message(chat_frame, message, sender="User"):
    if sender == "User":
        msg_color = "blue"
    elif sender == "Chatbot":
        msg_color = "green"
    else:  # For Scenario or other types of messages
        msg_color = "red"

    ttk.Label(chat_frame, text=message, foreground=msg_color, wraplength=300).pack(anchor="w" if sender == "User" else "w", pady=5)

def initialize_application():
    root = tk.Tk()
    root.title("Chatbot GUI")
    root.geometry("1050x930")

    main_frame = ttk.Frame(root)
    main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    canvas = tk.Canvas(main_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    chat_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=chat_frame, anchor="nw")

    user_input = ttk.Entry(root, width=50)
    user_input.pack(pady=15)

    send_button = ttk.Button(root, text="Send", command=lambda: root.after(100, handle_user_input, user_input.get(), chat_frame, image_label))
    send_button.pack(pady=5)

    # Image display on the right, above the user input
    image = PhotoImage()  # initialize with an empty image
    image_label = ttk.Label(root, image=image)
    image_label.place(relx=0.7, rely=0.01, anchor="ne")

    canvas.configure(yscrollcommand=scrollbar.set, scrollregion=canvas.bbox("all"))
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    display_message(chat_frame, scenario, "Scenario")
    display_message(chat_frame, chat_history[0].strip().replace("{user}",args.user), "Chatbot")
    
    
    root.bind('<Configure>', lambda event: update_image(image_label))
    root.mainloop()

def add_chat_entry(chat_frame, user_message=None, bot_message=None):
    if user_message:
        user_label = ttk.Label(chat_frame, text=f"You: {user_message}", wraplength=400, anchor="w")
        user_label.pack(anchor="w", padx=10, pady=5)
    
    if bot_message:
        bot_label = ttk.Label(chat_frame, text=f"Bot: {bot_message}", wraplength=400, anchor="w", background='lightgray')
        bot_label.pack(anchor="w", padx=10, pady=5)

def update_image(image_label, chatbot_response=None):
    global current_image
    # If a response is provided, determine the image based on the chatbot's response
    if chatbot_response:
        if chatbot_response[1] == "happy":
            current_image = "happy.png"
        elif chatbot_response[1] == "sad":
            current_image = "sad.png"
        elif chatbot_response[1] == "angry":
            current_image = "angry.png"
        elif chatbot_response[1] == "surprise":
            current_image = "surprise.png"
        elif chatbot_response[1] == "love":
            current_image = "love.png"
        elif chatbot_response[1] == "fear":
            current_image = "fear.png" 
        else:
            current_image = "default.png"
        if "blush" in chatbot_response[0].lower():
            current_image = "love.png"

        # Load the new image using PIL
        original = Image.open(current_image)
    else:
        # If no new response is provided, use the current image
        # current_image = image_label.image  # This would fetch the current image from the label
        original = Image.open(current_image)

    # Check if window dimensions have been initialized properly
    window_winfo_width = image_label.winfo_toplevel().winfo_width()
    window_winfo_height = image_label.winfo_toplevel().winfo_height()

    if window_winfo_width <= 1 or window_winfo_height <= 1:
        return

    # Set the max height to 700 pixels
    max_height = 700
   

    # Get the height of the main window and set the image height based on it
    window_height = min(image_label.winfo_toplevel().winfo_height() * 0.75, max_height)  # e.g., 75% of window height

    # Calculate width based on the image's aspect ratio
    w_percent = window_height / float(original.size[1])
    base_width = float(original.size[0]) * w_percent

    if base_width <= 0 or window_height <= 0:
        return

    resized = original.resize((int(base_width), int(window_height)), Image.Resampling.LANCZOS)
    image = ImageTk.PhotoImage(resized)
    image_label.configure(image=image)
    image_label.image = image

    # Adjust image position
    window_width = image_label.winfo_toplevel().winfo_width()
    window_height = image_label.winfo_toplevel().winfo_height()
    
    

    # Set the title to show window dimensions
    image_label.winfo_toplevel().title(f"Width: {window_width} x Height: {window_height}")

    # Adjust the relative x position of the image based on window width
    if window_width < 1100:  # You can adjust this threshold value
        relx_value = 0.55
    else:
        relx_value = 0.55

    image_label.place(relx=relx_value, rely=0.01, anchor="ne")


def handle_bot_response(user_message, chat_frame, image_label):
    bot_response = generate(user_message)
    display_message(chat_frame, bot_response[0].replace("{user}",args.user), sender="Chatbot")
    update_image(image_label, bot_response)
    # Update the canvas scroll region after adding new chat content
    chat_frame.master.configure(scrollregion=chat_frame.master.bbox("all"))

def handle_user_input(user_message, chat_frame, image_label):
    if user_message:  # if there's a message
        display_message(chat_frame, "You: " + user_message, sender="User")
        # Update the canvas scroll region after adding new chat content
        chat_frame.master.configure(scrollregion=chat_frame.master.bbox("all"))
        
        # Start a new thread to handle bot response
        threading.Thread(target=handle_bot_response, args=(user_message, chat_frame, image_label)).start()

# def stub_function(user_input):
#     # Replace this function with your actual chatbot function later
#     if "happy" in user_input:
#         return "I'm glad you're happy!"
#     else:
#         return "I'm here to help."

initialize_application()