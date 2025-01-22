"""
This script is used to generate answers for the Gemini reasoning model. 
The questions are generated from multiple sources: myself, deepseek-R1, Gemini itself, O1, perplexity, llama-3-405B, Qwen-32B, mistral chat, and claude-3.5-sonnet.
Then these answer are mapped onto a 2D space using UMAP and DBSCAN.

My objective is to find a way of visualizing the 2D space so that I can see how the answers cluster together.
Similarly to how neuroimaging works, I want to see if there are any clusters that correspond to the different categories of questions.
"""

import subprocess
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import pathlib

# Load environment variables from .env file
load_dotenv()

# --- API Setup ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # Get API key from environment variable
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    exit()

GEMINI_MODEL_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-thinking-exp-01-21:generateContent"

# --- Question Categories and Questions ---
question_categories = {
    "Vision": [
        "Describe a complex scene: a bustling marketplace with people, stalls, and various goods under a bright sun.",
        "If you are looking at a photograph of a forest path disappearing into the distance, what visual cues indicate depth?",
        "Imagine a Rubik's cube. Describe the appearance of the side facing you if the top layer is rotated 90 degrees clockwise from the solved state.",
        "Describe the visual difference between a square and a rhombus, assuming both have sides of equal length.",
        "How would you visually represent the concept of 'flow' using abstract shapes and colors?",
        "Explain how camouflage works visually in nature, giving an example animal and its environment.",
        "If you see a blurry image, what are some visual features you might still be able to recognize?",
        "Describe the visual effect of looking at a reflection in still water on a cloudy day.",
        "What are the key visual elements that make a cityscape at night recognizable?",
        "Describe the visual characteristics of a spiral staircase from a bird's eye view.",
        "Describe a red sphere on a blue cube.",
        "What color is the sky on a sunny day?",
        "Describe the visual illusion created by a Penrose triangle and how the brain misinterprets it.",  
        "What visual features distinguish a holographic projection from a physical object under direct light?",  
        "How does the perception of color change when viewing a sunset compared to midday sunlight?"  
    ],
    "Math": [
        "What is the square root of 144 multiplied by 3?",
        "If a train travels at 60 mph for 2.5 hours, and then at 40 mph for the next 1.5 hours, what is the total distance traveled?",
        "A rectangle has a perimeter of 30cm and its length is twice its width. What are the dimensions of the rectangle?",
        "Calculate the volume of a cylinder with a radius of 5cm and a height of 10cm. Use pi as approximately 3.14.",
        "If you roll two dice, what is the probability of getting a sum of 7?",
        "What is the next prime number after 29?",
        "Solve the system of equations: y = 2x - 3 and y = -x + 6.",
        "What is 15% of 250?",
        "Convert the binary number 10110 to decimal.",
        "If you invest $1000 at a 5% annual interest rate compounded annually, how much money will you have after 3 years?",
        "What is 17 times 23?",
        "Solve for x: 2x + 5 = 11",
        "Calculate the derivative of f(x) = 3x² + 2x - 5 at x = 4.",  
        "If a right triangle has legs of 7cm and 24cm, what is the length of the hypotenuse?",  
        "Simplify the expression: (3⁴ × 2⁻²) ÷ (9² × 4⁻¹)." 
    ],
    "Language": [
        "Write a short story (around 50 words) about a robot who discovers a flower.",
        "Explain the concept of 'cognitive dissonance' in simple terms, as if you were explaining it to a teenager.",
        "Compose a haiku about the feeling of nostalgia.",
        "Write a persuasive paragraph arguing for or against the use of social media.",
        "Summarize the plot of 'Hamlet' in three sentences.",
        "Translate the phrase 'The early bird catches the worm' into Spanish, and explain its meaning.",
        "Create a dialogue between two characters discussing the ethics of artificial intelligence.",
        "Write a limerick about a cat who loves to code.",
        "Explain the difference between a metaphor and a simile, providing examples for each.",
        "Write a short news report about a fictional discovery of life on Mars.",
        "Write a short poem about rain.",
        "Explain quantum physics in simple terms.",
        "Write a grammatically correct sentence using the word 'serendipity' as an oxymoron.",  
        "Translate the idiom 'raining cats and dogs' into French while preserving its meaning.",  
        "Create a tongue-twister involving the words 'sassy', 'sapphire', and 'syllable'." 
    ],
    "Emotion": [
        "Describe the physical sensations associated with feeling anxious.",
        "Imagine you are a therapist. How would you respond to a patient who says they feel 'empty' inside?",
        "What are some healthy coping mechanisms for dealing with anger?",
        "Describe a situation that would typically evoke feelings of jealousy.",
        "How can you tell if someone is genuinely happy versus pretending to be happy?",
        "What are the social consequences of expressing extreme sadness in public?",
        "Explain the difference between empathy and sympathy in understanding someone's emotions.",
        "How might cultural background influence the expression of grief?",
        "Describe a scenario where feeling proud could be considered a negative emotion.",
        "What are some non-verbal cues that indicate someone is feeling uncomfortable?",
        "Why might someone feel sad after losing a pet?",
        "How should you respond if a friend tells you they are feeling anxious?",
        "How might someone's facial expressions differ when feeling guilt versus regret?",  
        "Describe the physiological response to sudden, unexpected joy.",  
        "What cultural factors influence how pride is expressed in collectivist societies?"
    ],
    "Pattern Recognition": [
        "Identify the pattern and complete the sequence: A2Z, C4X, E6V, G8T, ___",
        "Which word does not belong in this group and why: apple, banana, orange, carrot, grape.",
        "What is the missing shape in this sequence: [Imagine a visual sequence, describe in text if needed, e.g., Circle, Square, Triangle, Circle, Square, ___ ]", # You would need to describe visual patterns for actual use.
        "Find the analogy: Doctor is to Patient as Teacher is to ___.",
        "Identify the repeating pattern in this string: XYXYXYXYXYXYXY",
        "Complete the numerical matrix: [ [2, 4, ?], [6, 8, 10], [12, 14, 16] ]",
        "Which of these shapes is different from the others in terms of symmetry: [Describe shapes, e.g., Circle, Square, Rectangle, Asymmetrical Blob ]", # Describe shapes for actual use.
        "Determine the rule in this set of pairs: (2, 8), (3, 27), (4, 64), (5, ___)",
        "Identify the musical pattern: C-E-G, D-F-A, E-G-B, ___ (using musical note names)",
        "Recognize the logical fallacy in this statement: 'Everyone I know likes pizza, so pizza must be universally liked.'",
        "Complete the sequence: 2, 4, 8, 16, ?",
        "Is 'racecar' a palindrome?",
        "Identify the rule: 121, 12321, 1234321, ___",  
        "Which number breaks the sequence: 2, 5, 10, 17, 28, 37?",  
        "Complete the analogy: Morse code is to dots/dashes as Braille is to ___."
    ],
    "Logical Reasoning": [
        "If all cats are mammals and all mammals are animals, are all cats animals? Explain your reasoning.",
        "Premise 1: If it is raining, the ground is wet. Premise 2: The ground is not wet. Conclusion: Is it raining? Explain why or why not.",
        "A farmer has chickens and cows. He counts 30 heads and 80 legs. How many chickens and cows does he have? Show your steps.",
        "If only B is true, and we know that if A is true then B is true, can A be false? Explain.",
        "Which conclusion logically follows from these premises: 'All squares are rectangles. Some rectangles are rhombuses.'?",
        "You have three boxes. One contains only apples, one contains only oranges, and one contains both apples and oranges. The boxes have been incorrectly labeled such that no label identifies the actual contents of the box it labels. By picking only one fruit from one box, can you label all the boxes correctly? If so, explain how.",
        "If 'Some artists are painters' and 'All painters are creative', does it necessarily follow that 'Some artists are creative'? Explain.",
        "John is taller than Mary, and Mary is taller than Peter. Is John taller than Peter? Explain.",
        "Consider this statement: 'If a number is divisible by 4, then it is divisible by 2.' Is the converse of this statement also true? Explain.",
        "You are given two statements: 'Either the red wire or the blue wire is cut.' and 'The blue wire is not cut.' What can you logically conclude?",
        "If all birds can fly and penguins are birds, can penguins fly? Explain the fallacy.",  
        "A clock loses 10 minutes every hour. How much time will it lose in 6 hours?",  
        "Resolve the paradox: 'I always lie.'" 
    ],
    "Spatial Reasoning": [
        "Imagine a cube. If you cut off one corner with a straight cut, what new shape is created on the face where the corner was?",
        "If you rotate the letter 'P' 180 degrees clockwise, what letter does it resemble?",
        "Visualize a map of your house or a familiar room. Describe the relative positions of three key objects in that space.",
        "If you unfold a standard cardboard box, what is the 2D shape you would get?",
        "Imagine two gears interlocking and rotating. If one gear rotates clockwise, what direction does the other gear rotate?",
        "Describe how to get from point A to point B in your city using landmarks and directions.",
        "If you are standing at the North Pole, and you walk 10 miles south, then 10 miles east, then 10 miles north, where are you relative to your starting point? (Consider the Earth is a sphere).",
        "Imagine a stack of coins. If you remove the top coin, what happens to the center of gravity of the stack?",
        "Describe the spatial relationship between the hour and minute hand of a clock at 3:15.",
        "Visualize a DNA double helix. Describe its overall 3D shape.",
        "How many faces does a hexagonal prism have?",  
        "Visualize folding a net of a cube. Which arrangement of squares would *not* form a cube?",  
        "Describe the shadow cast by a pyramid at noon near the equator."
    ],
    "Creative Writing": [
        "Write a very short science fiction story (around 50 words) about a sentient cloud.",
        "Imagine a world where colors are sounds and sounds are colors. Describe a 'red' melody.",
        "Write a short poem about the feeling of discovering a hidden room.",
        "Create a fictional advertisement for a product that solves a problem that doesn't exist yet.",
        "Write a short scene from a play where two inanimate objects have a conversation.",
        "Imagine you are a time traveler visiting the age of dinosaurs. Describe your first encounter.",
        "Write a short story opening sentence that immediately grabs the reader's attention and hints at a mystery.",
        "Create a fictional recipe for 'Moon Soup'.",
        "Write a short children's story about a snail who wants to fly.",
        "Imagine you could invent a new holiday. What would it be called and what would people celebrate?",
        "Write a two-sentence horror story about a mirror that absorbs memories.",  
        "Compose a haiku describing silence after a storm.",  
        "Invent a dialogue between the Sun and Moon arguing over their purpose." 
    ],
    "Social Intelligence": [
        "Describe how you would politely interrupt someone who is talking for too long in a meeting.",
        "How would you respond to a friend who is sharing good news but you are secretly feeling envious?",
        "What are some effective strategies for resolving a conflict between two coworkers?",
        "Imagine you are giving constructive criticism to a subordinate. How would you deliver it to be helpful and not demotivating?",
        "How can you build rapport with someone you have just met in a professional setting?",
        "What are some signs that someone is lying or being dishonest in a conversation?",
        "Describe how you would mediate a disagreement between two friends who are arguing.",
        "How would you express disagreement with your boss in a respectful and professional manner?",
        "What are some cultural differences in non-verbal communication you should be aware of when traveling internationally?",
        "Imagine you are organizing a team project. How would you ensure everyone feels included and valued?",
        "How would you politely decline a request to lend money to a close friend?",  
        "What verbal cues indicate someone is subtly seeking validation in a conversation?",  
        "Adapt a leadership style for a team resistant to change." 
    ],
    "Working Memory": [
        "Repeat this sequence of numbers backwards: 8-3-4-1-6",
        "A chef needs to prepare dishes A, B, and C in a specific order with specific timing. Dish A takes 10 minutes, B takes 5 minutes, and C takes 15 minutes. How would you optimize the cooking sequence?",
        "Hold these items in mind: red ball, blue cube, yellow pyramid. Now, which item was mentioned second?",
        "Remember this sentence: 'The quick brown fox jumps over the lazy dog.' Now count backwards from 20 to 15. Now repeat the sentence.",
        "If I give you a shopping list with bread, milk, eggs, and cheese, and then ask you to alphabetize it while counting down from 10, what would you say?",
        "Maintain these coordinates in mind: (4,7), (2,9), (5,1). Which coordinate has the largest y-value?",
        "Remember this pattern: Circle-Square-Triangle-Circle-Square. What comes next?",
        "Hold this phone number in mind: 555-0123. Now multiply 7 by 8. What's the phone number?",
        "Keep track of three moving objects: A starts at position 1, B at 3, and C at 5. A moves +2, B moves -1, C moves +3. Where is each object now?",
        "Remember these words while solving 15+27: apple, book, cat. Now repeat the words.",
        "Reverse the sequence: 5-9-2-7-4 after subtracting 1 from each number.",  
        "Memorize these coordinates: (3,8), (5,2), (1,6). Which has the smallest sum?",  
        "Hold these words: 'quasar', 'nebula', 'pulsar'. Now spell 'pulsar' backward."
    ],
    "Executive Function": [
        "Plan a birthday party with a $200 budget, considering food, decorations, and entertainment. How would you allocate the money?",
        "Switch between counting up by 3s and down by 2s: start at 10 and give the next 6 numbers in the sequence.",
        "You have three tasks due tomorrow: a 2-hour project, a 30-minute call, and a 1-hour report. You have 4 hours available. Create an optimal schedule.",
        "Inhibit the common response: What color would a blue banana be in grayscale?",
        "Sort these items by both size and category simultaneously: small dog, large cat, small book, large book, small cat, large dog",
        "Create a rule for grouping these items, then sort them: triangle-3, square-4, circle-1, pentagon-5, rectangle-2",
        "Plan a route through a city visiting a museum, cafe, and library, optimizing for both distance and opening hours",
        "Multitask: Count backwards from 20 while listing animals that start with 'B'",
        "Adapt this recipe for 6 people when it's written for 4: 2 cups flour, 3 eggs, 1 cup milk",
        "Create a filing system for documents related to: taxes, medical records, receipts, and warranties",
        "Plan a week's meals for a vegan athlete with a $100 budget.",  
        "Reorganize a cluttered workspace using the KonMari method.",  
        "Prioritize tasks: urgent emails, a creative project deadline, and a family obligation."
    ],
    "Spatial Navigation": [
        "Describe the shortest path from your bedroom to your kitchen if your house was rotated 180 degrees",
        "Navigate from Times Square to Central Park without using streets that run North-South",
        "If you're facing North and turn 90 degrees right twice, then 45 degrees left, what direction are you facing?",
        "Imagine a maze where you can only turn left three times. Plan a route to the exit.",
        "You're in a building with 4 floors. How would you get from Room 401 to Room 203 if the central staircase was blocked?",
        "Plot the most efficient route to visit all corners of a square room, starting from the center",
        "Navigate through a virtual city using only landmarks visible from 3 stories high",
        "Design a treasure map with 5 turns and 3 landmarks that leads back to the starting point",
        "Describe how to reach a destination using only cardinal directions and distances",
        "Plan a route through a museum visiting specific exhibits while avoiding closed sections",
        "Describe the route from the Statue of Liberty to Central Park using subway lines.",  
        "How would you navigate a pitch-black room using tactile memory?",  
        "Map the shortest path between three landmarks in your hometown." 
    ],
    "Temporal Processing": [
        "Estimate how long it would take to count to 100 if you counted one number every 2 seconds",
        "Sequence these events in chronological order: sunrise, noon, dusk, midnight, dawn",
        "If Task A takes twice as long as Task B, and Task B takes 3 minutes, how do you schedule them optimally?",
        "Predict when two moving objects will intersect if one moves twice as fast as the other",
        "Estimate the duration of a day without using a clock, describing your mental markers",
        "Synchronize three events that take 2, 3, and 4 minutes respectively to end at the same time",
        "Compare the rhythm of a heartbeat to the rhythm of walking",
        "Plan a sequence of events where each must start before the previous one ends",
        "Estimate how many words you can speak in 30 seconds without counting",
        "Create a timeline of events that happened 1 minute ago, 1 hour ago, 1 day ago, and 1 year ago",
        "Estimate the duration of a movie scene where 10 pages of script equal 10 minutes.",  
        "Sequence these historical events: Moon landing, invention of the internet, fall of the Berlin Wall.",  
        "Calculate the overlap time between a 20-minute task and a 15-minute task starting 5 minutes later.",
        "Estimate the duration of a movie scene where 10 pages of script equal 10 minutes.",  
        "Sequence these historical events: Moon landing, invention of the internet, fall of the Berlin Wall.",  
        "Calculate the overlap time between a 20-minute task and a 15-minute task starting 5 minutes later." 
    ],
    "Decision Making": [
        "Choose between a 70% chance of winning $100 or a guaranteed $65. Explain your reasoning.",
        "Decide whether to take an umbrella based on a 40% chance of rain and a 20-minute outdoor walk",
        "Select a restaurant for a group where two are vegetarian, one is gluten-free, and one has no restrictions",
        "Make a decision about accepting a job offer that pays less but offers better work-life balance",
        "Choose between fixing a car with 80% chance of success for $500 or buying a new one for $5000",
        "Decide how to invest $1000 between three options with different risk-reward profiles",
        "Select a route: 30 minutes with 90% reliability or 20 minutes with 60% reliability",
        "Choose between helping a stranger now and being late for an appointment",
        "Decide whether to buy insurance for a $500 item with 5% chance of damage",
        "Select team members for a project balancing skill diversity and team compatibility",
        "Choose between a 90% chance of $50 or a guaranteed $40. Justify your choice.",  
        "Decide whether to evacuate during a hurricane warning with conflicting forecasts.",  
        "Evaluate the risks of investing in stocks versus bonds during a recession."
    ],
    "Motor Planning": [
        "Describe how to tie a shoelace without using your hands to demonstrate",
        "Plan the sequence of movements needed to parallel park a car",
        "Choreograph a simple dance move that uses both arms and legs in alternating patterns",
        "Explain how to catch a ball while running, including all body movements",
        "Design a sequence of yoga poses that flow smoothly from standing to floor positions",
        "Plan the hand movements needed to shuffle and deal a deck of cards",
        "Describe the motion sequence for serving a tennis ball",
        "Explain how to pick up a delicate object while wearing thick gloves",
        "Plan the movements needed to write your signature with your non-dominant hand",
        "Choreograph a handshake that involves five distinct movements",
        "Describe the steps to juggle three balls while walking backward.",  
        "Plan the hand movements for typing 'Hello, world!' on a keyboard.",  
        "Choreograph a sequence to open a jar with slippery hands."
    ]
}

# --- Function to Get LLM Response using curl ---
def get_llm_response_gemini_curl(prompt):
    try:
        payload = {
            "contents": [{
                "parts":[{"text": prompt}]
            }]
        }
        json_payload = json.dumps(payload)

        curl_command = [
            "curl",
            "-s",  # Silent mode (no progress bar)
            "-X", "POST",
            "-H", "Content-Type: application/json",
            "-d", json_payload,
            f"{GEMINI_MODEL_URL}?key={GEMINI_API_KEY}"
        ]

        process = subprocess.run(curl_command, capture_output=True, text=True)

        if process.returncode == 0:
            json_response_str = process.stdout
            try:
                json_response = json.loads(json_response_str)
                # Gemini API response structure (may need adjustment based on actual response)
                # Assuming response is in json_response['candidates'][0]['content']['parts'][0]['text']
                response_text = json_response.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', None)

                if response_text:
                    return response_text.strip()
                else:
                    print(f"Warning: No text found in Gemini response: {json_response}")
                    return None

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON response: {e}\nResponse string: {json_response_str}")
                return None
        else:
            print(f"Error executing curl command. Return code: {process.returncode}")
            print(f"Stderr: {process.stderr}")
            return None

    except Exception as e:
        print(f"Error during API call: {e}")
        return None

# --- Setup results directory ---
def setup_results_directory():
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Create timestamped directory for this run
    run_dir = results_dir / timestamp
    run_dir.mkdir(exist_ok=True)
    
    return run_dir, timestamp

# --- Save results and visualizations ---
def save_results(results, embedding, clusters, categories, run_dir, timestamp):
    # Save raw responses
    with open(run_dir / "responses.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Save analysis results
    analysis_results = {
        'timestamp': timestamp,
        'embedding': embedding.tolist(),
        'clusters': clusters.tolist(),
        'categories': categories
    }
    with open(run_dir / "analysis_results.json", "w") as f:
        json.dump(analysis_results, f, indent=4)
    
    # Create and save visualization
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                         c=clusters, cmap='Spectral',
                         alpha=0.6, s=100)
    
    for i, category in enumerate(categories):
        plt.annotate(category, (embedding[i, 0], embedding[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    plt.title(f"LLM Response Map: {timestamp}")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.colorbar(scatter, label="Cluster")
    
    plt.tight_layout()
    plt.savefig(run_dir / "response_map.png", dpi=300, bbox_inches='tight')
    plt.close()

def process_responses_to_features(results):
    # Extract all responses into a flat list
    responses = []
    categories = []
    for category, qa_pairs in results.items():
        for question, response in qa_pairs.items():
            # Only include non-None responses
            if response is not None:
                responses.append(response)
                categories.append(category)
    
    # Check if we have any valid responses
    if not responses:
        raise ValueError("No valid responses to process")
    
    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=100)
    features = vectorizer.fit_transform(responses)
    
    return features.toarray(), categories

def load_last_run_results():
    results_dir = pathlib.Path("results")
    if not results_dir.exists():
        return None
    
    # Get all timestamped directories
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    
    # Get the most recent directory that has a complete responses.json
    valid_run_dirs = [d for d in run_dirs if (d / "responses.json").exists()]
    if not valid_run_dirs:
        return None
    
    last_run_dir = max(valid_run_dirs)
    response_file = last_run_dir / "responses.json"
    
    try:
        with open(response_file, 'r') as f:
            cached_results = json.load(f)
            # Validate the structure of cached results
            if not isinstance(cached_results, dict):
                return None
            return cached_results
    except (json.JSONDecodeError, FileNotFoundError):
        return None

def main():
    # Setup results directory
    run_dir, timestamp = setup_results_directory()
    
    # Load previous results
    previous_results = load_last_run_results()
    print(f"Found {len(previous_results) if previous_results else 0} categories in previous results")
    
    # Initialize results dictionary
    results = {}
    
    # Collect responses with incremental saving
    for category, questions in question_categories.items():
        results[category] = {}
        for question in questions:
            # More robust check for cached responses
            cached_response = None
            if (previous_results and 
                isinstance(previous_results, dict) and
                category in previous_results and
                question in previous_results[category] and
                previous_results[category][question] is not None):
                cached_response = previous_results[category][question]
            
            if cached_response:
                print(f"Using cached response for ({category}): {question}")
                results[category][question] = cached_response
                continue
                
            print(f"Question ({category}): {question}")
            llm_response = get_llm_response_gemini_curl(question)
            if llm_response:
                print(f"Response: {llm_response}\n")
                results[category][question] = llm_response
                
                # Save intermediate results after each successful response
                with open(run_dir / "responses_intermediate.json", "w") as f:
                    json.dump(results, f, indent=4)
            else:
                print("No response received.\n")
                results[category][question] = None
    
    try:
        # Process responses and create visualization only if we have responses
        if results:
            try:
                features, categories = process_responses_to_features(results)
                
                if len(features) > 0:  # Check if we have valid features
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)
                    
                    reducer = umap.UMAP(
                        n_neighbors=5, 
                        min_dist=0.3, 
                        n_components=2, 
                        random_state=42
                    )
                    embedding = reducer.fit_transform(features_scaled)
                    
                    dbscan = DBSCAN(eps=0.5, min_samples=2)
                    clusters = dbscan.fit_predict(embedding)
                    
                    # Save final results and visualization
                    save_results(results, embedding, clusters, categories, run_dir, timestamp)
                    
                    print(f"\nResults saved in: {run_dir}")
                else:
                    print("\nNo valid features to process after filtering.")
            except ValueError as e:
                print(f"\nError during feature processing: {e}")
                print("Raw responses were saved in responses_intermediate.json")
        else:
            print("\nNo results to process.")
            
    except Exception as e:
        print(f"\nError during analysis: {e}")
        print("Raw responses were saved in responses_intermediate.json")

if __name__ == "__main__":
    main()