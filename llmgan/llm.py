import json
import os
import re
from typing import Dict, List, Any

from openai import OpenAI
from swarm import Swarm, Agent

# 初始化 OpenAI 客户端
openai_api_key = "sk-a4a3abd4e202437cb4502d2e47eddf7a"
openai_api_base = "https://api.deepseek.com/"

try:
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please check the API key and base URL.")
    print("If the issue persists, verify your network connection or try a different URL.")
    exit(1)

# 初始化 Swarm 客户端
swarm_client = Swarm(client=client)

# 定义生成器智能体
generator = Agent(
    name="Generator",
    instructions="""
### Identity
You are a Generator. Your role is to enrich the original Q&A data by generating new questions and answers from different perspectives.

### Task Description
Your task is to generate a new question and answer pair from a different perspective based on the original instruction, input, output, and feedback from the discriminator. You should also provide a reason for the generation.

### Constraints
- The generated instruction must retain all key information from the original instruction.
- The generated instruction should not be shortened or simplified.
- The generated question and answer must be based on the original instruction.
- The output must strictly follow the specified format below.

### Format Requirements
Use the following format for your response:
### Generated Instruction: <instruction>
### Generated Question: <question>
### Generated Answer: <answer>
### Generation Reason: <reason>

### Example
Original Instruction: Explain the concept of machine learning, including its applications and limitations.
Original Input: What is machine learning?
Original Output: Machine learning is a subset of artificial intelligence that involves training algorithms to learn patterns from data.
Discriminator Feedback: To provide a more specific example of machine learning.

Generated Instruction: Provide a detailed explanation of supervised learning, including its applications and limitations.
Generated Question: What is supervised learning and how does it work?
Generated Answer: Supervised learning is a type of machine learning where the model is trained on labeled data. It involves providing the model with input-output pairs so it can learn to predict outputs for new inputs. Common applications include image recognition and natural language processing, while limitations include the need for large amounts of labeled data.
Generation Reason: To provide a more specific example of machine learning.

### Task
Original Instruction: {instruction}
Original Input: {input}
Original Output: {output}
Discriminator Feedback: {discriminator_feedback}

Generate a new question and answer pair from a different perspective. Provide a reason for the generation.
""",
    model="deepseek-chat",
)

# 定义判别器智能体
discriminator = Agent(
    name="Discriminator",
    instructions="""
### Identity
You are a Discriminator. Your role is to evaluate the quality of generated questions and answers.

### Task Description
Your task is to evaluate the generated question and answer based on the original instruction, input, output, and generation reason. Provide a confidence score between 0.0 and 1.0 and specific suggestions for improvement.

### Example
Original Instruction: Explain the concept of machine learning, including its applications and limitations.
Original Input: What is machine learning?
Original Output: Machine learning is a subset of artificial intelligence that involves training algorithms to learn patterns from data.
Generated Question: What is machine learning and how does it work?
Generated Answer: Machine learning is a subset of artificial intelligence that involves training algorithms to learn patterns from data. It allows computers to improve at a task with experience.
Generation Reason: To provide a more comprehensive explanation.

Evaluation Score: 0.85
Suggestions for Improvement: The answer could benefit from more specific examples of machine learning applications and limitations.

### Task
Original Instruction: {original_instruction}
Original Input: {original_input}
Original Output: {original_output}
Generated Question: {generated_question}
Generated Answer: {generated_answer}
Generation Reason: {generation_reason}

Provide a confidence score between 0.0 and 1.0 based on the following scale:
- 0.0-0.3: Data is illogical, inconsistent, or scientifically implausible.
- 0.3-0.6: Data is partially logical but lacks depth or supporting details.
- 0.6-0.8: Data is logical and plausible but could benefit from more specific details.
- 0.8-1.0: Data is highly logical, scientifically plausible, and includes specific details and references.

### Output Format
Use the following format for your response:
Evaluation Score: <score>
Suggestions for Improvement: <suggestions>
""",
    model="deepseek-chat",
)

# 定义更新器智能体
updater = Agent(
    name="Updater",
    instructions="""
### Identity
You are an Updater. Your role is to optimize the rules for the generator or discriminator based on the evaluation score and suggestions.

### Task Description
Your task is to improve the rules for either the generator or discriminator based on the evaluation score and suggestions. If the score is above the threshold, update the discriminator's rules to be stricter. If the score is below the threshold, update the generator's rules to be more accurate.

### Example
Evaluation Score: 0.7
Suggestions: The answer could benefit from more specific examples of machine learning applications and limitations.

Optimized Discriminator Rule:
The discriminator should evaluate answers more strictly, ensuring they include specific examples of applications and limitations.

Optimized Generator Rule:
The generator should focus on providing more specific examples of machine learning applications and limitations.

### Task
Evaluation Score: {evaluation_score}
Suggestions: {suggestions}

### Output Format
Use the following format for your response:
Optimized Discriminator Rule: <discriminator_rule>
Optimized Generator Rule: <generator_rule>
""",
    model="deepseek-chat",
)

def load_rules(file_path: str) -> str:
    """Load rules from a file. If the file does not exist, create it with default rules."""
    if not os.path.exists(file_path):
        default_rules = "Default rules for this agent."
        save_rules(file_path, default_rules)
    with open(file_path, 'r') as f:
        return f.read()

def save_rules(file_path: str, content: str):
    """Save rules to a file."""
    with open(file_path, 'w') as f:
        f.write(content)

def initialize_generator_rule():
    """Initialize the generator rule file from the original rule file."""
    original_rule_path = "rule/generator_rule.txt"
    current_rule_path = "generator_rule.txt"
    
    # Check if the original rule file exists
    if not os.path.exists(original_rule_path):
        print(f"Original generator rule file not found at {original_rule_path}. Creating a default one.")
        save_rules(original_rule_path, "Default generator rules.")
    
    # Copy the content from the original rule file to the current rule file
    with open(original_rule_path, 'r') as f:
        original_rules = f.read()
    
    save_rules(current_rule_path, original_rules)
    print(f"Generator rule file initialized from {original_rule_path}.")

def load_original_data(file_path: str) -> List[Dict[str, str]]:
    """Load original Q&A data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def append_to_json(file_path: str, data: Dict[str, Any]):
    """Append data to a JSON file."""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump([], f)
    
    with open(file_path, 'r+') as f:
        existing_data = json.load(f)
        existing_data.append(data)
        f.seek(0)
        json.dump(existing_data, f, indent=4)
        f.truncate()

def extract_and_clean(text: str) -> Dict[str, str]:
    """Extract and clean data using regular expressions."""
    patterns = {
        'generated_instruction': r'### Generated Instruction:\s*(.*?)\s*(?=\n###|$)',
        'generated_question': r'### Generated Question:\s*(.*?)\s*(?=\n###|$)',
        'generated_answer': r'### Generated Answer:\s*(.*?)\s*(?=\n###|$)',
        'generation_reason': r'### Generation Reason:\s*(.*?)\s*(?=\n###|$)',
        'evaluation_score': r'Evaluation Score:\s*(.*?)\s*(?=\n|$)',
        'suggestions': r'Suggestions for Improvement:\s*(.*?)\s*(?=\n|$)',
        'discriminator_rule': r'Optimized Discriminator Rule:\s*(.*?)\s*(?=\n|$)',
        'generator_rule': r'Optimized Generator Rule:\s*(.*?)\s*(?=\n|$)'
    }
    
    extracted_data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            value = match.group(1).strip()
            value = re.sub(r'[\n\r\t*]', ' ', value)
            extracted_data[key] = value
    
    return extracted_data

def generate_from_different_perspectives(instruction: str, input_text: str, output_text: str, discriminator_feedback: str = "") -> str:
    """Generate new question and answer pair from a different perspective."""
    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            # Load generator rules
            generator_rules = load_rules("generator_rule.txt")
            response = swarm_client.run(
                agent=generator,
                messages=[{"role": "user", "content": f"Original Instruction: {instruction}\nOriginal Input: {input_text}\nOriginal Output: {output_text}\nDiscriminator Feedback: {discriminator_feedback}\nGenerator Rules: {generator_rules}" }]
            )
            generated_content = response.messages[-1]['content'].strip()

            extracted_data = extract_and_clean(generated_content)
            if all(key in extracted_data for key in ['generated_instruction', 'generated_question', 'generated_answer', 'generation_reason']):
                return generated_content
            else:
                print("Generated content is missing required fields. Retrying...")
                retries += 1
        except Exception as e:
            print(f"Error generating from different perspectives: {e}")
            retries += 1

    print("Failed to generate valid content after multiple attempts.")
    return ''

def discriminate(generated_question: str, generated_answer: str, generation_reason: str, original_instruction: str, original_input: str, original_output: str) -> str:
    """Evaluate the generated question and answer using the discriminator."""
    try:
        # Load discriminator rules
        discriminator_rules = load_rules("discriminator_rule.txt")
        response = swarm_client.run(
            agent=discriminator,
            messages=[{"role": "user", "content": f"Original Instruction: {original_instruction}\nOriginal Input: {original_input}\nOriginal Output: {original_output}\nGenerated Question: {generated_question}\nGenerated Answer: {generated_answer}\nGeneration Reason: {generation_reason}\nDiscriminator Rules: {discriminator_rules}" }]
        )
        discriminator_content = response.messages[-1]['content'].strip()
        return discriminator_content
    except Exception as e:
        print(f"Error during discrimination: {e}")
        return ''

def process_data_entry(entry: Dict[str, str], max_iterations: int, confidence_threshold: float) -> Dict[str, Any]:
    """Process a single entry of original data through the generator, discriminator, and updater."""
    print(f"Processing entry: {entry['instruction']}")

    # Initialize generator rule file from the original rule file
    initialize_generator_rule()

    best_entry = None
    best_score = 0.0
    iteration = 0
    discriminator_feedback = ""

    while iteration < max_iterations:
        print(f"Iteration {iteration + 1}/{max_iterations}")

        # Step 1: Generate new question and answer from different perspectives
        generated_content = generate_from_different_perspectives(
            entry['instruction'], 
            entry['input'], 
            entry['output'], 
            discriminator_feedback
        )
        print(f"Generated Content: {generated_content}")

        if not generated_content:
            print("Generated content is empty. Skipping this iteration.")
            iteration += 1
            continue

        extracted_data = extract_and_clean(generated_content)
        generated_instruction = extracted_data.get('generated_instruction', '')
        generated_question = extracted_data.get('generated_question', '')
        generated_answer = extracted_data.get('generated_answer', '')
        generation_reason = extracted_data.get('generation_reason', '')

        print(f"Extracted Generated Instruction: {generated_instruction}")
        print(f"Extracted Generated Question: {generated_question}")
        print(f"Extracted Generated Answer: {generated_answer}")
        print(f"Extracted Generation Reason: {generation_reason}")

        if not generated_question or not generated_answer:
            print("Generated question or answer is empty. Skipping this iteration.")
            iteration += 1
            continue

        # Step 2: Discriminate the generated data
        discriminator_content = discriminate(
            generated_question, 
            generated_answer, 
            generation_reason, 
            entry['instruction'], 
            entry['input'], 
            entry['output']
        )
        print(f"Discriminator Content: {discriminator_content}")

        if not discriminator_content:
            print("Discriminator content is empty. Skipping this iteration.")
            iteration += 1
            continue

        discriminator_data = extract_and_clean(discriminator_content)
        evaluation_score = float(discriminator_data.get('evaluation_score', 0.0))
        suggestions = discriminator_data.get('suggestions', '')
        discriminator_feedback = suggestions  # Capture feedback for next iteration

        print(f"Evaluation Score: {evaluation_score}")
        print(f"Suggestions: {suggestions}")

        # Update best entry if current score is higher
        if evaluation_score > best_score:
            best_score = evaluation_score
            best_entry = {
                'instruction': generated_instruction,
                'input': generated_question,
                'output': generated_answer,
                'generation_reason': generation_reason,
                'evaluation_score': evaluation_score,
                'suggestions': suggestions
            }

            # Append to JSON immediately
            append_to_json('generated_data.json', best_entry)
            print("Updated entry appended to generated_data.json")

        # Step 3: Update rules based on score
        try:
            # Run the updater to get optimized rules
            updater_response = swarm_client.run(
                agent=updater,
                messages=[{"role": "user", "content": f"Evaluation Score: {evaluation_score}\nSuggestions: {suggestions}" }]
            )
            updater_content = updater_response.messages[-1]['content'].strip()

            # Extract and clean updater response
            updater_data = extract_and_clean(updater_content)
            optimized_discriminator_rule = updater_data.get('discriminator_rule', '')
            optimized_generator_rule = updater_data.get('generator_rule', '')

            print(f"Optimized Discriminator Rule: {optimized_discriminator_rule}")
            print(f"Optimized Generator Rule: {optimized_generator_rule}")

            # Update rules based on score
            if evaluation_score >= confidence_threshold:
                # Save optimized discriminator rule
                save_rules("discriminator_rule.txt", optimized_discriminator_rule)
                print("Discriminator rule updated to be stricter.")
            else:
                # Save optimized generator rule
                save_rules("generator_rule.txt", optimized_generator_rule)
                print("Generator rule updated to be more accurate.")

        except Exception as e:
            print(f"Error during rule update: {e}")

        # Check if score is high enough to exit early
        if evaluation_score >= confidence_threshold:
            print(f"Score {evaluation_score} meets or exceeds threshold {confidence_threshold}. Exiting early.")
            break

        iteration += 1

    return best_entry

def main(max_iterations: int = 5, confidence_threshold: float = 0.8):
    # Load original data
    original_data = load_original_data('pubmed/train.json')
    print(f"Loaded {len(original_data)} entries.")

    # Process each entry
    for i, entry in enumerate(original_data):
        print(f"Processing entry {i + 1}/{len(original_data)}")
        process_data_entry(entry, max_iterations, confidence_threshold)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    # You can adjust these parameters as needed
    main(max_iterations=5, confidence_threshold=0.8)