import json

def convert_question_format(input_data):
    output = []
    question_id = 0
    
    for item in input_data:
        # Extract metadata that applies to all questions in this item
        institute = item.get("Institute")
        survey = item.get("Survey")
        date = item.get("Date")
        file_name = item.get("File")
        
        for question in item.get("Questions", []):
            # Create new question object with lowercase keys
            new_question = {
                "id": question_id,  # Add the unique ID
                "question": question.get("Question"),
                "answers": question.get("Answers", {}),
                "distribution": question.get("Distribution", {}),
                "institute": institute,
                "survey": survey,
                "survey_qid": question.get("Variable_Name"),
                "date": date,
                "file": file_name
            }
            
            # Remove None values (for missing fields) except 'id'
            new_question = {k: v for k, v in new_question.items() if v is not None or k == "id"}
            
            output.append(new_question)
            question_id += 1  # Increment the ID for the next question
    
    return output

# Example usage:
if __name__ == "__main__":
    # Load the input JSON
    with open(r".\1_data\questions_filled.json", "r", encoding="utf-8") as f:
        input_data = json.load(f)
    
    # Convert the format
    output_data = convert_question_format(input_data)
    
    # Save the output JSON
    with open(r".\1_data\converted_questions.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Conversion completed. {len(output_data)} questions processed. Output saved to converted_questions.json")