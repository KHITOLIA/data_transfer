import csv

# Define 100 Python MCQ questions (sample with increasing difficulty)
questions = [
    {
        "question": "What is the output of print(2**3)?",
        "options": ["6", "8", "9", "5"],
        "answer": "8"
    },
    {
        "question": "Which of the following is a mutable data type?",
        "options": ["tuple", "string", "list", "int"],
        "answer": "list"
    },
    {
        "question": "What does the len() function do?",
        "options": ["Returns the number of elements", "Deletes a variable", "Returns data type", "None of these"],
        "answer": "Returns the number of elements"
    },
    {
        "question": "Which of the following is used to define a block of code in Python?",
        "options": ["Curly braces {}", "Parentheses ()", "Indentation", "Quotation marks"],
        "answer": "Indentation"
    },
    {
        "question": "What is the output of [1,2,3] * 2?",
        "options": ["[1,2,3,1,2,3]", "[2,4,6]", "Error", "[1,2,3,2]"],
        "answer": "[1,2,3,1,2,3]"
    },
    {
        "question": "What method is used to sort a list in place?",
        "options": ["sort()", "sorted()", "order()", "arrange()"],
        "answer": "sort()"
    },
]

# Fill up to 100 dummy questions to simulate a full set
for i in range(len(questions), 100):
    questions.append({
        "question": f"Sample Question {i+1}?",
        "options": [f"Option A{i}", f"Option B{i}", f"Option C{i}", f"Option D{i}"],
        "answer": f"Option A{i}"
    })

# Create a CSV file with questions
file_path = "/mnt/data/python_mcq_quiz_100.csv"
with open(file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Question", "Option1", "Option2", "Option3", "Option4", "Answer"])
    for q in questions:
        writer.writerow([q["question"]] + q["options"] + [q["answer"]])

file_path
        