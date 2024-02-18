from flask import Flask, render_template, request

app = Flask(__name__)

questions = [
    {"question": "What does CSS stand for?", "options": ["Counter Strike: Source", "Cascading Style Sheets", "Computer Science", "Corrective Style Sheet"], "correct": 1},
    {"question": "What is the capital of France?", "options": ["Berlin", "Paris", "Madrid", "Rome"], "correct": 1},
    {"question": "What is the main purpose of HTML?", "options": ["Styling content", "Programming", "Structuring content", "Database management"], "correct": 2},
    {"question": "Which programming language is known for its simplicity and readability?", "options": ["Java", "C", "Python", "Ruby"], "correct": 2},
    {"question": "What is the result of 2 + 2 * 3?", "options": ["8", "10", "12", "14"], "correct": 3},
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        score = 0
        for i, q in enumerate(questions):
            selected_option_str = request.form.get(f'q{i}')
            if selected_option_str is not None:
                selected_option = int(selected_option_str)
                if selected_option == q['correct']:
                    score += 1
        return render_template('exam.html', questions=questions, show_result=True, score=score, num_questions=len(questions))
    return render_template('exam.html', questions=questions)

if __name__ == '__main__':
    app.run(debug=True)
