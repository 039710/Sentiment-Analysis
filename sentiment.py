from sentiment_logic import preprocessing_input,predict_input
from flask import Flask, render_template, request

app = Flask(__name__)
sameInput = 0
isinya =""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_review_response():
    global sameInput
    global isinya
    inputText = request.args.get('msg')
    if (inputText == ''):
        life = False
        return str('Please input the reviewe text to predict')
    if(inputText!=isinya):
        isinya = inputText
        sameInput=0
        return str(predict_input(preprocessing_input(inputText)))
    elif (inputText==isinya):
        sameInput += 1
        return(predict_input(preprocessing_input(inputText)))

if __name__ == "__main__":
    app.run()
