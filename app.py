from flask import Flask, render_template, request, jsonify, render_template_string
from flask_cors import CORS
from newspaper import Article
from transformers import pipeline
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.preprocessing import LabelEncoder
import joblib
import mysql.connector
from flask import send_file
from reportlab.pdfgen import canvas
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak, Paragraph
from nltk.tokenize import sent_tokenize
from reportlab.platypus import Spacer
from reportlab.platypus.flowables import KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
import datetime



app = Flask(__name__, template_folder='templates')
CORS(app)
chat_history = []

mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '9553641651',
    'database': 'articles'
}

def insert_question_and_answer(question, answer,timestamp):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()

        # SQL query to insert a new record into the 'supplychain' table
        query = "INSERT INTO supplychain143 (question, answer, timestamp) VALUES (%s, %s, %s);"
        values = (question, answer,timestamp)

        # Execute the query
        cursor.execute(query, values)

        # Commit the changes
        connection.commit()

        # Close the cursor and connection
        cursor.close()
        connection.close()

        print("Record inserted successfully!")
    except Exception as e:
        print("Error inserting record:", str(e))

def retrieve_article_content(timestamp):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()

        # SQL query to retrieve article content based on the question
        query = "SELECT question, answer FROM supplychain143 WHERE timestamp = %s;"
        values = (timestamp,)

        # Execute the query
        cursor.execute(query, values)

        # Fetch the results
        results = cursor.fetchall()
        # Close the cursor and connection
        cursor.close()
        connection.close()

        return results
    except Exception as e:
        print("Error retrieving article content:", str(e))
        return None

def scrape_news_content(url):
    # ... (Your existing implementation)
    try:
      article = Article(url)
      article.download()
      article.parse()

      title = article.title
      content = article.text

      return content
 # Remove leading/trailing whitespaces
    except Exception as e:
      return "Error: " + str(e)
      

def summarize_with_t5(article_content, classification, model, tokenizer, device):
    # ... (Your existing implementation)
    article_content = str(article_content)
    prompt = "Classification: " + str(classification) + "\n"
    if not article_content or article_content == "nan":
        return "", ""
    if classification == "risks":
        prompt = "summarize the key supply chain risks: "
    elif classification == "opportunities":
        prompt = "summarize the key supply chain opportunities: "
    elif classification == "neither":
        print("Nooo")
        return "None", "None"

    input_text = prompt + article_content
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    model = model.to(device)  #/ Move the model to the correct device
    summary_ids = model.generate(input_ids.to(device), max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(summary)
    if classification in ["risks", "opportunities"]:
        if classification == "risks":
            return summary, "None"
        elif classification == "opportunities":
            return "None", summary
        else:
          return None,None
    else:
        return ("This article is not classified as related to the supply chain.")


def classify_and_summarize(input_text, cls_model, tokenizer_cls, label_encoder, model_summ, tokenizer_summ, device):
    # ... (Your existing implementation)
    results = []
    request_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    input_text=input_text.split(",")
    for url in input_text:
        if url.startswith("http"):
            # If the input starts with "http", assume it's a URL and extract content
            article_content = scrape_news_content(url)
        else:
            # If the input is not a URL, assume it's the content
            article_content = url

        # Perform sentiment classification
        inputs_cls = tokenizer_cls(article_content, return_tensors="pt", max_length=512, truncation=True, padding=True)
        inputs_cls = {key: value.to(device) for key, value in inputs_cls.items()}

        # Move cls_model to the specified device
        cls_model = cls_model.to(device)

        outputs_cls = cls_model(**inputs_cls)
        logits_cls = outputs_cls.logits
        predicted_class = torch.argmax(logits_cls, dim=1).item()
        classification = label_encoder.inverse_transform([predicted_class])[0]

        # Perform summarization based on the classification
        summary_risk, summary_opportunity = summarize_with_t5(article_content, classification, model_summ, tokenizer_summ, device)

        if summary_risk is None:
            summary_risk = "No risk summary available"
        if summary_opportunity is None:
            summary_opportunity = "No opportunity summary available"
        answer=article_content
        article_content_words = article_content.split()[:200]
        short_article_content = ' '.join(article_content_words)
        insert_question_and_answer(url,answer, request_timestamp)
        current_request_timestamp=request_timestamp
        results.append({"Question": url, "Article content":article_content,"Short Article content":short_article_content,"Classification": classification, "Summary risk": summary_risk, "Opportunity Summary": summary_opportunity})
        print("Result",results)
    return results

def generate_sentence_from_keywords(keywords):
    # Concatenate keywords into a single string
    keyword_sentence = ' '.join(keywords)

    # Tokenize the concatenated keywords into sentences
    sentences = sent_tokenize(keyword_sentence)

    # If there are sentences, return the first one; otherwise, return a default message
    return sentences[0] if sentences else "Unable to generate a sentence."

def is_question(input_text):
    questioning_words = ["who", "what", "when", "where", "why", "how"]
    return any(input_text.lower().startswith(q) for q in questioning_words)


def process_question(user_question,articlecontent):
    answers = [item[1] for item in articlecontent]
    context_string = ' '.join(map(str, answers))
    model_name = "deepset/tinyroberta-squad2"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {'question': user_question, 'context': context_string}
    print("Debug - QA_input:", QA_input)
    res = nlp(QA_input)
    print("Debug - res:", res)
    print(res['answer'])
    return res["answer"]

def generate_pdf(chat_history):
    # Create a PDF document using ReportLab
    buffer = io.BytesIO()

    # Adjust the page size and margins as needed
    pdf = SimpleDocTemplate(buffer, pagesize=letter)

    # List to store the content for the PDF
    pdf_content = []

    # Get sample styles for formatting
    styles = getSampleStyleSheet()

    # Maximum characters per line
    max_chars_per_line = 100

    # Write chat history to the PDF
    for message in chat_history:
        if isinstance(message, dict):
            for key, value in message.items():
                formatted_value = value[:max_chars_per_line] + ('...' if len(value) > max_chars_per_line else '')
                pdf_content.append(Paragraph(f"<strong>{key}:</strong> {formatted_value}", styles['Normal']))
        elif isinstance(message, str):
            formatted_message = message[:max_chars_per_line] + ('...' if len(message) > max_chars_per_line else '')
            pdf_content.append(Paragraph(formatted_message, styles['Normal']))
        else:
            formatted_message = str(message)[:max_chars_per_line] + ('...' if len(str(message)) > max_chars_per_line else '')
            pdf_content.append(Paragraph(formatted_message, styles['Normal']))
        pdf_content.append(Spacer(1, 10))  # Add space between messages

    # Build PDF document
    pdf.build(pdf_content)

    buffer.seek(0)
    return buffer.getvalue()

@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    # Generate a PDF document based on chat history
    pdf_buffer = generate_pdf(chat_history)
    
    # Provide the PDF as a download
    return send_file(
        io.BytesIO(pdf_buffer),
        as_attachment=True,
        download_name='chat_history.pdf',
        mimetype='application/pdf'
    )

current_request_timestamp = None

@app.route('/', methods=['GET', 'POST'])
def home():
    global current_request_timestamp
    classification = None
    summary_risk = None
    summary_opportunity = None
    article_content = None 
    input_submitted = False

    if request.method == 'POST':
        url_input = request.form['userInput']
        print("Form Data:", request.form)
        input_submitted = True
        print(url_input)
        cls_model = AutoModelForSequenceClassification.from_pretrained("riskclassification_finetuned_xlnet_model_ld")
        tokenizer_cls = AutoTokenizer.from_pretrained("xlnet-base-cased")
        label_encoder_path = "riskclassification_finetuned_xlnet_model_ld/encoder_labels.pkl"
        label_encoder = LabelEncoder()

        # Assuming 'label_column values' is the column you want to encode
        label_column_values = ["risks","opportunities","neither"]


        label_encoder.fit_transform(label_column_values)

        joblib.dump(label_encoder, label_encoder_path)


        model_summ = T5ForConditionalGeneration.from_pretrained("t5-small")
        tokenizer_summ = T5Tokenizer.from_pretrained("t5-small")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if url_input.startswith("http"):
            current_request_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # If the input starts with "http", assume it's a URL and extract content
            totalresult = classify_and_summarize(
                url_input, cls_model, tokenizer_cls, label_encoder, model_summ, tokenizer_summ, device
            )
            chat_history.extend(totalresult)
            '''first={"Classification":classification}
            second={"Summary risk":summary_risk}
            opp={"Opportunity Summary":summary_opportunity}
            third={"Article content":article_content}
            chat_history.extend([{"Question":url_input}])
            chat_history.extend([first])
            chat_history.extend([second])
            chat_history.extend([opp])
            chat_history.extend([third])
            chat_history.extend([{"Short Article content":short_article_content}]) ''' # Display only the first 200 words
            '''return render_template('index.html', classification=classification, summary_risk=summary_risk,
                               summary_opportunity=summary_opportunity, article_content=article_content,
                               input_submitted=input_submitted, chat_history=chat_history)'''
        elif is_question(url_input):
            # If the input starts with questioning words, process the question
            timestamp= datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if current_request_timestamp and current_request_timestamp is not None:
                articlecontent = retrieve_article_content(current_request_timestamp)
                #articlecontent=retrieve_article_content()
                answer = process_question(url_input,articlecontent) 
             # You need to implement process_question function
                insert_question_and_answer(url_input,answer,timestamp)
                uq={"User Question": url_input}
                chat_history.extend([uq])
                ma={"Model Answer": answer}
                chat_history.extend([ma])
           # return render_template('index.html', question=url_input,answer=answer,chat_history=chat_history)
    print("chat history",chat_history)
    return render_template('index.html', chat_history=chat_history,classification=classification, summary_risk=summary_risk, summary_opportunity=summary_opportunity, article_content=article_content, input_submitted=input_submitted)

if __name__ == '__main__':
    app.run(debug=True)
