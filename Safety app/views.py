from django.shortcuts import render
import os, io, base64, pickle, smtplib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymysql
import pandas as pd
import numpy as np
import seaborn as sns
from email.mime.text import MIMEText

# ==============================
# GLOBAL VARIABLES
# ==============================
global username, email
username = ""
email = ""

# ==============================
# PATH SETTINGS
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "CrimesOnWomenData.csv")
MODEL_RESULTS_PATH = os.path.join(BASE_DIR, "model", "model_results.csv")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoders.pkl")
MODEL_META_PATH = os.path.join(BASE_DIR, "model", "best_model_info.pkl")  
GRAPH_DIR = os.path.join(BASE_DIR, "model", "confusion_matrices")

BEST_CM_FALLBACK_1 = os.path.join(GRAPH_DIR, "best_model_cm.png")
BEST_CM_FALLBACK_2 = os.path.join(GRAPH_DIR, "LogisticRegression_cm.png")

# ==============================
# LOAD DATASET SAFELY
# ==============================
dataset = pd.DataFrame()
states = []
state_col = None
year_col = None

try:
    if os.path.exists(DATASET_PATH):
        dataset = pd.read_csv(DATASET_PATH)
        dataset.columns = [str(c).strip() for c in dataset.columns]

        if 'Unnamed: 0' in dataset.columns:
            dataset.drop(columns=['Unnamed: 0'], inplace=True)

        state_col = next((c for c in dataset.columns if str(c).strip().lower() in ['state', 'area_name']), None)
        year_col = next((c for c in dataset.columns if str(c).strip().lower() == 'year'), None)

        if state_col:
            dataset[state_col] = dataset[state_col].astype(str).str.strip()
            states = sorted(np.unique(dataset[state_col]))
        else:
            states = []

        # Ensure Total_Crime column exists
        if state_col and year_col:
            crime_cols = [c for c in dataset.columns if c not in [state_col, year_col]]
            for col in crime_cols:
                dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
            dataset['Total_Crime'] = dataset[crime_cols].sum(axis=1)
except Exception:
    dataset = pd.DataFrame()
    states = []

# ==============================
# DATABASE CONNECTION
# ==============================
def get_db_connection():
    return pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='******',
        database='safety',
        charset='utf8'
    )

# ==============================
# HELPER FUNCTIONS
# ==============================
def get_best_model_name_from_results():
    try:
        if not os.path.exists(MODEL_RESULTS_PATH):
            return None
        df = pd.read_csv(MODEL_RESULTS_PATH)
        df.columns = [str(c).strip() for c in df.columns]
        if 'Model' not in df.columns or 'Accuracy' not in df.columns or df.empty:
            return None
        df = df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
        return str(df.loc[0, 'Model']).strip()
    except Exception:
        return None

def get_best_confusion_matrix_path():
    try:
        if os.path.exists(MODEL_META_PATH):
            meta = pickle.load(open(MODEL_META_PATH, 'rb'))
            if isinstance(meta, dict):
                best_name = str(meta.get('best_model_name', '')).strip()
                if best_name:
                    best_cm = os.path.join(GRAPH_DIR, f"{best_name}_cm.png")
                    if os.path.exists(best_cm):
                        return best_cm
        best_name = get_best_model_name_from_results()
        if best_name:
            best_cm = os.path.join(GRAPH_DIR, f"{best_name}_cm.png")
            if os.path.exists(best_cm):
                return best_cm
        if os.path.exists(BEST_CM_FALLBACK_1):
            return BEST_CM_FALLBACK_1
        if os.path.exists(BEST_CM_FALLBACK_2):
            return BEST_CM_FALLBACK_2
    except Exception:
        pass
    return None

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return img

# ==============================
# BASIC PAGES
# ==============================
def index(request):
    return render(request, 'index.html')

# ==============================
# LOGIN
# ==============================
def UserLogin(request):
    return render(request, 'UserLogin.html')

def UserLoginAction(request):
    global username, email
    if request.method == 'POST':
        try:
            user = request.POST.get('t1', '').strip()
            password = request.POST.get('t2', '').strip()
            con = get_db_connection()
            with con:
                cur = con.cursor()
                cur.execute("SELECT username,email FROM register WHERE username=%s AND password=%s",(user, password))
                row = cur.fetchone()
                if row:
                    username, email = row[0], row[1]
                    return render(request, "UserScreen.html", {'data': f"Welcome {username}"})
                else:
                    return render(request, 'UserLogin.html', {'data': "Invalid login"})
        except Exception as e:
            return render(request, 'UserLogin.html', {'data': f"Login failed: {str(e)}"})
    return render(request, 'UserLogin.html')

# ==============================
# REGISTER
# ==============================
def Register(request):
    return render(request, 'Register.html')

def RegisterAction(request):
    if request.method == 'POST':
        try:
            user = request.POST.get('t1', '').strip()
            password = request.POST.get('t2', '').strip()
            contact = request.POST.get('t3', '').strip()
            email_id = request.POST.get('t4', '').strip()
            address = request.POST.get('t5', '').strip()
            parent_email = request.POST.get('t6', '').strip()
            con = get_db_connection()
            with con:
                cur = con.cursor()
                cur.execute("SELECT username FROM register WHERE username=%s", (user,))
                if cur.fetchone():
                    return render(request, 'Register.html', {'data': "Username already exists"})
                cur.execute("INSERT INTO register (username,password,contact,email,address,parent_email) VALUES (%s,%s,%s,%s,%s,%s)",
                            (user, password, contact, email_id, address, parent_email))
                con.commit()
            return render(request, 'Register.html', {'data': "Registered Successfully"})
        except Exception as e:
            return render(request, 'Register.html', {'data': f"Registration failed: {str(e)}"})
    return render(request, 'Register.html')

# ==============================
# MODEL EVALUATION
# ==============================
def TrainML(request):
    try:
        if not os.path.exists(MODEL_RESULTS_PATH):
            return render(request, 'UserScreen.html', {'data': "Run train_model.py first<br>Error: model_results.csv not found"})
        df = pd.read_csv(MODEL_RESULTS_PATH)
        df.columns = [str(c).strip() for c in df.columns]
        required_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score']
        if not all(col in df.columns for col in required_cols):
            return render(request, 'UserScreen.html', {'data': "Run updated train_model.py first<br>Error: Missing columns in model_results.csv"})
        df = df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
        display_df = df.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1_Score']:
            display_df[col] = (display_df[col].astype(float)*100).round(2).astype(str)+'%'
        plt.figure(figsize=(8,5))
        plt.bar(df['Model'], df['Accuracy'], color='skyblue')
        plt.xticks(rotation=20)
        plt.title("Model Comparison (Accuracy)")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        img = fig_to_base64()
        cm_img = None
        best_cm_path = get_best_confusion_matrix_path()
        if best_cm_path and os.path.exists(best_cm_path):
            with open(best_cm_path, 'rb') as f:
                cm_img = base64.b64encode(f.read()).decode()
        html_table = display_df.to_html(index=False, escape=False)
        best_model_name = None
        try:
            if not df.empty:
                best_model_name = str(df.loc[0, 'Model']).strip()
        except Exception:
            best_model_name = None
        if cm_img:
            extra_html = "<br><br><h3>Best Model Confusion Matrix"
            if best_model_name:
                extra_html += f" ({best_model_name})"
            extra_html += "</h3>"
            extra_html += f'<img src="data:image/png;base64,{cm_img}" width="600"/>'
            html_table += extra_html
        return render(request, 'UserScreen.html', {'data': html_table, 'img': img})
    except Exception as e:
        return render(request, 'UserScreen.html', {'data': f"Run train_model.py first<br>Error: {str(e)}"})

# ==============================
# CRIME PREDICTION
# ==============================
def CrimePredict(request):
    return render(request, 'CrimePredict.html', {'states': states})

def CrimePredictAction(request):
    if request.method == 'POST':
        try:
            state = request.POST.get('state', '').strip()
            year = request.POST.get('year', '').strip()
            if state == '' or year == '':
                return render(request, 'CrimePredict.html', {'states': states, 'data': "Please select state and enter year", 'error': True})
            try:
                year = int(year)
            except ValueError:
                return render(request, 'CrimePredict.html', {'states': states, 'data': "Year must be a valid number", 'error': True})
            if not os.path.exists(BEST_MODEL_PATH) or not os.path.exists(ENCODER_PATH):
                return render(request, 'CrimePredict.html', {'states': states, 'data': "Run train_model.py first (best_model.pkl or encoders missing)", 'error': True})
            model = pickle.load(open(BEST_MODEL_PATH, 'rb'))
            encoders = pickle.load(open(ENCODER_PATH, 'rb'))
            state_encoder = encoders.get('state_encoder') if isinstance(encoders, dict) else encoders
            risk_encoder = encoders.get('risk_encoder') if isinstance(encoders, dict) else None
            if state_encoder is None:
                return render(request, 'CrimePredict.html', {'states': states, 'data': "State encoder missing in label_encoders.pkl", 'error': True})
            if state not in state_encoder.classes_:
                return render(request, 'CrimePredict.html', {'states': states, 'selected_state': state, 'year': year, 'data': "State not found in trained model", 'error': True})
            state_encoded = state_encoder.transform([state])[0]
            # Derived features
            crime_last_year = 0
            crime_3year_avg = 0
            crime_growth_rate = 0
            if not dataset.empty and state_col and 'Total_Crime' in dataset.columns:
                state_df = dataset[dataset[state_col]==state].sort_values(year_col)
                prev_years = state_df[state_df[year_col]<year]
                if not prev_years.empty:
                    crime_last_year = prev_years['Total_Crime'].iloc[-1]
                    crime_3year_avg = prev_years['Total_Crime'].tail(3).mean()
                    if len(prev_years)>1:
                        second_last = prev_years['Total_Crime'].iloc[-2]
                        crime_growth_rate = (crime_last_year - second_last)/max(second_last,1)
            X_input = np.array([[state_encoded, year, crime_last_year, crime_3year_avg, crime_growth_rate]])
            pred_encoded = model.predict(X_input)[0]
            if risk_encoder:
                risk = risk_encoder.inverse_transform([int(pred_encoded)])[0]
            else:
                risk = str(pred_encoded)
            risk_map = {'0':'Low','1':'Medium','2':'High'}
            risk = risk_map.get(str(risk).strip().lower(), risk)
            categories = ['Low','Medium','High']
            values = [1 if c==risk else 0 for c in categories]
            plt.figure(figsize=(5,4))
            plt.bar(categories, values, color=['green','orange','red'])
            plt.title("Predicted Risk Level")
            plt.ylabel("Risk Indicator")
            plt.tight_layout()
            img = fig_to_base64()
            output = f"<b>State:</b> {state}<br><b>Year:</b> {year}<br><b>Predicted Risk Level:</b> {risk}"
            return render(request, 'CrimePredict.html', {'states': states, 'selected_state': state, 'year': year, 'data': output, 'img': img})
        except Exception as e:
            return render(request, 'CrimePredict.html', {'states': states, 'data': f"Prediction failed: {str(e)}", 'error': True})
    return render(request, 'CrimePredict.html', {'states': states})

# ==============================
# ROUTE
# ==============================
def Route(request):
    return render(request, 'Route.html')

def RouteAction(request):
    try:
        area = request.POST.get('t1','').strip()
        if area == '':
            return render(request,'UserScreen.html',{'data':"Please enter area name"})
        area = area + "+Police+Station"
        map_html = f'<iframe width="800" height="600" src="https://maps.google.com/maps?q={area}&output=embed"></iframe>'
        return render(request,'UserScreen.html',{'data':map_html})
    except Exception as e:
        return render(request,'UserScreen.html',{'data':f"Route error: {str(e)}"})

# ==============================
# PANIC ALERT
# ==============================
def Panic(request):
    return render(request,'Panic.html')

def PanicAction(request):
    global username,email
    if request.method=='POST':
        location=request.POST.get('location','').strip() or "Location could not be detected"
        try:
            con = get_db_connection()
            parent_email=""
            with con:
                cur = con.cursor()
                cur.execute("SELECT parent_email FROM register WHERE username=%s",(username,))
                row = cur.fetchone()
                if row and row[0]:
                    parent_email=row[0]
            recipients=[e for e in [email,parent_email] if e]
            if not recipients:
                return render(request,'Panic.html',{'data':"No recipient email found. Please login/register correctly."})
            subject = "Emergency Panic Alert"
            message = f"""EMERGENCY ALERT

User: {username}

The user needs immediate help.

Live Location:
{location}
"""
            email_address='emailaddress@gmail.com'
            email_password='***********'
            msg = MIMEText(message,'plain','utf-8')
            msg['Subject']=subject
            msg['From']=email_address
            msg['To']=", ".join(recipients)
            with smtplib.SMTP_SSL('smtp.gmail.com',465) as connection:
                connection.login(email_address,email_password)
                connection.sendmail(email_address,recipients,msg.as_string())
            message="Emergency alert with live location sent successfully"
        except Exception as e:
            message=f"Error sending email: {str(e)}"
        return render(request,'Panic.html',{'data':message})
    return render(request,'Panic.html')

# ==============================
# HEATMAP
# ==============================
def Heatmap(request):
    try:
        if dataset.empty:
            return render(request,'UserScreen.html',{'data':'<font color="red">Dataset not found or empty</font>'})
        if not state_col or not year_col:
            return render(request,'UserScreen.html',{'data':'<font color="red">State or Year column not found</font>'})
        crime_cols = [c for c in dataset.columns if c not in [state_col, year_col,'Total_Crime']]
        temp = dataset.copy()
        for col in crime_cols:
            temp[col] = pd.to_numeric(temp[col], errors='coerce')
        temp.dropna(inplace=True)
        temp['Total_Crime'] = temp[crime_cols].sum(axis=1)
        heatmap_df = temp.groupby(state_col)['Total_Crime'].sum().sort_values(ascending=False).head(10)
        plt.figure(figsize=(10,6))
        sns.heatmap(np.array(heatmap_df).reshape(-1,1), annot=True, fmt=".0f", cmap="Reds", yticklabels=heatmap_df.index, xticklabels=['Total Crime Score'])
        plt.title("Top 10 States by Women Crime Composite Score")
        plt.tight_layout()
        img_b64 = fig_to_base64()
        return render(request,'UserScreen.html',{'data':'Women Safety Heatmap based on Composite Crime Score','img':img_b64})
    except Exception as e:
        return render(request,'UserScreen.html',{'data':f'<font color="red">Heatmap generation failed: {str(e)}</font>'})
