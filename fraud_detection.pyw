import random
import numpy as np
import pandas as pd
from tkinter import *
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from bidi.algorithm import get_display
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import IsolationForest
from tkinter import messagebox, ttk
import matplotlib.pyplot as pltree
import customtkinter as ctk
import arabic_reshaper
from PIL import Image
import seaborn as sns
import sqlite3

conn=sqlite3.connect("bank_fraud_detection_db.db")
cursor=conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS fraud_detection( 
id INTEGER PRIMARY KEY AUTOINCREMENT, 
transaction_amount INTEGER,
transaction_hour INTEGER,
transaction_weekday INTEGER,
is_weekend INTEGER,
account_age_days INTEGER,
customer_tenure_months INTEGER,
num_transactions_24h INTEGER,
num_failed_logins_24h INTEGER,
avg_transaction_amount_7d FLOAT,
is_international INTEGER,
is_new_device INTEGER,
distance_from_home FLOAT,
card_present  INTEGER,
previous_fraud_flag  INTEGER,
total_balance  FLOAT,
is_fraud INTEGER
);''')
conn.commit()

# query = f'''DELETE FROM fraud_detection WHERE id=1 '''
# cursor.execute(query)
# conn.commit()

def random_data(n=1000):
    data={
        'transaction_amount':[random.randint(0, 1000) for _ in range(n)] ,
        'transaction_hour': [random.randint(0, 24)for _ in range(n)],
        'transaction_weekday': [random.randint(0, 7)for _ in range(n)],
        'is_weekend': [random.randint(0, 2)for _ in range(n)],
        'account_age_days': [random.randint(30, 2000)for _ in range(n)],
        'customer_tenure_months': [random.randint(0, 500)for _ in range(n)],
        'num_transactions_24h': [random.randint(0, 10)for _ in range(n)],
        'num_failed_logins_24h': [random.randint(0, 10)for _ in range(n)],
        'avg_transaction_amount_7d': [random.uniform(0,50)for _ in range(n)],
        'is_international': [random.choice([0])for _ in range(n)],
        'is_new_device': [random.choice([0])for _ in range(n)],
        'distance_from_home': [random.uniform(0,1000)for _ in range(n)],
        'card_present': [random.choice([1])for _ in range(n)],
        'previous_fraud_flag': [random.randint(0, 2)for _ in range(n)],
        'total_balance': [random.uniform(0, 10000)for _ in range(n)],}
    data2={
        'transaction_amount': [random.randint(1000,50000)for _ in range(n)],
        'transaction_hour': [random.randint(0, 24)for _ in range(n)],
        'transaction_weekday': [random.randint(0, 7)for _ in range(n)],
        'is_weekend': [random.randint(0, 2)for _ in range(n)],
        'account_age_days': [random.randint(30, 2000)for _ in range(n)],
        'customer_tenure_months': [random.randint(0, 500)for _ in range(n)],
        'num_transactions_24h': [random.randint(10, 30)for _ in range(n)],
        'num_failed_logins_24h': [random.randint(10, 50)for _ in range(n)],
        'avg_transaction_amount_7d': [random.uniform(50,100)for _ in range(n)],
        'is_international': [random.choice([1])for _ in range(n)],
        'is_new_device':[random.choice([1])for _ in range(n)],
        'distance_from_home': [random.uniform(1000, 10000)for _ in range(n)],
        'card_present': [random.choice([0])for _ in range(n)],
        'previous_fraud_flag': [random.randint(0, 2)for _ in range(n)],
        'total_balance': [random.uniform(10000, 100000)for _ in range(n)],}
 
    for j in data.keys():
        data[j].extend(data2[j])
    indices=list(range(len(data['transaction_amount'])))
    random.shuffle(indices)
    for j in data.keys():
        data[j] = [data[j][i] for i in indices]
    is_fraud=[]
    # توليد عمود churn بناءً على شروط منطقية
    for i in range(2000):
        if data['transaction_amount'][i] > 200 and data['num_transactions_24h'][i] >5 and data['num_failed_logins_24h'][i] >3 and data['avg_transaction_amount_7d'][i]>50 and data['is_international'][i] ==1 and data['is_new_device'][i] ==1 and data['distance_from_home'][i]>1000 and data['card_present'][i] ==0 and data['previous_fraud_flag'][i]==1 and data['total_balance'][i] > 1000 :
            is_fraud.append(-1)
        else:
            is_fraud.append(1)
    data['is_fraud']=is_fraud
    return pd.DataFrame(data)
df=random_data()
# معالجة البيانات
df.isnull()
print(f"\n data null :\n \n{df.isnull().sum()}")
df.duplicated()
print(f"\n data duplicated: \n {df.duplicated().sum()}")
df.to_csv("bank_fraud_detection.csv",index=False)
df=pd.read_csv("bank_fraud_detection.csv")

features = [col for col in df.columns if col != 'is_fraud']

X = df[features]
y = df['is_fraud']

# scaler=StandardScaler()
# X_scaled=scaler.fit_transform(X)
X_temp,X_test,y_temp,y_test = train_test_split(X, y, test_size=0.2 , random_state=42)
X_train,X_vald, y_train,y_vald = train_test_split(X_temp, y_temp, test_size=0.20 , random_state=42)


model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)



def allplots(tv,ytv):
    y_pred = model.predict(tv)
    cm = confusion_matrix(ytv, y_pred)
    accuracy = accuracy_score(ytv, y_pred)
    
    models = ['IsolationForest']
    asfig, ax = plt.subplots(figsize=(5, 4))
    plt.bar(models, [accuracy], color='#106ccd')
    plt.title(get_display(arabic_reshaper.reshape("دقة النموذج")), fontsize=12,fontweight="bold")  
    plt.ylabel(get_display(arabic_reshaper.reshape("الدقة")), fontsize=12,fontweight="bold")
    plt.ylim(0.0, 1.0)
    plt.text(0, accuracy + 0.02, f"{accuracy:.2%}", ha='center',fontsize=12,fontweight="bold")
    plt.grid(axis="y",linestyle="--",alpha=0.5,linewidth=0.50,color="white")
    plt.tight_layout()
    plt.close()
 

    
    cmfig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[get_display(arabic_reshaper.reshape("is_Fraud")),get_display(arabic_reshaper.reshape("NOT Fraud"))], yticklabels=[get_display(arabic_reshaper.reshape("is_Fraud")),get_display(arabic_reshaper.reshape("NOT Fraud"))])
    plt.xlabel(get_display(arabic_reshaper.reshape('القيمة المتوقعة (Predicted)')))
    plt.ylabel(get_display(arabic_reshaper.reshape('القيمة الحقيقية (True)')))
    plt.title(get_display(arabic_reshaper.reshape('مصفوفة الارتباك - Confusion Matrix')))
    plt.tight_layout()
    plt.close()

    # الحصول على التقرير بشكل ديكشنري
    report = classification_report(ytv, y_pred, output_dict=True)
    # تحويله إلى DataFrame لتسهيل الرسم
    report_df = pd.DataFrame(report).transpose()
    # رسم البيانات
    crvfig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(report_df, annot=True, cmap="Blues", fmt=".2f")
    plt.title(get_display(arabic_reshaper.reshape("Classification Report Visualization")) )
    plt.tight_layout()
    plt.close()
    # 3. Scatter Plot للمقارنة بين عمودين (مثال: V2 و Amount)
    scatterfig, ax = plt.subplots(figsize=(5, 4))
    sns.scatterplot(data=tv, x=tv['distance_from_home'], y=tv['total_balance'], hue=y_pred, alpha=0.6, palette='coolwarm')
    plt.title(get_display(arabic_reshaper.reshape("رسم بياني مبعثر (Scatter Plot) بين  distance_from_home و total_balance")) )
    plt.xlabel('distance_from_home')
    plt.ylabel('total_balance')
    plt.legend(title=get_display(arabic_reshaper.reshape('احتيال')))
    plt.tight_layout()
    plt.close()

    # 1. Count Plot
    counterfig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=tv, x=y_pred)
    plt.title(get_display(arabic_reshaper.reshape('عدد الحالات المصنفة كاحتيال أو غير احتيال (بواسطة النموذج)')) )
    plt.xticks([0, 1], [get_display(arabic_reshaper.reshape('طبيعي')) ,get_display(arabic_reshaper.reshape('احتيال')) ])
    plt.tight_layout()
    plt.close()
    # 2. Histogram لعمود المبلغ حسب نوع الحالة
    hostfig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(data=tv, x=tv['total_balance'], hue=y_pred, bins=50, kde=True, palette='coolwarm')
    plt.title(get_display(arabic_reshaper.reshape('توزيع المبالغ المالية حسب الاحتيال')) )
    plt.xlabel(get_display(arabic_reshaper.reshape('total_balance')))
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.close()
    return asfig,crvfig,cmfig,scatterfig,hostfig,counterfig


    
WS,HS=750,550
#واجهة التطبيق
app=ctk.CTk()
app.title("تنبؤ بمغادرة العميل")
app.geometry(f"{WS}x{HS}+{(app.winfo_screenwidth()-WS)//2}+{(app.winfo_screenheight()-HS)//8}")
# ctk.set_appearance_mode("dark")
levellist={}
btnlist=[]  
btnlist2=[]
btndown_frame_list=[]
navlist=["Home","Details","Accurcy Matrex","Classification Report","Confusion Matrex","counter plot","scatter plot","Histogram"]
entries = {}
nav_btn_down=[]
def all_level():
    levellist["top"]=Frame(app)
    levellist["side"]=Frame(app,bg="#c8d9f3",pady=1)
    levellist["midel"]=Frame(app)
    levellist["end"]=Frame(app)
    for idx,level in enumerate(levellist.keys()):
        levellist["top"].place(relwidth=1.0,relheight=0.10)
        levellist["side"].place(rely=0.10,relwidth=0.25,relheight=0.87)
        levellist["midel"].place(relx=0.25,rely=0.18,relwidth=0.75,relheight=0.79)
        levellist["end"].place(rely=0.97,relwidth=1.0,relheight=0.03)
    title=Label(levellist["top"], text="اكتشاف الاحتيال البنكي",bg="#072b74",fg="white",font=("bold",16))
    title.pack(fill="both",expand=True)  
    title=Label(levellist["end"], text="",bg="#072b74",fg="white",font=("bold",16))
    title.pack(fill="both",expand=True) 

def btn_sid():
    if  btnlist:
        btnlist.clear()
    if  btndown_frame_list:
        btndown_frame_list.clear()
    for idx,navtext in enumerate(navlist):
        btndown_frame=Frame(levellist["side"])
        btndown_frame.place(rely=(idx*0.09),relwidth=1.0,relheight=0.09)
        btndown_frame_list.append(btndown_frame)
        
        for i in range(1):
            btnNav=ctk.CTkButton(btndown_frame, text=navtext,font=("bold",14),corner_radius=5, command=lambda indx=idx ,text=navtext:showMidel(indx,text))
            btnNav.place(relwidth=1.0,relheight=1.0) 
            btnlist.append(btnNav)
        
               
all_level()
btn_sid()
def showfigure(indx,i,text):
    # مسح محتوى الإطار
    levellist["midel"]=Frame(app)
    levellist["midel"].place(relx=0.25,rely=0.18,relwidth=0.75,relheight=0.79)
    for idx,btn in enumerate(nav_btn_down):
        btn.configure(fg_color='gray')
    nav_btn_down[i].configure(fg_color="#054fc6")
    TitleModel(text)
    if indx == 2:
       if text == "test":
           accurcyMatrex(X_test,y_test)
       
       else:
           accurcyMatrex(X_vald,y_vald)
    
    if indx == 3:
       if text == "test":
          
          classificationReport(X_test,y_test)
       else:
         
          classificationReport(X_vald,y_vald)
    if indx == 4:
       if text == "test":
          
          confusionMatrex(X_test,y_test)
       else:
          confusionMatrex(X_vald,y_vald)
    if indx == 5:
       if text == "test":
          counterplot(X_test,y_test)
       else:
          counterplot(X_vald,y_vald)
    if indx == 6:
       if text == "test":
          scatterplot(X_test,y_test)
       else:
          scatterplot(X_test,y_test)  
    if indx == 7:
       if text == "test":
          histogramplot(X_test,y_test)
       else:
          histogramplot(X_test,y_test)  
def drop_down_btn(btn):

    ybtnlist=[1.0/3,(1.0/3)*2]
    text_btn={"one":["test","Validition"]}
    
    if  btnlist2:
        btnlist2.clear()
    for indx,frame in enumerate(btndown_frame_list):
        frame.place(rely=indx*0.09,relheight=0.09)
        btnNav=ctk.CTkButton(frame, text=navlist[indx],font=("bold",14),corner_radius=5, command=lambda indx=indx ,text=navlist[indx]:showMidel(indx,text))
        btnNav.place(relwidth=1.0,relheight=1.0)
        btnlist2.append(btnNav)
    
    if btn == 2 or btn == 3 or btn == 4 or btn == 5 or btn == 6 or btn == 7:
        btndown_frame_list[btn].place(relheight=0.27)
        btnNav=ctk.CTkButton(btndown_frame_list[btn], text=navlist[btn],fg_color="#054fc6",font=("bold",14),corner_radius=5, command=lambda indx=btn ,text=navlist[btn]:showMidel(indx,text))
        btnNav.place(rely=0.0,relwidth=1.0,relheight=1.0/3)
        if btn == 2:
            btndown_frame_list[3].place(rely=0.45,relwidth=1.0,relheight=0.10)
            btndown_frame_list[4].place(rely=0.54,relwidth=1.0,relheight=0.10)
            btndown_frame_list[5].place(rely=0.63,relwidth=1.0,relheight=0.10)
            btndown_frame_list[6].place(rely=0.72,relwidth=1.0,relheight=0.10)
            btndown_frame_list[7].place(rely=0.81,relwidth=1.0,relheight=0.10)
        if btn ==3:
            btndown_frame_list[4].place(rely=0.54,relwidth=1.0,relheight=0.10)
            btndown_frame_list[5].place(rely=0.63,relwidth=1.0,relheight=0.10)
            btndown_frame_list[6].place(rely=0.72,relwidth=1.0,relheight=0.10)
            btndown_frame_list[7].place(rely=0.81,relwidth=1.0,relheight=0.10)
        if btn ==4:
            btndown_frame_list[5].place(rely=0.63,relwidth=1.0,relheight=0.10)
            btndown_frame_list[6].place(rely=0.72,relwidth=1.0,relheight=0.10)
            btndown_frame_list[7].place(rely=0.81,relwidth=1.0,relheight=0.10)
        if btn ==5:
           
            btndown_frame_list[6].place(rely=0.72,relwidth=1.0,relheight=0.10)
            btndown_frame_list[7].place(rely=0.81,relwidth=1.0,relheight=0.10)
        if btn == 6:
                btndown_frame_list[7].place(rely=0.81,relwidth=1.0,relheight=0.10)
        
        if  nav_btn_down:
            nav_btn_down.clear()
        for i in range(2):
            btnNav=ctk.CTkButton(btndown_frame_list[btn],fg_color="gray", text=text_btn["one"][i],font=("bold",14),corner_radius=5, command=lambda btn=btn,i=i, text=text_btn["one"][i]:showfigure(btn,i,text))
            btnNav.place(rely=ybtnlist[i],relwidth=1.0,relheight=1.0/3)
            nav_btn_down.append(btnNav)
    

# قائمة الحقول التي سيتم إدخالها
fields = [
    ("transaction_amount", get_display(arabic_reshaper.reshape("مبلغ العملية")) ),
    ("transaction_hour", get_display(arabic_reshaper.reshape("ساعة التنفيذ (0-23)")) ),
    ("transaction_weekday", get_display(arabic_reshaper.reshape("يوم الأسبوع (0-6)")) ),
    ("is_weekend", get_display(arabic_reshaper.reshape("عطلة نهاية الأسبوع (0=لا، 1=نعم)"))),
    ("account_age_days", get_display(arabic_reshaper.reshape("عمر الحساب بالأيام")) ),
    ("customer_tenure_months", get_display(arabic_reshaper.reshape("مدة العميل (بالأشهر)"))),
    ("num_transactions_24h", get_display(arabic_reshaper.reshape("عدد العمليات خلال 24 ساعة"))),
    ("num_failed_logins_24h", get_display(arabic_reshaper.reshape("عدد محاولات الدخول الفاشلة خلال 24 ساعة"))),
    ("avg_transaction_amount_7d", get_display(arabic_reshaper.reshape("متوسط العمليات خلال 7 أيام"))),
    ("is_international", get_display(arabic_reshaper.reshape("عملية دولية (0=لا، 1=نعم)"))),
    ("is_new_device", get_display(arabic_reshaper.reshape("جهاز جديد (0=لا، 1=نعم)"))),
    ("distance_from_home", get_display(arabic_reshaper.reshape("المسافة من الموقع المعتاد (كم)")) ),
    ("card_present", get_display(arabic_reshaper.reshape("البطاقة موجودة (0=لا، 1=نعم)"))),
    ("previous_fraud_flag", get_display(arabic_reshaper.reshape("سجل احتيال سابق (0=لا، 1=نعم)")) ),
    ("total_balance", get_display(arabic_reshaper.reshape("الرصيد الكلي")) )
]

 
def homeMidel():
    predictFrame=Frame(levellist["midel"],padx=10)
    predictFrame.place(relwidth=1.0,relheight=1.0)
    info_frame=Frame(predictFrame,pady=10)
    info_frame.pack(fill="both")
    ptn_preduct=ctk.CTkButton(info_frame, text="window preduction",font=("bold",12),corner_radius=5, command=predict_form)
    ptn_preduct.pack(fill="both")
   
    

def TitleModel(text):
    titleFrame=ctk.CTkFrame(app,fg_color="white")
    titleFrame.place(relx=0.25,rely=0.10,relwidth=0.75,relheight=0.08)
    titleMidel= ctk.CTkLabel(titleFrame, text=text,text_color="white",fg_color="#054fc6",font=("bold",15),corner_radius=5)
    titleMidel.pack(fill="both",padx=15,pady=5)

def back_home():
    all_level()
    btn_sid()
    
    TitleModel("Home")
    levellist["midel"]=Frame(app)
    levellist["midel"].place(relx=0.25,rely=0.18,relwidth=0.75,relheight=0.79)
    homeMidel()    

def predict_form():
    
    header=ctk.CTkFrame(app,fg_color="#dcdcdc",corner_radius=10)
    header.place(relwidth=1.0,relheight=0.10)
    back=ctk.CTkButton(app, text="back",fg_color="#072b74",text_color="white",font=("bold",16),corner_radius=10,command=back_home)
    back.place(relwidth=0.10,relheight=0.10) 
    title=Label(header, text="اكتشاف الاحتيال البنكي",bg="#072b74",fg="white",font=("bold",16),pady=10)
    title.pack(side="right",fill="both",expand=True)  
    top_level=ctk.CTkFrame(app ,fg_color="#dcdcdc",corner_radius=10)
    top_level.place(rely=0.10,relwidth=1.0,relheight=0.87)
    title= ctk.CTkLabel(top_level, text="predict",text_color="white",fg_color="#054fc6",font=("bold",17),corner_radius=5)
    title.pack(fill="both",padx=40,pady=7)
    predictFram=ctk.CTkFrame(top_level,fg_color="#dcdcdc",border_width=2,border_color="black",corner_radius=10)
    predictFram.pack(fill="both",expand=True,padx=15,pady=20)
    predictFrame=ctk.CTkFrame(predictFram,fg_color="#dcdcdc")
    predictFrame.pack(fill="both",expand=True,padx=15,pady=15)
    predictLeft=ctk.CTkFrame(predictFrame,fg_color="#dcdcdc",corner_radius=10)
    predictLeft.place(relx=0.0,rely=0.05,relwidth=0.25,relheight=0.7)
    predictRight=ctk.CTkFrame(predictFrame,fg_color="#dcdcdc",corner_radius=10)
    predictRight.place(relx=0.25,rely=0.05,relwidth=0.25,relheight=0.7)
    predictRight2=ctk.CTkFrame(predictFrame,fg_color="#dcdcdc",corner_radius=10)
    predictRight2.place(relx=0.50,rely=0.05,relwidth=0.25,relheight=0.7)
    predictRight3=ctk.CTkFrame(predictFrame,fg_color="#dcdcdc",corner_radius=10)
    predictRight3.place(relx=0.75,rely=0.05,relwidth=0.25,relheight=0.7)
    def create_input(idx,key,label_text):
        if idx >7:
            ctk.CTkLabel(predictRight, text=label_text,font=("bold",14),fg_color="white",text_color="black",corner_radius=6).pack(fill="both",padx=5,pady=2)
            entry = ctk.CTkEntry(predictLeft,justify="center",corner_radius=10,fg_color="white",text_color="black")
            entry.pack(fill="both",padx=5,pady=2)
            entries[key] = entry 
        else:
            ctk.CTkLabel(predictRight3, text=label_text,font=("bold",14),fg_color="white",text_color="black",corner_radius=6).pack(fill="both",padx=10,pady=2)
            entry = ctk.CTkEntry(predictRight2,justify="center",corner_radius=10,fg_color="white",text_color="black")
            entry.pack(fill="both",padx=10,pady=2)
            entries[key] = entry
    for idx,(key, field) in enumerate(fields):
        create_input(idx,key,field)
 
    def predict():
        try:
            input_data = [entries[key].get() for key,_ in fields]
            X_new = pd.DataFrame({
                'transaction_amount': [int(input_data[0])],
                'transaction_hour': [int(input_data[1])],
                'transaction_weekday':[input_data[2]] ,
                'is_weekend': [int(input_data[3])],
                'account_age_days': [int(input_data[4])],
                'customer_tenure_months':[int(input_data[5])],
                'num_transactions_24h': [int(input_data[6])],
                'num_failed_logins_24h':[int(input_data[7])],
                'avg_transaction_amount_7d':[float(input_data[8])],
                'is_international':[ int(input_data[9])],
                'is_new_device': [int(input_data[10])],
                'distance_from_home': [float(input_data[11])],
                'card_present': [int(input_data[12])],
                'previous_fraud_flag': int(input_data[13]),
                'total_balance':[float(input_data[14])]})
            pred = model.predict(X_new)
            if pred[0] == 1 :
                msg = "🟢 not fraud transaction"
                messagebox.showinfo("نتيجة التنبؤ", msg)
            else:
                msg ="🔴 is fraud transaction"
                data_insrt=[entries[key].get() for key,_ in fields]
                data_insrt.append(-1)
                placeholders = ', '.join(['?' for _ in data_insrt])
                query = f'''INSERT INTO fraud_detection (
                    transaction_amount,transaction_hour, transaction_weekday,
                    is_weekend,account_age_days,customer_tenure_months,
                    num_transactions_24h,num_failed_logins_24h,avg_transaction_amount_7d,
                    is_international,is_new_device,distance_from_home,
                    card_present,previous_fraud_flag,total_balance,is_fraud)
                    VALUES ({placeholders})'''
                cursor.execute(query, 
                [int(data_insrt[0]),int(data_insrt[1]),int(data_insrt[2]),int(data_insrt[3]),
                int(data_insrt[4]),int(data_insrt[5]),int(data_insrt[6]),int(data_insrt[7]),
                int(data_insrt[8]),int(data_insrt[9]),int(data_insrt[10]),float(data_insrt[11]),
                int(data_insrt[12]),int(data_insrt[13]),float(data_insrt[14]),int(data_insrt[15])])
                conn.commit()
                messagebox.showinfo("نتيجة التنبؤ", msg)
            
        except Exception as e:
            messagebox.showerror("خطأ", f"تأكد من إدخال البيانات بشكل صحيح\n{e}")
    ctk.CTkButton(predictFrame, text="تنبؤ",font=("bold",20),corner_radius=5, command=predict).place(relx=0.22,rely=0.87,relwidth=0.60,relheight=0.10) 
    
    drag_data = {"x": 0, "y": 0}
    def start_drag(event):
        drag_data["x"] = event.x
        drag_data["y"] = event.y
    def do_drag(event):
            dx = event.x-drag_data["x"]
            dy = event.y-drag_data["y"]
            x =top_level.winfo_x() + dx
            y = top_level.winfo_y() + dy
            top_level.place(relx=x/750,rely=y/463)
            
    top_level.bind("<ButtonPress-1>", start_drag)
    top_level.bind("<B1-Motion>",do_drag)
    
def detailsMidel():
    
    # Canvas لعمل التمرير
    
    canvas = Canvas(levellist["midel"])
    canvas.pack(side="left", fill="both",expand=True)
    # Scrollbar رأسي
    scrollbar = ttk.Scrollbar(levellist["midel"], orient="vertical", command=canvas.yview)
    scrollbar.place(relx=0.97,rely=0.0,relwidth=0.03,relheight=1.0)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar_x = ttk.Scrollbar(levellist["midel"], orient="horizontal", command=canvas.xview)
    scrollbar_x.place(relx=0.0,rely=0.97,relwidth=0.97)
    canvas.configure(xscrollcommand=scrollbar_x.set)
    # إطار داخلي داخل الـ Canvas
    inner_frame = Frame(canvas)

    canvas_window = canvas.create_window(0,0,window=inner_frame)

    # تحديث scrollregion عند تغيّر المحتوى
    def update_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    inner_frame.bind("<Configure>", update_scroll_region)
    rowCol=ctk.CTkFrame(inner_frame,border_color="blue",border_width=1,height=40)
    rowCol.pack(fill="both") 
    cursor.execute("PRAGMA table_info(fraud_detection)")
    columns = [col[1] for col in cursor.fetchall()]
    for idx,col in enumerate(columns):
        lbl=ctk.CTkLabel(rowCol, text=col,width=150,text_color="white",fg_color="#106ccd",font=("",10),corner_radius=6)
        lbl.pack(side="left",fill="both")
        if idx==16:
           lbl.configure(fg_color="#ec5409") 
    cursor.execute(f"SELECT * FROM fraud_detection")
    records = cursor.fetchall()

    for idxrec,fields in enumerate(records):
        rowVal=Frame(inner_frame,height=40)
        rowVal.pack(fill="both")
        for idx,field in enumerate(fields):
            fieldval=ctk.CTkLabel(rowVal,width=150, text=field,fg_color="#68bbff",font=("",10),corner_radius=6)
            fieldval.pack(side="left",fill="both",expand=True)
            if idx==16:
               fieldval.configure(fg_color="#ec5409")
def accurcyMatrex(tv,ytv):
    asfig,_,_,_,_,_=allplots(tv,ytv)
    canvas = FigureCanvasTkAgg(asfig, master=levellist["midel"])
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

def classificationReport(tv,ytv):
    _,crvfig,_,_,_,_= allplots(tv,ytv)
    canvas = FigureCanvasTkAgg(crvfig, master=levellist["midel"])
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

def confusionMatrex(tv,ytv):
    _,_,cmfig,_,_,_=allplots(tv,ytv)
    canvas = FigureCanvasTkAgg(cmfig, master=levellist["midel"])
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True) 
def scatterplot(tv,ytv):
   
    _,_,_,scatterfig,_,_=allplots(tv,ytv)
    canvas = FigureCanvasTkAgg(scatterfig, master=levellist["midel"])
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True) 
def counterplot(tv,ytv):
    _,_,_,_,_,counterfig=allplots(tv,ytv)
    canvas = FigureCanvasTkAgg(counterfig, master=levellist["midel"])
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True) 
def histogramplot(tv,ytv):
    _,_,_,_,hostfig,_=allplots(tv,ytv)
    canvas = FigureCanvasTkAgg(hostfig, master=levellist["midel"])
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True) 
           
def showMidel(indx,text):
    # مسح محتوى الإطار
    if indx == 0 or indx ==1 or indx ==5:
        levellist["midel"]=Frame(app)
        levellist["midel"].place(relx=0.25,rely=0.18,relwidth=0.75,relheight=0.79)
        TitleModel(text)
   
    if indx == 0:
        homeMidel()
        drop_down_btn(indx)
    if indx == 1:
        detailsMidel()
        drop_down_btn(indx)
    if indx == 2 :
        
        drop_down_btn(indx)
        
    if indx == 3 :
       
        drop_down_btn(indx)
    if indx == 4 :
        drop_down_btn(indx)
    if indx == 5 :
        drop_down_btn(indx)
    if indx == 6 :
        drop_down_btn(indx)
    if indx == 7 :
        drop_down_btn(indx)
   
    
    for idx,btn in enumerate(btnlist2):
        btn.configure(fg_color='#328bcb')
    btnlist2[indx].configure(fg_color="#054fc6")
       
showMidel(0,navlist[0])

app.mainloop()

