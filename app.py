from flask import *
import pandas as pd 
# import seaborn as sns
# from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression,LogisticRegression


app = Flask(__name__) 
#######IRIS
url1="https://raw.githubusercontent.com/sarwansingh/Python/master/ClassExamples/data/iris.csv"
iris_df=pd.read_csv(url1)
x=iris_df.iloc[:,:4]
y=iris_df.iloc[:,4]
model1=LogisticRegression()
model1.fit(x,y)
r1=model1.predict([[5.2,1.3,2.6,0.1]])
#######PASS_OR_FAIL
url="https://raw.githubusercontent.com/sarwansingh/Python/master/ClassExamples/data/student-pass-fail-data.csv"
df=pd.read_csv(url)
x=df.iloc[:,:2].values
y=df.iloc[:,2].values
model=LinearRegression()
model.fit(x,y)
#####Calorie

calories=pd.read_csv("calories.csv")
excercise=pd.read_csv("exercise.csv")
cl_data=pd.concat([excercise,calories['Calories']],axis=1)
cl_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)
# correlation=cl_data.corr()
# plt.figure(figsize=(10,10))
# heat_map=sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

x=cl_data.drop(["User_ID","Calories"],axis=1)
y=cl_data.Calories

model=LinearRegression()
model.fit(x,y)


@app.route('/calorie',methods=["Post"])
def caloriefile():
  weight=int(request.form["weight"])
  height=int(request.form["height"])
  age=int(request.form["age"])
  gender=int(request.form["gender"])
  Duration=int(request.form["Duration"])
  hr=int(request.form["hr"])
  bd_temp=int(request.form["bd_temp"])
  o=[[gender,age,height,weight,Duration,hr,bd_temp]]
  re=model.predict(o)
  op=" Calories Burnt:  " + str(round(re[0],2))+"cals"
  return render_template("calorie.html",re=op)
                       
  
@app.route("/calorie")
def calorie():
  return render_template('calorie.html')

@app.route('/form')
def form():
  return render_template('form.html')


@app.route('/') 
def hello_world(): 
  return render_template('project.html')
  # return 'Hello, AIML Champ for Python web development! '  #+op
  
@app.route('/predict',methods=["POST"]) 
def predict(): 
  dhrs=int(request.form["dhrs"])
  mhrs=int(request.form["mhrs"])
  add1="Daily Study:"+str(dhrs-1)+"hrs"+" Monthly tutorials:"+str(mhrs-1)+"hrs"
  
  res=model.predict([[dhrs,mhrs]])
  op=" Result Prediction :"+str(round(res[0]*100,2))+'%'
  return render_template('form.html',result=add1,r=op)


@app.route('/iris')
def iris():
  return render_template('iris.html',result=r1)
@app.route('/irisfile',methods=["POST"]) 
def irisfile(): 
  sl=float(request.form["sl"])
  sw=float(request.form["sw"])
  pl=float(request.form["pl"])
  pw=float(request.form["pw"])
  
  ans1=" Sepal Length: "+str(sl)+" Sepal width"+str(sw)
  ans2=" Petal Length: "+str(pl)+" Petal width"+str(sw)     
  res1=model1.predict([[sl,sw,pl,pw]])
  ans="Result Prediction"+res1[0]                
  return render_template('iris.html',ans=ans1+ans2,sol=ans)
if __name__=='__main__':
  app.run(debug=True)
  
  
  