from flask import Flask,render_template,request
app = Flask(__name__)
import pickle

file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        bpain = int(myDict['bpain'])
        rnose = int(myDict['rnose'])
        bd = int(myDict['bd'])

        # Code for inference
        inputFeatures = [fever,bpain,age,rnose,bd]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        # return 'Hello, World' + str(infProb)
        return render_template('show.html',inf=round(infProb*100))
    return render_template('index.html')


@app.route("/Eyeball")
def about():
    return render_template('Eyeball.html')
if __name__ == '__main__':
    app.run(debug = True)

