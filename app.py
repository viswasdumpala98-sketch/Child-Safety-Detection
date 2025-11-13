import pickle
from flask import Flask , request, render_template
app = Flask(__name__)
model = pickle.load(open("XGBoo_70.pkl","rb"))

@app.route('/')
def indput():
    return render_template('index.html')

@app.route('/index', methods=['GET','POST'])
def input():
    return render_template('inner-page.html')

@app.route('/portfolio', methods=['GET','POST'])
def portfolio():
    return render_template('portfolio-details.html') 

@app.route('/predict',methods = ['GET','POST'])
def admin():
    Warehouse_block=(request.form["Warehouse_block"])
    Mode_of_Shipment=(request.form["Mode_of_Shipment"])
    Customer_care_calls=int(request.form["Customer_care_calls"])
    Customer_rating=int(request.form["Customer_rating"])
    Cost_of_the_Product = int(request.form["Cost_of_the_Product"])
    Prior_purchases = int(request.form["Prior_purchases"])
    Product_importance = (request.form["Product_importance"])
    Gender = (request.form["Gender"])
    Discount_offered = int(request.form["Discount_offered"])
    Weight_in_gms = int(request.form["Weight_in_gms"])
    
    # Convert categorical features to numeric codes if necessary
    Warehouse_block_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    Mode_of_Shipment_mapping = {"Flight": 0, "Ship": 1, "Road": 2}
    Product_importance_mapping = {"low": 0, "medium": 1, "high": 2}
    Gender_mapping = {"Male": 0, "Female": 1}

    Warehouse_block = Warehouse_block_mapping[Warehouse_block]
    Mode_of_Shipment = Mode_of_Shipment_mapping[Mode_of_Shipment]
    Product_importance = Product_importance_mapping[Product_importance]
    Gender = Gender_mapping[Gender]
    
    preds=[[Warehouse_block,Mode_of_Shipment,Customer_care_calls,Customer_rating,Cost_of_the_Product,
           Prior_purchases,Product_importance,Gender,Discount_offered,Weight_in_gms]]
    xx=model.predict(preds)
    prob=model.predict_proba(preds)[0]
    n_reach = prob[0]
    reach = prob[1]
    prediction_text = 'There is a {0:.2f}% chance that your product will reach in time'.format(reach * 100)
    print(prediction_text)
    print(xx)
    return render_template("result.html", prediction_text=prediction_text)
    # print('There is a {0:.2f} % chance that your product will reach in time'.format(reach*100))
    # print(xx)
    # return render_template("index3.html",Prediction_result='There is a {0:.2f} chance that your product will reach in time'.format(reach*100))
if __name__ == '__main__':
    app.run(debug='True')
    app.run(port=4000)
