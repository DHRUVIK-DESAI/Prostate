import pickle
import bz2

radius=float(input("Enter radius="))
texture=float(input("Enter texture="))
perimeter=float(input("Enter perimeter="))
area=float(input("Enter area="))
smoothness=float(input("Enter smoothness="))
compactness=float(input("Enter compactness="))
symmetry=float(input("Enter symmetry="))
fractal_dimension=float(input("Enter fractal_dimension="))

sfile = bz2.BZ2File('model.pkl', 'rb')
model=pickle.load(sfile)

names = ["K-Nearest Neighbors", "SVM",
         "Decision Tree", "Random Forest",
         "Naive Bayes","ExtraTreesClassifier","VotingClassifier"]
for i in range(len(model)):
    print(names[i])
    test_prediction = model[i].predict([[radius,texture,perimeter,area,smoothness,compactness,symmetry,fractal_dimension]])
    print(test_prediction[0])