# smart-insole-fudan-turku

## Code

- Pydoc documentation

## Data

- Data labeled as "Normal" and "Fall" based on collected sets

	- "Normal", "Risk" and "Fall" could be another option

## Classifier

The classifier and data processing library is currently at **"classifier/jupyter notebooks/insoleLib"**

Various python files in **"classifier/jupyter notebooks"** are jupyter notebooks in alternate format

### Notes

- Including steps with errors alters the accuracy of classifiers a bit

	- Seems to be a small change towards worse accuracy. Might be bigger problem when using data from multiple different persons.

### To-Do

- Test leave-three-out as CV?

	- Leave-one-out might give too optimistic results

- More feature selection/generation methods

- scaling forces could use more testing and observing

	- The data needs to be standardized since every session can have different forces. (different persons probably walk and weigh differently)

	- Needs further testing things

	- https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html

	- https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029

	- http://www.grroups.com/blog/about-feature-scaling-and-normalization-and-the-effect-of-standardization-for-machine-learning-algorithms

	- https://sebastianraschka.com/Articles/2014_about_feature_scaling.html

	- https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

- Testing more different feature sets. Which is the best?

- Balance classes in dataset(s) before training

	- Currently varied dataset has too many Normal steps compared to fall steps

	- genRandomDatasets with dropping some normal data

	- Up-sampling fall data? https://elitedatascience.com/imbalanced-classes

	- Or just use all/most normal data and rely mostly on AUC

- Building different classifiers

	- Deep learning things

		- https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

	- Multi-layer Perceptron?

		- https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html

	- Custom KNN which is extra sensitive for fall labels?

		- even one fall label in neighbor could cause labeling row as fall?

	- https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html

- Testing semi-supervised learning

- Classifier parameter tuning

	- Better accuracy

	- Avoiding overfitting and underfitting

	- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

	- https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65

	- https://scikit-learn.org/stable/modules/feature_selection.html

- Testing how much the amount of classifiers affect the results

![Sklearn Flowchart](https://scikit-learn.org/stable/_static/ml_map.png)

---

### Speeding up the classifying process (maybe)

- Could use multiprocessing maybe

- https://stackoverflow.com/questions/20548628/how-to-do-parallel-programming-in-python

- https://docs.python.org/release/3.6.8/library/multiprocessing.html


---

### Implemented

- Some features

- Classifiers

KNN average accuracy:  **0.88**

            	precision    recall  f1-score   support
        Fall		0.81      0.42      0.55        84
      Normal		0.89      0.98      0.93       389
   	micro avg       0.88      0.88      0.88       473
  	macro avg       0.85      0.70      0.74       473
	weighted avg    0.87      0.88      0.86       473


---

Gini decision tree average accuracy:  **0.84**

            	precision    recall  f1-score   support
        Fall       	0.55      0.46      0.50        84
      Normal       	0.89      0.92      0.90       389
   	micro avg       0.84      0.84      0.84       473
   	macro avg       0.72      0.69      0.70       473
	weighted avg    0.83      0.84      0.83       473


---

Entropy decision tree average accuracy:  **0.85**

              	precision    recall  f1-score   support
        Fall       	0.60      0.51      0.55        84
      Normal       	0.90      0.93      0.91       389
   	micro avg       0.85      0.85      0.85       473
   	macro avg       0.75      0.72      0.73       473
	weighted avg    0.84      0.85      0.85       473


---

Extreme gradient boosted tree average accuracy:  **0.89**

              	precision    recall  f1-score   support
        Fall       	0.78      0.56      0.65        84
      Normal       	0.91      0.97      0.94       389
   	micro avg       0.89      0.89      0.89       473
   	macro avg       0.85      0.76      0.80       473
	weighted avg    0.89      0.89      0.89       473


---	

Support Vector Machine average accuracy:  **0.86**

              	precision    recall  f1-score   support
        Fall       	0.70      0.39      0.50        84
      Normal       	0.88      0.96      0.92       389
   	micro avg       0.86      0.86      0.86       473
   	macro avg       0.79      0.68      0.71       473
	weighted avg    0.85      0.86      0.85       473


---

- Tests

	- Data labels suffled and test many times + plots + compared to real results

		- Conclusion from them: Those classifiers don't learn anything from the data (AUC always near 0.5). Meanwhile our real classifiers learn things from the data.

#### Ensemble learning

Bagging average accuracy:  **0.89**

              	precision    recall  f1-score   support
        Fall       	0.74      0.60      0.66        84
      Normal       	0.92      0.95      0.93       389
   	micro avg       0.89      0.89      0.89       473
   	macro avg       0.83      0.77      0.80       473
	weighted avg    0.88      0.89      0.89       473

AUC score:  0.77


Boosting accuracy:  **0.9**

              	precision    recall  f1-score   support
        Fall       	0.75      0.62      0.68        84
      Normal       	0.92      0.96      0.94       389
   	micro avg      	0.90      0.90      0.90       473
   	macro avg      	0.84      0.79      0.81       473
	weighted avg   	0.89      0.90      0.89       473

AUC score:  0.79


Fall skewed boosting accuracy:  **0.88**

              	precision    recall  f1-score   support
        Fall       	0.65      0.69      0.67        84
      Normal       	0.93      0.92      0.93       389
   micro avg       	0.88      0.88      0.88       473
   macro avg       	0.79      0.81      0.80       473
	weighted avg    0.88      0.88      0.88       473

AUC score:  0.81


---

## Webapp

### To-Do (see issues)

The web app should have the following features:

- login page [OK]
- registration page [OK]
- profile page [OK]

---

- upload session files.
- show in a table the data of a table file.
- Movesole csv file handling? (parsing it and adding the values to the database).
- show session files.
- edit, remove session files.

---

- same info and features as the report generated by the mobile app.
- It has a lot of charts.

---

- Integrating the classifier(s) - Interface.

---

- UI that allows the user to select the sessions file to send to the classifier.
- Prediction Results: Which results will be given to the users. It should refer the sessions files.
- List, edit, remove the predictions results.


---


### Django tutorial

https://www.youtube.com/watch?v=UmljXZIypDc

### Implemented
- Login and account management

### Django configuration

https://medium.com/python-pandemonium/better-python-dependency-and-package-management-b5d8ea29dff1

Requirements

```
certifi==2018.11.29
Django==2.1.7
django-crispy-forms==1.7.2
mysqlclient==1.4.2.post1
Pillow==6.0.0
pytz==2018.9
scikit-fuzzy==0.3.1
virtualenv==16.4.3
wincertstore==0.2
```

List dependencies
```
pip freeze > requirements.txt
```


Install depencencies
```
pip install -r requirements.txt
```

Actions will be similar to the one below:

Create a virtual environment $ python3 -m venv /path/to/new/virtual/env
Install packages using $pip install <package> command
Save all the packages in the file with $pip freeze > requirements.txt. Keep in mind that in this case, requirements.txt file will list all packages that have been installed in virtual environment, regardless of where they came from
Pin all the package versions. You should be pinning your dependencies, meaning every package should have a fixed version.
Add requirements.txt to the root directory of the project. Done.
Install project dependencies
When if you’re going to share the project with the rest of the world you will need to install dependencies by running $pip install -r requirements.txt

To find more information about individual packages from the requiements.txt you can use $pip show <packagename>. But how informative the output is?
	

### Info

There is already a template with charts and layout that we could use in our application. It is inside HTML folder. It has to be inegrated into Django project.



## Other things

- Project page text etc (Tommi)
	- Needs a new illustration (with small fixes)

---
---
---


## Documentation(links)

1. (movesole - pediatric therapists ) https://www.theseus.fi/bitstream/handle/10024/143390/Manselius_Lauri.pdf?sequence=1&isAllowed=y

2. (usability) https://www.theseus.fi/bitstream/handle/10024/155327/Ylikulju_Marko.pdf?sequence=1&isAllowed=y

3. (insole - hardware) https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3444133/


## Ground Rules

### Meetings

1.	Meetings will be held in English. If you don’t know a word, ask another member to help.
2.	Fixed meeting time:  Wednesdays. 14:30 – 16:00 at Agora.
3.	Working with the status report (15 mins) in each meeting.
4.	Each member informs asap, if he cannot attend the meeting.

### Communication

5.	We will use WeChat to keep everybody up-to-date with our project. 
	Each member will check out WeChat once per day.
6.	Inform the team and leader asap, if you cannot make a deadline, have troubles with a task given to you or a problem is coming up.
7.	On Tuesday, each member writes a message on WeChat asking doubts/questions or simply saying “no questions”.

### Norms

8.	We help each other. If you are stuck, ask your team members for help.
9.	To prevent escalation, we address conflicts early on, before we get frustrated.
10.	We use Trello to monitor the progress/milestones.

### Decision making

11.	Decisions will be made only after hearing the opinion of each relevant member.
12.	It is not acceptable to withhold your opinion (“whatever you all think…”) and then later be against the decision made.
