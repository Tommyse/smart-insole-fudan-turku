# smart-insole-fudan-turku

# Classifier integration

Boosting selected as integrated classifier for the website.

## Code

- Using pydoc for documentation

## Data

- Data labeled as "Normal" and "Fall" based on collected sets

## Classifier

The classifier and data processing library is currently at **"classifier/jupyter notebooks/insoleLib"** and copied to website for use.

Various python files in **"classifier/jupyter notebooks"** are jupyter notebooks in alternate format. Visual studio code with python extension is recommended.

#%% starts a new cell in this format

### Tests

- Some of the test results are in the **"classifier/jupyter notebooks/test_results"**  folder.

- Data labels suffled and test many times + plots + compared to real results

- Those classifiers don't learn anything from the data (AUC always near 0.5). Meanwhile our real classifiers learn things from the data. (AUC around 0.9 or 0.8 usually)

### Notes

- Including steps with errors alters the accuracy of classifiers a bit

	- Seems to be a small change towards worse accuracy. Might be bigger problem when using data from multiple different persons.

- The implemented classifiers can only predict normal walking scenarios on a flat surface. Many things can throw the predictions off such as carrying something.

	- Similar scenarios like the training data.

- Some of the early data exploration code might not work anymore due to many changes to the code that it used.


---

## Webapp

### Features

The web app have the following features:

- login page [OK]
- registration page [OK]
- profile page [OK]
- login and logout functionalities

- upload session files.
- show in a table the data of a table file.
- Movesole csv file handling
- show session files.
- edit, remove session files.
- Make predictions.
- Visualize data into charts.
- See history patient.


---

- Integrating the classifier(s) - Interface.

---

- UI that allows the user to select the sessions file to send to the classifier.
- Prediction Results: Which results will be given to the users. It should refer the sessions files.
- List, edit, remove the predictions results.


---
### How to run the website
```
pip install -r website_requirements.txt
```
```
cd web-app
```

Initialize the DB:
```
python manage.py makemigrations
```
```
python manage.py migrate
```
Run DJANGO server:
```
python manage.py runserver
```

### Dependencies

List dependencies
```
pip freeze > website_requirements.txt
```


Install depencencies
```
pip install -r website_requirements.txt
```

### Django tutorial

https://www.youtube.com/watch?v=UmljXZIypDc


### Django configuration

https://medium.com/python-pandemonium/better-python-dependency-and-package-management-b5d8ea29dff1

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
