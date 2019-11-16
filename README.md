# smart-insole-fudan-turku

# Classifier integration

Boosting selected as integrated classifier for the website.

Classifier requirements can be installed using:

```
pip install -r classifier_requirements.txt
```

## Code

- Pydoc for documentation

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

To find more information about individual packages from the requiements you can use $pip show <packagename>. 

---


## Documentation(links)

1. (movesole - pediatric therapists ) https://www.theseus.fi/bitstream/handle/10024/143390/Manselius_Lauri.pdf?sequence=1&isAllowed=y

2. (usability) https://www.theseus.fi/bitstream/handle/10024/155327/Ylikulju_Marko.pdf?sequence=1&isAllowed=y

3. (insole - hardware) https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3444133/
