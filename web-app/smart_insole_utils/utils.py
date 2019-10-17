import os

def get_files_from_directory(directory, default_label=True):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            files.append((directory + '/' + filename, default_label))
    return files
	
def get_data(filePath, delimiter=';'):
    '''
        TODO optimize the method.
    '''
    fields = []
    samples = []
    with open(filePath, 'r') as f:
        # remove the header of the file
        lines = f.readlines()
        
        # Getting the smart insole fields.
        fieldLine = lines[2]
        fields = fieldLine.split(delimiter)
        fields = list(map(str.strip, fields))

        # Getting the smart insole samples
        for sampleLine in lines [3:]:
            if sampleLine != fieldLine:
                fieldValues = sampleLine.split(delimiter)
                fieldValues = list(map(str.strip, fieldValues))
                samples.append(fieldValues)
                            
    return (fields, samples)

def get_data_from_files(filePaths, delimiter=';'):
    fieldsList = []
    samplesList = []
    for filePath in filePaths:
        if os.path.isfile(filePath) and filePath.endswith('.csv'):
            fields, samples = get_data(filePath, delimiter)
            fieldsList.append(fields)
            samplesList.append(samples)
    return (fieldsList, samplesList)

# Create a function called 'chunks' with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def combine(ls):
    combineList = []
    for l in ls:
        combineList += l
    return combineList