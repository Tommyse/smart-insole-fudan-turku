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
        fields, samples = get_data(filePath, delimiter)
        fieldsList.append(fields)
        samplesList.append(samples)
    return (fieldsList, samplesList)
