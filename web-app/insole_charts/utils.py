import random
from insoleLib.columns import DataColumns

def generateColor():
    color = f'rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})'
    return color

def generateColors(n):
    colors = set()
    while len(colors) < n:
        color = generateColor()
        colors.add(color)
    
    return colors

def getNumberLabels(n, number=30):
    increment = 1
    if n <= number:
        increment = 1
    else:
        increment = n // number
    
    return [i for i in range(0, n+1, increment)]

class LinearContainer(object):

    def __init__(self, label, data, borderColor, fill=False):
        self.label = label
        self.data = data
        self.borderColor = borderColor
        self.fill = "true" if fill else "false" 
    
    def __eq__(self, obj):
        return isinstance(obj, LinearContainer) and obj.data == self.data and obj.label == self.label and obj.borderColor == self.borderColor and obj.fill == self.fill

    def __str__(self):
        print(f'{self.label} {self.borderColor} {self.fill} : {self.data}\n')

class ChartContainerFactory(object):

    @staticmethod
    def createForceLinearChartContainers(input_data):
        forceLinearChartContainers = []
        labels = [
            'S0_force', 'S1_force', 'S2_force', 'S3_force',
            'S4_force', 'S5_force', 'S6_force'
        ]

        allLabelData = ChartContainerFactory.getLabelData(input_data, labels)
        colors = generateColors(len(labels))
        fills = [False] * 10

        for label, data, color, fill in zip(labels, allLabelData, colors, fills):
            forceLinearChartContainers.append(LinearContainer(label, data, color, fill))
        return forceLinearChartContainers  

    @staticmethod
    def createStartTimeLinearChartContainers(input_data):
        forceLinearChartContainers = []
        labels = [
            "S0_start_time", "S1_start_time", "S2_start_time", "S3_start_time",
            "S4_start_time", "S5_start_time", "S6_start_time",
        ]

        allLabelData = ChartContainerFactory.getLabelData(input_data, labels)
        colors = generateColors(len(labels))
        fills = [False] * 10

        for label, data, color, fill in zip(labels, allLabelData, colors, fills):
            forceLinearChartContainers.append(LinearContainer(label, data, color, fill))
        return forceLinearChartContainers

    @staticmethod
    def createMaxTimeLinearChartContainers(input_data):
        forceLinearChartContainers = []
        labels = [
            "S0_max_time", "S1_max_time", "S2_max_time", "S3_max_time",
            "S4_max_time", "S5_max_time", "S6_max_time",
        ]

        allLabelData = ChartContainerFactory.getLabelData(input_data, labels)
        colors = generateColors(len(labels))
        fills = [False] * 10

        for label, data, color, fill in zip(labels, allLabelData, colors, fills):
            forceLinearChartContainers.append(LinearContainer(label, data, color, fill))
        return forceLinearChartContainers

    @staticmethod
    def createEndTimeLinearChartContainers(input_data):
        forceLinearChartContainers = []
        labels = [
           "S0_end_time", "S1_end_time", "S2_end_time", "S3_end_time",
           "S4_end_time", "S5_end_time", "S6_end_time",
        ]

        allLabelData = ChartContainerFactory.getLabelData(input_data, labels)
        colors = generateColors(len(labels))
        fills = [False] * 10

        for label, data, color, fill in zip(labels, allLabelData, colors, fills):
            forceLinearChartContainers.append(LinearContainer(label, data, color, fill))
        return forceLinearChartContainers

    @staticmethod
    def getLabelData(input_data, labels):

        data = []
        rowNumber = len(input_data)
        for label in labels:
            colIndex = DataColumns.getColumnIndex(label)
            colType = DataColumns.getColumnType(label)
            labelData = []
            for rowIndex in range(rowNumber):
                element = input_data[rowIndex][colIndex]
                if 'int' in colType:
                    element = int(element)
                labelData.append(element)
            data.append(labelData)
        
        return data
