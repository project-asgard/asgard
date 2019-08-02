from options import LEVELS, DEGREES


# A tool to output a CSV file to use in excel or google sheets
class ExcelSheet:
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.sheet = []
        for _ in range(0, h):
            self.sheet.append([''] * w)

    def put(self, x, y, item):
        self.sheet[int(y)][int(x)] = item

    def get(self, x, y): return self.sheet[int(y)][int(x)]

    def __str__(self):
        result = ''
        for row in self.sheet:
            result += ', '.join(list(map(str, row))) + '\n'
        return result


# Generates a CSV sheet with time data
class TimeSheet(ExcelSheet):
    def __init__(self):
        super().__init__(2 + LEVELS[-1], DEGREES[-1])

        self.put(
            0, 0, 'PDE Time used'
        )

        self.put(1, 0, 'degree')
        for level in range(2, LEVELS[-1]+1):
            self.put(
                level, 0, f'level {level}'
            )

        for degree in range(2, DEGREES[-1]+1):
            self.put(
                1, degree-1, f'{degree}'
            )

    # Record the seconds elapsed for asgard for a level and degree
    def add_seconds(self, level, degree, seconds):
        if degree <= DEGREES[-1] and level <= LEVELS[-1]:
            self.put(level, degree - 1, seconds)

    # Returns the seconds for compute time at a given level and degree
    def get_seconds(self, level, degree):
        if degree <= DEGREES[-1] and level <= LEVELS[-1]:
            return self.get(level, degree - 1)
        return 0


# Generates a CSV sheet with memory data
# TODO: make MemorySheet's output spreadsheet size
# adapt to the data added to the spreadsheet
class MemorySheet(ExcelSheet):
    def __init__(self):
        super().__init__(2 + LEVELS[-1], DEGREES[-1])

        self.put(
            0, 0, 'PDE Actual Peak Mem Usage'
        )

        self.put(1, 0, 'degree')
        for level in range(2, LEVELS[-1]+1):
            self.put(
                level, 0, f'level {level}'
            )

        for degree in range(2, DEGREES[-1]+1):
            self.put(
                1, degree-1, f'{degree}'
            )

    # Add total and workspace mem usage for a level and degree
    def add_data(self, level, degree, total):
        if degree <= DEGREES[-1] and level <= LEVELS[-1]:
            self.put(level, degree - 1, total)

    # Returns the total and workspace memory usage at a given level and degree
    def get_data(self, level, degree):
        if degree <= DEGREES[-1] and level <= LEVELS[-1]:
            return self.get(level, degree - 1)
        return 0
