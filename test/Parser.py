import re

class Parser():
    '''
    @brief A simple parser class to match a given string in a file and return the
    associated values
    '''

    '''
    @brief Constructor
    @param filename Name of the file to parse
    '''
    def __init__(self, data, _is_direct_ = True):
        if _is_direct_:
            msg = '''__init__ of Parser cannot be called directly, use either
            fromFilename(filename) or fromString(dataString) methods'''
            raise ValueError(msg) 
        self.data = data 

    @classmethod
    def fromFilename(cls, filename):
        f = open(filename, "r")
        data = f.read()
        f.close()
        return cls(data, _is_direct_=False)

    @classmethod
    def fromString(cls, dataString):
        return cls(dataString, _is_direct_=False)


    def __checkbool__(self, x):
        if(x.lower()=='true' or x.lower()=='yes'):
            return True
        elif(x.lower()=='false' or x.lower()=='no'):
            return False
        else:
            raise ValueError('''Invalid string passed. Valid string include case insensitive
                             forms of either of the following: 'True', 'False', 'Yes', 'No'.
                             String passed is {}'''.format(x))

    def __getRegex__(self, dtype):
        if(dtype=='int'):
            return r"[+-]?\b(?<!\.)(?<![eE][+-])\d+(?!\.)\b" #r"(^|\s|,)+([+-]?\b(?<!\.)\d+(?!\.)\b)"
        if(dtype=='float' or dtype=='double' or dtype=='complex'):
            return r"[+-]?(\d+\.?\d*([eE][+-]?\d+)?)"
        #if(dtype=='complex'):
        #    regexCStyle = r"(\(\s*)([+-]?(\d+\.?\d*([eE][+-]?\d+)?))\s*,\s*([+-]?(\d+\.?\d*([eE][+-]?\d+)?)(\s*\))"
        #    regexPythonStyle = r"[+-]?(\d+\.?\d*([eE][+-]?\d+)?)\s*[+-]\s*[+-]?(\d+\.?\d*([eE][+-]?\d+)?)\j"
        #    return r"(" + regexCStyle + "|" + regexPythonStyle +")"
        if(dtype=='bool' or dtype=='string'):
            return r"\w+"

    def __dtypeConverter__(self, dtype):
        if(dtype=='int'):
            return int
        elif(dtype=='float'):
            return float
        elif(dtype=='bool'):
            return self.__checkBool__
        elif(dtype=='complex'):
            return complex
        else:
            raise ValueError('''Invalied string for dtype passed. Valid strings
                             are: 'int', 'float', 'complex', 'bool'. dtype
                             string passed is {}'''.format(dtype))

    def __checkValidDType__(self, dtype):
        if dtype not in ['int', 'float', 'bool', 'complex']:
            raise ValueError('''Invalied string for dtype passed. Valid strings
                             are: 'int', 'float', 'complex', 'bool'. dtype
                             string passed is {}'''.format(dtype))

    def __splitValues__(self, string, dtype, dtypeRegex):
        if dtypeRegex is None:
            dtypeRegex = self.__getRegex__(dtype)

        pattern = re.compile(dtypeRegex)
        matches = pattern.finditer(string)
        values = [] 
        for match in matches:
            values.append(match.group(0))
        if(dtype=='complex'):
            valuesComplex = []
            if len(values) % 2 != 0:
                raise RuntimeError("Detected unpaired value while parsing complex numbers.")
            for x,y in zip(values[::2],values[1::2]):
                connectingSign = ""
                if y[0] =='+':
                    connectingSign = "+"
                    y = y[1:]
                
                elif y[0]=='-':
                    connectingSign = '-'
                    y = y[1:]
                
                else:
                    connectingSign = '+'

                valuesComplex.append(x+connectingSign+y+"j")
            
            return valuesComplex
        else:
            return values

    def extractKeyValues(self, key, dtype='float', patternToCapture = r".*",
                         groupId = 0, dtypePattern = None):
        '''
        @brief This function takes a key (a string) and a regular-expression pattern
        to match after the key
        @param key: [Input] string to match
        @param dtype: [Input] A string specifying the datatype of the values to be
        returned. Default: 'float'. Allowed values: 'int', 'float', 'complex',
        'bool'.
        @param patternToCapture: [Input] A string containing the regular-expression
        to capture after the `key` string. Default: captures everything after
        `key` until end of line
        @param groupId: The group Id to capture from the pattern matched using
        `patterToCapture`. Default: 0 (i.e., uses the entire string matched
        after the `key`)
        @param dtypePattern: A regular expression that defines how to extract a
        value of `dtype` from a given string (i.e., how to identify int, float,
        complex, bool from a string). Default: None (i.e., we use in-built regular
        expressions based on the `dtype`. This should work for most cases while
        dealing with int, float, complex, and bool datatypes)
        @return list of list where the outer list is the index of the occurence
        of `key` in the file and the inner list contains the list of `dtype`
        values extracted from a given occurence of `key`
        '''
        dtypeLower = dtype.lower()
        self.__checkValidDType__(dtypeLower)
        converter = self.__dtypeConverter__(dtypeLower)
        patternToSearch = r"(" + key + r")" + r"(" + patternToCapture + r")"
        pattern = re.compile(patternToSearch)
        values = []
        matches = pattern.finditer(self.data)
        numMatches = len(pattern.findall(self.data))
        #numMatches = sum (1 for _ in matches)
        #if numMatches==0:
        #    msg = '''No match for key='{}' and the given patternToCapture='{}'
        #           found.'''.format(key,patternToCapture)
        #    raise ValueError(msg)
        for match in matches:
            #NOTE: The way patternToSearch is defined,
            # groupId 0 is the whole string
            # groupId 1 is the `key` match itself
            # groupId 2 is the `patternToCapture` match
            valueString = match.group(groupId+2)
            splitValues = self.__splitValues__(valueString, dtypeLower, dtypePattern)
            if len(splitValues)==0:
                msg = '''No matching {} value found in
                      string='{}' '''.format(dtypeLower, valueString)
                raise ValueError(msg) 

            values.append([converter(x) for x in splitValues])

        return values
