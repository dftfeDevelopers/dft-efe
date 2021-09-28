import Parser as parser

filename = "testParser.txt"
p = parser.Parser.fromFilename(filename)
key = "Val"
values = p.extractKeyValues(key)
print(key, values)
key = "Int" 
values = p.extractKeyValues(key, dtype='int')
print(key, values)
key = "Complex" 
values = p.extractKeyValues(key, dtype='complex')
print(key, values)
