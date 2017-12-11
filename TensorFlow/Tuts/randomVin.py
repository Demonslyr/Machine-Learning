def vinProb(inputStr):
    return inputStr +' -> '+ mapOnDelimitedSubstrings(inputStr,firstLastAndNumDistinct)

def mapOnDelimitedSubstrings(inputStr, mapFunc):
    if not len(inputStr):
        return '0'
    output=''
    subStr=''
    for x in range(len(inputStr)):
        if inputStr[x].isalpha():
            subStr += inputStr[x]
        else:
            output += mapFunc(subStr)+inputStr[x]
            subStr = ''
    if inputStr[-1:].isalpha():
        output += mapFunc(subStr)
    else:
        output += '0'
    return output

def firstLastAndNumDistinct(inputStr):
    if len(inputStr):
        return inputStr[0]+str(len(set(inputStr.lower())))+inputStr[-1]
    return '0'

def main():
    print(vinProb("hello-how=are*you*today?"))
    print(vinProb("-=**?"))
    print(vinProb("-how=*you*?"))
    print(vinProb("hellohowareyoutoday"))
    print(vinProb("*today"))
    print(vinProb("?"))
    print(vinProb(""))

if __name__ == "__main__":
    main()