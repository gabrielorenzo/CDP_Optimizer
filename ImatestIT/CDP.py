import json
from datetime import datetime

fichero = r'C:\Users\F80lab1\Desktop\CDP_IM\Optimizer_CDP 1\Optimizer_CDP\optimizer\out\Results\Optimized_CDP_multitest.json'
out_file = r'C:\Users\F80lab1\Desktop\CDP_IM\Optimizer_CDP 1\Optimizer_CDP\CDP_Results.json'

def readCDP(file):
    with open(file) as f:
        data = json.load(f)
        cdp = data['jsonResults']['cdp']['cdp_values']  

    if cdp is not None:
        print(cdp)
        return cdp
    else:
        print("La clave 'cdp' no está presente en el archivo JSON.")
        return None

def AverageCDP():
    CDP_List = readCDP(fichero)
    
    if CDP_List is not None and len(CDP_List) >= 14:
        selected_values = CDP_List[2:14]
        CDP = sum(selected_values) / len(selected_values)
    else:
        print("La lista CDP no tiene suficientes valores.")
        return None

    print(CDP)
    return CDP


def main():
    AverageCDP()

if __name__ == "__main__":
    main()
