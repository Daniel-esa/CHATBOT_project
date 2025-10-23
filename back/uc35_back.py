import pandas as pd


def load_spreadsheet_sheet(file):
    
    if file.name.endswith('.xlsx'):
        data = load_pdf(file)
    elif file.name.endswith('.xlsm'):
        data = load_docx(file)
    elif file.name.endswith('.xlsx'):
        data = load_txt(file)
    elif file.name.endswith('.csv'):
        return ["default"]
