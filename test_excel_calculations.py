
import xlwings
import openpyxl 

# Load the workbook and change a cell
wb = openpyxl.load_workbook(r'Data/test.xlsx')
sheet = wb.active
sheet['A1'] = 30
wb.save(r'Data/test.xlsx')

# Open excel to do the caching
excel_app = xlwings.App(visible=False)
excel_book = excel_app.books.open(r'Data/test.xlsx')
excel_book.save()
excel_book.close()
excel_app.quit()

# Load the workbook and read updated values (that have been cached)
wb = openpyxl.load_workbook(r'Data/test.xlsx', data_only=True)
sheet = wb.active
updated_value = sheet['B1'].value
print(updated_value)