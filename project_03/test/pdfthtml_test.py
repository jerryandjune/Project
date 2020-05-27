# -*- coding: utf-8 -*-

# from pdfminer.layout import LAParams
# from pdfminer.converter import TextConverter
# from pdfminer.pdfinterp import PDFResourceioManager, PDFPageInterpreter
import os
import subprocess
import io

input_file = './test2.pdf'
print(os.path.abspath(input_file))

# # 方法1
# output_file = './cv1.html'
# subprocess.run(['./pdf2htmlEX/pdf2htmlEX.exe', input_file, output_file])

# # 方法2      pip install pdfminer.six  :pdf2txt.py -o output.html -t html file.pdf
# output_file ='./cv2.html'
# pdf2txt_file = './pdf2txt.py'
# command = 'python '+ os.path.abspath(pdf2txt_file) +' -o ' + os.path.abspath(
#     output_file) +' -t html ' + os.path.abspath(input_file)
# print (command)
# os.system(command) 


# # #方法3 extract text as bytes
# from pdfminer.pdfpage import PDFPage
# from io import BytesIO

# def pdf_to_text(path):
#     manager = PDFResourceManager()
#     retstr = BytesIO()
#     layout = LAParams(all_texts=True)
#     device = TextConverter(manager, retstr, laparams=layout)
#     filepath = open(path, 'rb')
#     interpreter = PDFPageInterpreter(manager, device)

#     for page in PDFPage.get_pages(filepath, check_extractable=True):
#         interpreter.process_page(page)

#     text = retstr.getvalue()

#     filepath.close()
#     device.close()
#     retstr.close()
#     return text

# text = pdf_to_text(input_file)
# output_file = './cv3.html'
# with open(output_file, 'wb') as f:
#     f.write(text) 

# #  方法4 pdfminer3k
# fp = open(input_file, 'rb')
# output_file = './cv4.html'
# from pdfminer.pdfparser import PDFParser, PDFDocument
# from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
# from pdfminer.converter import PDFPageAggregator
# from pdfminer.layout import LAParams, LTTextBox, LTTextLine

# parser = PDFParser(fp)
# doc = PDFDocument()
# parser.set_document(doc)
# doc.set_parser(parser)
# doc.initialize('')
# rsrcmgr = PDFResourceManager()
# laparams = LAParams()
# laparams.char_margin = 1.0
# laparams.word_margin = 1.0
# device = PDFPageAggregator(rsrcmgr, laparams=laparams)
# interpreter = PDFPageInterpreter(rsrcmgr, device)
# extracted_text = ''

# for page in doc.get_pages():
#     interpreter.process_page(page)
#     layout = device.get_result()
#     for lt_obj in layout:
#         if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
#             extracted_text += lt_obj.get_text()

# with open(output_file,"wb") as txt_file:
#     txt_file.write(extracted_text.encode("utf-8"))


# # #  方法5 pdfminer3
# from pdfminer3.pdfinterp import PDFPageInterpreter
# from pdfminer3.converter import TextConverter
# from pdfminer3.layout import LAParams
# from pdfminer3.pdfpage import PDFPage
# from pdfminer3.pdfinterp import PDFResourceManager


# def convert_pdf_to_txt(path):
#     rsrcmgr = PDFResourceManager()
#     retstr = io.StringIO()
#     codec = 'utf-8'
#     laparams = LAParams()
#     device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
#     fp = open(path, 'rb')
#     interpreter = PDFPageInterpreter(rsrcmgr, device)
#     password = ""
#     maxpages = 0
#     caching = True
#     pagenos = set()

#     for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,
#                                   password=password,
#                                   caching=caching,
#                                   check_extractable=True):
#         interpreter.process_page(page)



#     fp.close()
#     device.close()
#     text = retstr.getvalue()
#     retstr.close()
#     return text

# output_file = './cv5.html'
# with open(output_file,"w",encoding='utf8') as txt_file:
#     txt_file.write(convert_pdf_to_txt(os.path.abspath(input_file)))

# # #  方法6 pdfminer
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
# From PDFInterpreter import both PDFResourceManager and PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
# Import this to raise exception whenever text extraction from PDF is not allowed
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.converter import PDFPageAggregator

''' This is what we are trying to do:
1) Transfer information from PDF file to PDF document object. This is done using parser
2) Open the PDF file
3) Parse the file using PDFParser object
4) Assign the parsed content to PDFDocument object
5) Now the information in this PDFDocumet object has to be processed. For this we need
   PDFPageInterpreter, PDFDevice and PDFResourceManager
 6) Finally process the file page by page 
'''

output_file = './cv6.html'

my_file = os.path.abspath(input_file)
log_file = os.path.abspath(output_file)

password = ""
extracted_text = ""

# Open and read the pdf file in binary mode
fp = open(my_file, "rb")

# Create parser object to parse the pdf content
parser = PDFParser(fp)

# Store the parsed content in PDFDocument object
document = PDFDocument(parser, password)

# Check if document is extractable, if not abort
if not document.is_extractable:
	raise PDFTextExtractionNotAllowed
	
# Create PDFResourceManager object that stores shared resources such as fonts or images
rsrcmgr = PDFResourceManager()

# set parameters for analysis
laparams = LAParams()

# Create a PDFDevice object which translates interpreted information into desired format
# Device needs to be connected to resource manager to store shared resources
# device = PDFDevice(rsrcmgr)
# Extract the decive to page aggregator to get LT object elements
device = PDFPageAggregator(rsrcmgr, laparams=laparams)

# Create interpreter object to process page content from PDFDocument
# Interpreter needs to be connected to resource manager for shared resources and device 
interpreter = PDFPageInterpreter(rsrcmgr, device)

# Ok now that we have everything to process a pdf document, lets process it page by page
for page in PDFPage.create_pages(document):
	# As the interpreter processes the page stored in PDFDocument object
	interpreter.process_page(page)
	# The device renders the layout from interpreter
	layout = device.get_result()
	# Out of the many LT objects within layout, we are interested in LTTextBox and LTTextLine
	for lt_obj in layout:
		if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
			extracted_text += lt_obj.get_text()
			
#close the pdf file
fp.close()

# print (extracted_text.encode("utf-8"))
			
with open(log_file, "wb") as my_log:
	my_log.write(extracted_text.encode("utf-8"))
print("Done !!")