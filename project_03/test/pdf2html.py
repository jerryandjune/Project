import os

filepath = 'D:/DockerDesktop/ShareFile'
filename = 'test2.pdf'
pdf2html = r'docker run -ti --rm -v {}:/pdf bwits/pdf2htmlex pdf2htmlEX --zoom 1.3 {}'

a = os.system(pdf2html.format(filepath, filename))

print(a)
