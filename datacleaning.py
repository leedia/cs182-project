from unidecode import unidecode
import csv
import re
import sys

def main():

	filename = "fake_or_real_news.csv"

	print sys.argv[0];
	print sys.argv;

	with open(filename, "r") as csvfilein:
		filereader = csv.reader(csvfilein, delimiter=',')

		with open(sys.argv[1], "w") as csvfileout:
			filewriter = csv.writer(csvfileout, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

			filewriter.writerow(["ID", 'title', 'text', 'label'])

			linenum = 0

			for line in filereader:


				if linenum != 0 and is_number(line[0]) and (line[3] == "FAKE" or line[3] == "REAL") and (line[1] != "" and line[2] != ""):
					



					# line[2].decode('UTF-8')

					text_string = line[2].decode('UTF-8')
					text_string = unidecode(text_string)

					title_string = line[1].decode('UTF-8')
					title_string = unidecode(title_string)

					if(sys.argv[2] == "true"):
						title_string = title_string.lower()
						text_string = text_string.lower()

					text_string = text_string.strip()

					if (len(text_string) > 2):

						if(sys.argv[3] == "true"):
							text_string = re.sub('([.,!?()])', r' \1 ', text_string)
							text_string = re.sub('\s{2,}', ' ', text_string)

						filewriter.writerow([line[0], title_string, text_string, line[3]])

					else:
						print "Rejected a line due to length"


					# line_string = line_string.replace("\u2018", "'").replace("\u2019", "'")

					


					#line_list = line_string.split(",")

					#print line_list

					#if(is_number(line_list[0])):
					#	print line_list[2]

					# print line_list

					# if linenum > 5:
					# 	break

				linenum = linenum + 1
					#alphanumeric characters only?
					#punctuation a separate word
					#case insesistive
					#clean spaces/returns/etc




def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__== "__main__":
	main()