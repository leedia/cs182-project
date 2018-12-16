from unidecode import unidecode
import csv
import re
import sys

def main():

	# check arguments for correct usage
	if len(sys.argv) != 4 and len(sys.argv) != 5:
		print "Usage: datacleaning.py inputfile.csv outputfile.csv True/False on lowercase [number of rows to clean]"
		sys.exit(0)

	# open input file
	csvfilein = open(sys.argv[1], "r")
	filereader = csv.reader(csvfilein, delimiter=',')
	# open output file
	csvfileout = open(sys.argv[2], "w")
	filewriter = csv.writer(csvfileout, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

	# add label row
	filewriter.writerow(["ID", 'title', 'text', 'label'])


	linenum = 0
	for line in filereader:
		# if line looks good, begin decoding it; good = ID is in correct place, label is there, and title and text are not blank
		if linenum != 0 and is_number(line[0]) and (line[3] == "FAKE" or line[3] == "REAL") and (line[1] != "" and line[2] != ""):
			
			#get title and text and decode them in unicode-8
			text_string = line[2].decode('UTF-8')
			text_string = unidecode(text_string)
			title_string = line[1].decode('UTF-8')
			title_string = unidecode(title_string)

			# if requested, lowercase everything
			if(sys.argv[3] == "true" or sys.argv[3] == "True"):
				title_string = title_string.lower()
				text_string = text_string.lower()

			# strip to get rid of extra while space on the edges
			text_string = text_string.strip()

			# if the string is the right length
			if (len(text_string) > 2):

				# if requested, remove punctuation
				if(sys.argv[3] == "true" or sys.argv[3] == "True"):
					text_string = re.sub('([.,!?()])', r' \1 ', text_string)
					text_string = re.sub('\s{2,}', ' ', text_string)

				# write data out to output file
				filewriter.writerow([line[0], title_string, text_string, line[3]])

			# the string was not the expected length (due to a messy csv) so ignore this line
			else:
				pass
				# print "Rejected a line due to length"

		# if a smaller dataset is requested, exit after the line number max
		if len(sys.argv) == 5  and linenum > int(sys.argv[4]):
			break

		linenum = linenum + 1

	# close output files
	csvfilein.close()
	csvfileout.close()

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__== "__main__":
	main()