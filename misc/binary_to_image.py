from json import JSONDecoder
import Image
import json
import re

"""
Extract images from media.json
"""
def main():
	#users = json.load(open('..\json\users.json'), cls=ConcatJSONDecoder, encoding="ISO-8859-1")
	#print users[0]["userId"] 
	#print users[0]["description"]

	# Method to load multiple json objects
	media = json.load(open('..\json\media.json'), cls=ConcatJSONDecoder, encoding="ISO-8859-1")

	for idx, myMedia in enumerate(media):
		number = idx + 1
		im_data = media[idx]['image']['$binary'] # Key: $type, $binary

		# Convert string in base64 to image
		# http://stackoverflow.com/questions/2323128/convert-string-in-base64-to-image-and-save-on-filesystem-in-python
		fh = open( str(number) + ".jpeg", "wb")
		fh.write(im_data.decode('base64'))
		fh.close()

		text_file = open("medialist.txt", "a")
		text_file.write(".\\media\\"+str(number) + ".jpeg" + '\n')
		text_file.close()

	

"""
http://stackoverflow.com/questions/8730119/retrieving-json-objects-from-a-text-file-using-python
"""
#shameless copy paste from json/decoder.py
FLAGS = re.VERBOSE | re.MULTILINE | re.DOTALL
WHITESPACE = re.compile(r'[ \t\n\r]*', FLAGS)

class ConcatJSONDecoder(json.JSONDecoder):
    def decode(self, s, _w=WHITESPACE.match):
        s_len = len(s)

        objs = []
        end = 0
        while end != s_len:
            obj, end = self.raw_decode(s, idx=_w(s, end).end())
            end = _w(s, end).end()
            objs.append(obj)
        return objs

main()